
import argparse
import glob
import os
import pathlib

import cv2
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# BM
# -a
# regression
# -d
# ../MoGe/out/british_museum
# -d_gt
# ../datasets/phototourism/phototourism/british_museum/set_100/depth_maps
# -i
# ../datasets/phototourism/phototourism/british_museum/set_100/images

class HypersimDS:

    def __init__(self, image_and_depth_gt, depth):
        """
        :param args: provide both image and depth
        """
        self.depth = depth
        self.image_and_depth_gt = image_and_depth_gt
        self.items = []

        # depths: ../MoGe/out/hypersim/ai_001_002
        # depth_gt and rgb example: ../Marigold/data/hypersim/train/ai_001_002
        # depth_gt and rgb example: depth_plane_cam_03_fr0098.png <-> rgb_cam_00_fr0000.png
        full_paths_rgb = glob.glob(os.path.join(self.image_and_depth_gt, "rgb_*.png"))
        for img_fp in full_paths_rgb:
            if not os.path.isfile(img_fp):
                continue
            key = pathlib.Path(img_fp).name[4:-4]
            depth_gt_fp = os.path.join(self.image_and_depth_gt, f"depth_plane_{key}.png")
            if not os.path.isfile(depth_gt_fp):
                continue
            depth_fp = os.path.join(self.depth, f"rgb_{key}.png.npz")
            if not os.path.isfile(depth_fp):
                continue
            self.items.append((depth_gt_fp, depth_fp, img_fp, key))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        dgt_fp, depth_fp, img_fp, key = self.items[index]
        depths_hw_gt = cv2.imread(dgt_fp, cv2.IMREAD_UNCHANGED) / 1000
        depths_hw = np.load(depth_fp)["depth"]
        img = cv2.imread(img_fp)
        return depths_hw_gt, depths_hw, img, key


class BritishMuseumDS:

    def __init__(self, image, depth, depth_gt):
        self.image = image
        self.depth = depth
        self.depth_gt = depth_gt
        self.items = []

        full_paths = glob.glob(os.path.join(self.depth_gt, "*.h5"))
        for dgt_fp in full_paths:
            key = pathlib.Path(dgt_fp).name[:-3]
            depth_fp = os.path.join(self.depth, f"{key}.jpg.npy")
            if not os.path.isfile(depth_fp):
                continue
            img_fp = os.path.join(self.image, f"{key}.jpg")
            if not os.path.isfile(img_fp):
                continue
            self.items.append((dgt_fp, depth_fp, img_fp, key))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        dgt_fp, depth_fp, img_fp, key = self.items[index]
        with h5py.File(dgt_fp, "r") as f:
            depths_hw_gt = f["depth"][:]
        depths_hw = np.load(depth_fp)
        img = cv2.imread(img_fp)
        return depths_hw_gt, depths_hw, img, key


def compute_normals(depths):
    every_other = 100
    win_w = win_h = 11
    assert win_w % 2 == 1
    assert win_h % 2 == 1

    H, W = depths.shape
    fx = 1000.0
    K = np.array([[fx,    0, (W - 1) / 2],
                  [0,    fx, (H - 1) / 2],
                  [0,     0,           1]])

    # depths_fl = depths.flatten()
    hom = np.ones(depths.shape + (3,), dtype=np.float32)
    hom[..., 0], hom[..., 1] = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H))
    K_invs = np.tile(np.linalg.inv(K)[None, None], (H, W, 1, 1))
    image_plane_3D = K_invs @ hom[..., None]
    points_3D = depths[..., None] * image_plane_3D[..., 0]

    # to torch and (3, 1, H, W)
    points_3D = torch.from_numpy(points_3D)[None].permute(3, 0, 1, 2)
    points_3D = torch.nn.functional.pad(points_3D, (win_h // 2, win_h // 2, win_w // 2, win_w // 2))

    unf = torch.nn.Unfold(kernel_size=(win_h, win_w))
    unfd = unf(points_3D)
    # centered
    # [3, win_h * win_w, H * W]
    centered = (unfd - (torch.sum(unfd, dim=1) / (win_h * win_w)).unsqueeze(dim=1))
    centered = centered.permute(2, 1, 0)

    # no weighting so far
    U, s_values, V = torch.svd(centered)

    normals = V[:, :, 2]
    normals = normals.reshape(H, W, 3)
    normals = normals / torch.norm(normals, dim=2).unsqueeze(dim=2)

    # not ideal
    normals[normals[..., 2] < 0] *= -1

    normals = normals.flatten(end_dim=-2).permute(1, 0)
    # every other n_th
    normals = normals[:, ::every_other]

    return normals


def linear_regression(depths_hw, depths_hw_gt):
    """
    :param depths_hw:
    :param depths_hw_gt:
    :return: a, b, residual
    """

    # filter on != np.inf
    x = depths_hw.flatten()
    y = depths_hw_gt.flatten()
    y_filtered = y[x != np.inf]
    x_filtered = x[x != np.inf].reshape(-1, 1)
    reg = LinearRegression().fit(x_filtered, y_filtered)

    # y_hat = depths_hw @ reg.coef_ + reg.intercept_
    assert reg.coef_.shape == (1,)
    # reg.intercept_ is just numpy number
    depths_hw_hat = depths_hw * reg.coef_ + reg.intercept_ - depths_hw_gt
    return reg.coef_.item(), reg.intercept_.item(), depths_hw_hat


def run_regression(ds):

    first_n = 5
    # for index in range(len(ds)):
    for index in range(first_n):

        depths_hw_gt, depths_hw, img, key = ds[index]
        scale, shift, residual_hw = linear_regression(depths_hw, depths_hw_gt)


        rows = 2
        cols = 4
        f, axs = plt.subplots(rows, cols, figsize=(7, 6))
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', labelsize=5)
        plt.suptitle(f"{key}.jpg: scale = {scale:.03f}, shift = {shift:.03f}")

        axs[0, 0].title.set_text("original")
        axs[0, 0].imshow(img)

        axs[0, 1].title.set_text("depth gt")
        axs[0, 1].imshow(depths_hw_gt)

        axs[0, 2].title.set_text("depth est")
        axs[0, 2].imshow(depths_hw)

        axs[0, 3].title.set_text("residual")
        axs[0, 3].imshow(residual_hw)

        normals = compute_normals(depths_hw)
        axs[1, 0].title.set_text("normals (est)")
        axs[1, 0].set_aspect('equal')
        axs[1, 0].plot(normals[0], normals[1], "r.", markersize=0.1)

        normals_gt = compute_normals(depths_hw_gt)
        axs[1, 1].title.set_text("normals (gt)")
        axs[1, 1].set_aspect('equal')
        axs[1, 1].plot(normals_gt[0], normals_gt[1], "r.", markersize=0.1)

        normals_res = compute_normals(residual_hw)
        axs[1, 2].title.set_text("normals (res)")
        axs[1, 2].set_aspect('equal')
        axs[1, 2].plot(normals_res[0], normals_res[1], "r.", markersize=0.1)

        # plt.savefig(f"{dir}/{color_file}.jpg", bbox_inches='tight')
        show = True
        if show:
            plt.show()
        plt.close()


def run_stats(ds):

    first_n = 5
    # for index in range(len(ds)):
    for index in range(first_n):
        depths_hw_gt, depths_hw, _, key = ds[index]
        scale, shift, _ = linear_regression(depths_hw, depths_hw_gt)
        print(f"{key}.jpg: scale = {scale}, shift = {shift}")


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--action", required=True, help="Action to take", choices=["regression", "all"])
    parser.add_argument("-d", "--depth", required=True, help="Depth dir / file path")
    parser.add_argument("-d_gt", "--depth_gt", required=True, help="Depth GT dir / file path")
    parser.add_argument("-i", "--image", required=True, help="Image dir / file path")
    args = parser.parse_args()
    # ds = BritishMuseumDS(args.image, args.depth, args.depth_gt)

    key = "ai_001_002"
    depths = f"../MoGe/out/hypersim/{key}"
    depth_gt_and_rgb = f"../Marigold/data/hypersim/train/{key}"
    ds = HypersimDS(depth_gt_and_rgb, depths)

    if args.action == "regression":
        run_regression(ds)
    elif args.action == "all":
        run_stats(ds)


if __name__ == "__main__":
    run()
