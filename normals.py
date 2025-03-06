import glob
import time
from pathlib import Path
from io_utils import *

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from gallery import HtmlGallery
import os


def vis_surface_normal(normals, mask=None) -> np.array:
    """
    Inspired by Metric3D
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    """
    n_img_L2 = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    n_img_norm = normals / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    if mask is not None:
        mask = mask.cpu().numpy().squeeze()
        normal_vis[~mask] = 0
    return normal_vis


def get_k(path):
    with open(os.path.join(path, "info.txt")) as fd:
        for l in fd.readlines():
            kv = l.split("=")
            if kv[0].strip() == "m_calibrationColorIntrinsic":
                l = [float(n) for n in kv[1].strip().split(" ")]
                K = np.array(l).reshape(4, 4)[:3, :3]
                return K

    assert False, "intrinsics not found"


def name_map(itr):
    r = {}
    for i in sorted(list(itr)):
        name = Path(i).name
        name = name[:name.rfind(".")]
        name = name[:name.rfind(".")]
        r[name] = i
    return r


def create_html_gallery(apt, room):
    dir = f"{OUT_DATA_DIR}/gallery/{apt}_{room}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    HtmlGallery(columns_by_width=True).write_gallery_file(in_folder=dir,
                                                          target_folder=dir,
                                                          width_pc=19)


def run():
    for apt, room in APT_ROOMS:
        if JUST_REGENERATE:
            mses, coeffs, shifts = [], [], []
        else:
            mses, coeffs, shifts = run_apt_room(apt, room, show=False)
        if LocalConf.WRITE_FILES:
            create_hist(apt, room, mses, coeffs, shifts, show=False)
            create_html_gallery(apt, room)

    if LocalConf.WRITE_FILES:
        create_hist_gallery()


def create_hist_gallery():
    dir = f"{OUT_DATA_DIR}/histograms"
    HtmlGallery().write_gallery_file(in_folder=dir,
                                     target_folder=dir,
                                     width_pc=90)


def create_hist(apt, room, mses, coeffs, shifts, show=True):
    LIN_SPACE_POINTS = 71  # was 123

    data_file = f"{OUT_DATA_DIR}/histograms/data_{apt}_{room}.npz"
    Path(data_file).parent.mkdir(parents=True, exist_ok=True)
    if not JUST_REGENERATE:
        np.savez(data_file, apt=apt, room=room, mses=mses, coeffs=coeffs, shifts=shifts)
    # else:
    loaded = np.load(data_file)
    mses = loaded['mses']
    coeffs = loaded['coeffs']
    shifts = loaded['shifts']

    plt.figure()
    plt.title(f"Scale coefficients for {apt}/{room}")
    plt.xlabel("no unit")
    plt.ylabel("# data points")

    # max_dist_value = 100
    # coeffs = np.clip(coeffs, a_min=0.0, a_max=coeffs.max())
    bins = np.linspace(0, coeffs.max() + 0.01, LIN_SPACE_POINTS)
    plt.hist(coeffs, bins, alpha=0.5)

    path = f"{OUT_DATA_DIR}/histograms/{apt}_{room}.coeffs.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.title(f"Shifts for {apt}/{room}")
    plt.xlabel("shift[m.]")
    plt.ylabel("# data points")

    # shifts = np.clip(coeffs, a_min=0.0, a_max=shifts.max())
    bins = np.linspace(shifts.min() - 0.01, shifts.max() + 0.01, LIN_SPACE_POINTS)
    plt.hist(shifts, bins, alpha=0.5)

    path = f"{OUT_DATA_DIR}/histograms/{apt}_{room}.shifts.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.title(f"MSE of linear regression for {apt}/{room}")
    plt.xlabel("MSE [m^2]")
    plt.ylabel("# data points")

    # mses = np.clip(coeffs, a_min=0.0, a_max=mses.max())

    bins = np.linspace(0, mses.max() + 0.01, LIN_SPACE_POINTS)
    plt.hist(mses, bins, alpha=0.5)

    path = f"{OUT_DATA_DIR}/histograms/{apt}_{room}.mses.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close()


def read_glob_map(glb, label="data"):
    data = set(glob.glob(glb))
    data_map = name_map(data)
    print(f"{label} length: {len(data_map)}")
    return data_map


def run_apt_room(apt, room, show=True):
    print(f"Running apt {apt} room {room}")

    ts_root = f"../twelve_scenes/data/ds/{apt}/{room}"
    ts_in = f"{ts_root}/data"

    K = get_k(ts_root)
    K_inv = np.linalg.inv(K)
    print(f"K = {K}")

    color_files = read_glob_map(f"{ts_in}/*.color.jpg")
    depth_files = read_glob_map(f"{ts_in}/*.depth.png")

    # monodepth / megadepth
    # FIXME rename
    # mge_in = f"./data/work_ts/{apt}_{room}"
    mge_in = f"{EST_IN_DIR}/{apt}_{room}"

    depth_ests = read_glob_map(f"{mge_in}/*_original_depth.png")
    up_normals = read_glob_map(f"{mge_in}/*_upscaled_normal.png")


    mses = []
    coeffs = []
    shifts = []
    color_files = list(color_files.items())[:MAX_FILES]
    for idx, (color_file, color_file_path) in enumerate(color_files):

        if (idx + 1) % 100 == 0:
            print(f"{idx}/{len(color_files)}")

        depth_gt_file = depth_files[color_file]
        depth_est = depth_ests[color_file]
        # depth_est_v = depth_est_vs[color_file]

        color_img = cv.imread(color_file_path)
        # depth_est_img = np.load(depth_est)
        depth_est_img = cv.imread(depth_est, cv.IMREAD_ANYDEPTH)
        # Metric3D
        depth_est_img = depth_est_img / 1000.0
        print(f"{color_file} depth est: max: {depth_est_img.max()}, min: {depth_est_img.min()}")

        # normals
        # up_normal = cv.imread(up_normals[color_file], cv.IMREAD_UNCHANGED)
        up_normal = read_data_png(up_normals[color_file], scale=PngConst.normal_scale, bias=PngConst.normal_bias)
        up_normal_vis = vis_surface_normal(up_normal)

        # # FIXME: the problem is with the flattenint I think ...
        # # X_1: point cloud
        def get_norms_from_depth(depths):
            ys, xs = np.mgrid[:depths.shape[0], :depths.shape[1]]
            ys = ys.flatten()
            xs = xs.flatten()
            X_1 = np.vstack([xs, ys, np.ones_like(xs)])
            X_1 = K_inv @ X_1
            X_1 = X_1 * depths.flatten()
            X_1 = X_1.T.reshape(depths.shape[0], depths.shape[1], -1)
            # # FIXME doesn't make sense
            # # x_test = K @ X_1
            # # test = x_test - depths
            # # print(f"test depth (backproject / project): {test}")
            #
            # # induced norms from X_1
            X_diff_right = X_1[:-1, 1:] - X_1[:-1, :-1]
            X_diff_down = X_1[1:, :-1] - X_1[:-1, :-1]
            norms = np.cross(X_diff_down, X_diff_right)
            norms = norms / np.linalg.norm(norms, axis=2, keepdims=True)
            return norms

        norms = get_norms_from_depth(depth_est_img)
        norms_diff = norms - up_normal[:-1, :-1]

        def contrast(a, c=3):
            m = a.mean(axis=(0, 1))
            return m + c * (a - m)

        norms_diff = contrast(norms_diff, c=3)

        norms = vis_surface_normal(norms)
        # norms_diff = vis_surface_normal(norms_diff)
        print(f"test normals (from depth / directly): {norms_diff}")

        depth_gt_img = cv.imread(depth_gt_file, cv.IMREAD_ANYDEPTH)
        depth_gt_img = depth_gt_img / 1000.0
        # FIXME - TODO
        print(f"gt cropped: {(depth_gt_img < 0.1).sum().item()} out of {depth_gt_img.flatten().shape[0]}")
        depth_gt_img[depth_gt_img < 0.1] = depth_gt_img.mean()
        print(f"{color_file} depth gt: max: {depth_gt_img.max()}, min: {depth_gt_img.min()}")
        depth_gt_img = cv.resize(depth_gt_img, depth_est_img.shape[::-1], interpolation=cv.INTER_LINEAR)

        # depth_est_v_img = cv.imread(depth_est_v)

        y = depth_gt_img.flatten()
        x = depth_est_img.flatten().reshape(-1, 1)
        reg = LinearRegression().fit(x, y)

        y_hat = x @ reg.coef_ + reg.intercept_
        residual = y - y_hat
        mse = (residual ** 2).sum() / residual.shape[0] / 2
        coeff = reg.coef_.item()
        shift = reg.intercept_.item()

        adj_depth = y_hat.reshape(depth_est_img.shape[0], depth_est_img.shape[1])
        adj_norms = get_norms_from_depth(adj_depth)
        adj_norms_diff = adj_norms - up_normal[:-1, :-1]
        adj_norms_diff = contrast(adj_norms_diff, c=3)
        adj_norms = vis_surface_normal(adj_norms)

        mses.append(mse)
        coeffs.append(coeff)
        shifts.append(shift)

        rows = 2
        cols = 4
        f, axs = plt.subplots(rows, cols, figsize=(12, 6))
        plt.suptitle(f"{color_file}: mse={mse:.04f}, coeff={coeff:.04f}, shift={shift:.04f}")

        axs[0, 0].title.set_text("original")
        axs[0, 0].imshow(color_img)
        axs[0, 1].title.set_text("depth GT")
        axs[0, 1].imshow(depth_gt_img)
        axs[1, 0].title.set_text("depth estimate")
        axs[1, 0].imshow(depth_est_img)
        axs[1, 1].title.set_text("ns from model")
        axs[1, 1].imshow(up_normal_vis)
        axs[0, 2].title.set_text("ns from depths")
        axs[0, 2].imshow(norms)
        axs[1, 2].title.set_text("ns diffs")
        axs[1, 2].imshow(norms_diff)
        axs[0, 3].title.set_text("ns adj depths")
        axs[0, 3].imshow(adj_norms)
        axs[1, 3].title.set_text("ns adj diffs")
        axs[1, 3].imshow(adj_norms_diff)

        dir = f"{OUT_DATA_DIR}/gallery/{apt}_{room}"
        Path(dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{dir}/{color_file}.jpg", bbox_inches='tight')
        if show:
            plt.show()
        plt.close()


    with open(f"{OUT_DATA_DIR}/{apt}_{room}_stats.txt", "w") as f:
        f.write("# color_file, mse, coeff, shift \n")
        for color_file, mse, coeff, shift in zip(color_files, mses, coeffs, shifts):
            f.write(f"{color_file[0]}, {mse:.04f}, {coeff:.04f}, {shift:04f}\n")

    return np.array(mses), np.array(coeffs), np.array(shifts)


class LocalConf:
    WRITE_FILES = True

# FIXME: argparse, using these as default

# OUT_DATA_DIR = "./data/normals_metric_3d_K_new"
# OUT_DATA_DIR = "./data/normals_metric_3d_no_K"
OUT_DATA_DIR = "./data/normals_metric_3d_K_normals_big"
# OUT_DATA_DIR = "./data/normals_metric_3d_rm_local"
MAX_FILES = 100
APT_ROOMS = [('apt1', 'kitchen'), ('apt1', 'living'), ('apt2', 'bed'), ('apt2', 'kitchen'), ('apt2', 'living'),
             ('apt2', 'luke'), ('office2', '5a'), ('office2', '5b')]
# APT_ROOMS = [('apt1', 'kitchen'), ('apt1', 'living')]
# APT_ROOMS = [('apt1', 'kitchen')]

# EST_IN_DIR = f"../Metric3D/show_dirs/vit.raft5.giant2/20240807_015734/vis" # K
# EST_IN_DIR = f"../Metric3D/show_dirs/vit.raft5.giant2/20240807_103753/vis" # no K
# FIXME group this into configs...
# EST_IN_DIR = f"../Metric3D/show_dirs/vit.raft5.small/20240903_232732/vis"  # K small
EST_IN_DIR = f"../Metric3D/show_dirs/vit.raft5.giant2/20240904_000023/vis"  # K big, rich

JUST_REGENERATE = False

if __name__ == "__main__":
    start = time.time()
    run()
    end = time.time()
    print(f"Done, took {(end - start):.03f} seconds")
