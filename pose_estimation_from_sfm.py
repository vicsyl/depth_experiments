import argparse
import math
import os
import time
import cv2 as cv
import numpy as np
import pycolmap
import torch
from pyquaternion import Quaternion

from p3p_ransac import CVP3P_solver, P3P_RANSAC


def get_scene(rec_dir):
    return rec_dir.split(os.sep)[3]


class SfmLayout:

    def get_image_path(self, image_name):
        raise NotImplementedError()


class ETH3DSfmLayout(SfmLayout):

    @staticmethod
    def get_all_layouts(root_dir, dm_root_dir):
        rec_paths = [os.path.join(root_dir, f, "dslr_calibration_undistorted") for f in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, f, "dslr_calibration_undistorted"))]
        img_paths = [os.path.join(root_dir, f, "images") for f in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, f, "images/dslr_images_undistorted"))]
        scenes = [f for f in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, f, "images/dslr_images_undistorted"))]
        dm_paths = [os.path.join(dm_root_dir, f, "images") for f in os.listdir(dm_root_dir)
                    if os.path.isdir(os.path.join(dm_root_dir, f, "images/dslr_images_undistorted"))]
        return [ETH3DSfmLayout(*kw) for kw in zip(rec_paths, img_paths, dm_paths, scenes)]
        # return [ETH3DSfmLayout(r, i, d, s) for r, i, d, s in zip(rec_paths, img_paths, dm_paths, scenes)]

    def __init__(self,
                 rec_dir: str,
                 img_dir: str,
                 dm_dir: str,
                 scene: str):
        self.rec_dir = rec_dir
        self.img_dir = img_dir
        self.dm_dir = dm_dir
        self.scene = scene

    def get_image_path(self, image_name):
        return os.path.join(self.img_dir, image_name)

    def get_dm_path(self, image_name):
        return os.path.join(self.dm_dir, f"pred_{image_name[:-4]}.npy")


class MegascenesSfmLayout(SfmLayout):

    def __init__(self, rec_dir="../datasets/megascenes/arc_de_triomf_barcelona/reconstruct/colmap/0"):
        self.rec_dir = rec_dir
        self.scene = get_scene(self.rec_dir)

    @staticmethod
    def get_all_layouts():
        raise NotImplementedError()

    def get_image_path(self, image_name):
        return f"../datasets/megascenes/{self.scene}/images/{image_name}"


def get_calibration_matrix(camera: pycolmap.Camera, check_for_swap: bool = False, img_path: str = None) -> np.ndarray:
    # TODO SfMs other camera models (also with check_for_swap)

    assert not check_for_swap or img_path is not None
    # ETH3D
    assert camera.model == pycolmap.CameraModelId.PINHOLE

    if camera.model == pycolmap.CameraModelId.PINHOLE:

        if check_for_swap:
            img_np = cv.imread(img_path)
            assert img_np.shape[0] == camera.height
            assert img_np.shape[1] == camera.width

        return np.array([
            [camera.focal_length_x, 0, camera.principal_point_x],
            [0, camera.focal_length_y, camera.principal_point_y],
            [0, 0, 1.0]
        ])

    else:
        raise NotImplementedError(f"{camera.model} is not supported.")


def evaluate_pose(R_gt, t_gt, R, t):
    # TODO original precision
    # rot_err = Quaternion._from_matrix(R_gt.dot(R.T), rtol=1e-05, atol=1e-08).angle
    # TODO but this works (the error was due to R_gt!!)
    rot_err = Quaternion._from_matrix(R_gt.dot(R.T), rtol=1e-04, atol=1e-05).angle
    rot_err = math.fabs(rot_err * 180 / math.pi)
    # TODO check = math.acos((trace(R_gt.dot(R.T)) - 1) / 2) * 180 / math.pi

    pos_err = np.linalg.norm(R.T @ t - R_gt.T @ t_gt)
    return rot_err, pos_err


def pose_est_from_sfm(sfm_layout: ETH3DSfmLayout, ransac: P3P_RANSAC, only_n_imgs: int = None, device: torch.device = torch.device('cpu')):
    # TODO SfMs with distortions (or other camera types)

    reconstruction = pycolmap.Reconstruction(sfm_layout.rec_dir)
    image_ids = list(reconstruction.images.keys())

    # print(f"{len(image_ids)} images found in {sfm_layout.rec_dir}")
    if only_n_imgs:
        image_ids = image_ids[:only_n_imgs]

    rot_errs = []
    pos_errs = []
    for image_id in image_ids:

        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]

        image_path = sfm_layout.get_image_path(image.name)
        K = get_calibration_matrix(camera, check_for_swap=True, img_path=image_path)
        # print("Calibration matrix:")
        # print(K)

        corrs_2d = []
        corrs_3d = []
        for p2d in image.points2D:

            # NOTE: p2d.point3D_id = 18446744073709551615 (== -1 with the right arithmetic)
            if p2d.point3D_id > 10 ** 11 or p2d.point3D_id == -1:
                continue

            # world coords => camera coords
            xyz_w = reconstruction.points3D[p2d.point3D_id].xyz
            camera_coords = image.cam_from_world * xyz_w  # DEBUG cam_from_world
            assert camera_coords[2] > 0

            corrs_2d.append(np.array(p2d.xy.tolist() + [1.0]))
            corrs_3d.append(xyz_w)

        corrs_2d = np.array(corrs_2d)
        corrs_2d = (np.linalg.inv(K) @ corrs_2d.T).T
        x = torch.from_numpy(corrs_2d).to(device)
        corrs_3d = np.array(corrs_3d)
        X = torch.from_numpy(corrs_3d).to(device)

        dm = sfm_layout.get_dm_path(image.name)
        Rt_gt = image.cam_from_world.matrix()

        start = time.time()
        best_model_Rt, best_inliers = ransac.forward(X, x)
        best_model_Rt = best_model_Rt.detach().cpu().numpy()
        end = time.time()
        Data.el_time += (end - start)

        rot_err, pos_err = evaluate_pose(Rt_gt[:, :3], Rt_gt[:, 3:4], best_model_Rt[:, :3], best_model_Rt[:, 3:4])
        pos_errs.append(pos_err)
        rot_errs.append(rot_err)

    pos_errs = np.array(pos_errs)
    rot_errs = np.array(rot_errs)
    return rot_errs, pos_errs


class Data:
    el_time: float = 0.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--rec_dir', type=str, required=False, default="../datasets/megascenes/arc_de_triomf_barcelona/reconstruct/colmap/0")
    # "../datasets/megascenes/arc_de_triomf_barcelona/reconstruct/colmap/0"
    # ../datasets/megascenes/arc_de_triomf_barcelona/images/
    parser.add_argument('--only_n', type=int, required=False, default=None)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    args = parser.parse_args()

    # TODO set random seed to get deterministic results?
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    min_solver = CVP3P_solver(cv_flag=cv.SOLVEPNP_P3P)
    ransac = P3P_RANSAC(minimal_solver=min_solver,
                        batch_size=args.batch_size,
                        confidence=1 - 1e-6,
                        inl_th=0.0005)

    # m_layout = MegascenesSfmLayout(args.rec_dir)

    # m_layouts = ETH3DSfmLayout.get_all_layouts(root_dir="data/local_data/eth3d",
    #                                            dm_root_dir="data/local_data/eth3d_dm")

    m_layouts = ETH3DSfmLayout.get_all_layouts(root_dir="../datasets/eth_3d/multi_view_training_dslr_undistorted",
                                               dm_root_dir="../marigold_eval/output/eval/marigold_e2e_eth/datasets/eth_3d/multi_view_training_dslr_undistorted")

    for m_layout in m_layouts:
        rot_errs, pos_errs = pose_est_from_sfm(m_layout, ransac, only_n_imgs=args.only_n, device=device)
        np.savez(f"output/ransac/eth/{m_layout.scene}", rot_errs=rot_errs, pos_errs=pos_errs)

        back = np.load(f"output/ransac/eth/{m_layout.scene}.npz")
        print(f"{m_layout.scene}:")
        print("rot err [deg.],          pos err (C) [m.]")
        [print(f"{r}, {p}") for p, r in zip(back["rot_errs"], back["pos_errs"])]

    print(f"RANSAC took {Data.el_time:.03f} seconds")
    print(f"Min solver took {ransac.elapsed_time_in_solver:.03f} seconds")
