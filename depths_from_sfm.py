import argparse
import glob
import os.path
import pathlib
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from tqdm import tqdm


# TODO - AIs:
# * clean the code! (in parallel to other tasks)
# * check/filter on the track length etc...

def get_data_file_key(colmap_image_name, conflict_names, old_file_keys):

    path = Path(colmap_image_name)
    if old_file_keys:
        depth_data_file_key = path.stem
    else:
        depth_data_file_key = f'{path.stem}_{path.suffix}'

    if conflict_names.__contains__(depth_data_file_key):
        for i in range(1000):
            depth_data_file_key_new = f"{depth_data_file_key}_{i}"
            if not conflict_names.__contains__(depth_data_file_key_new):
                depth_data_file_key = depth_data_file_key_new
                break
        if conflict_names.__contains__(depth_data_file_key):
            raise ValueError("cannot come up with a unique key")
    conflict_names.add(depth_data_file_key)
    return depth_data_file_key


def undistort_image(calib_matrix, dist_coeffs, uv, img_np):
    """
    :param calib_matrix:
    :param dist_coeffs:
    :param uv: [N, 2] - (n, x/y)
    :param img_np:
    :return: new_calib_matrix, undistorted_img_np, undistorted_uv
    """
    show = False
    log = False

    if log:
        print("DATA:")
        print(f"calib_matrix = {calib_matrix}")
        print(f"dist_coeffs = {dist_coeffs}")
        print(f"uv = {uv[:100]}")

    if len(uv) == 2:
        print("WARNING: ambiguity (len(uv) == 2)")
        Stats.stat_m["ambiguity (len(uv) == 2)"].append(1)
    else:
        Stats.stat_m["ambiguity (len(uv) == 2)"].append(0)

    # NOTE: validPixROI - Optional output rectangle that outlines all-good-pixels region in the undistorted image.
    # See roi1, roi2 description in stereoRectify
    new_calib_matrix, validPixROI = cv.getOptimalNewCameraMatrix(calib_matrix,
                                                         dist_coeffs,
                                                         (img_np.shape[1], img_np.shape[0]),
                                                         alpha=1.0,
                                                         centerPrincipalPoint=True)

    undistorted_img_np = cv.undistort(img_np, calib_matrix, dist_coeffs, None, new_calib_matrix)

    undistorted_uv = cv.undistortPoints(uv, calib_matrix, dist_coeffs, None, new_calib_matrix)
    undistorted_uv = undistorted_uv[:, 0]

    if show:
        plt.figure()
        plt.title("original image")
        plt.imshow(img_np)
        plt.show()

        plt.figure()
        plt.title("undistorted image")
        plt.imshow(undistorted_img_np)
        plt.show()

    return new_calib_matrix, undistorted_img_np, undistorted_uv


class Stats:
    stat_m = defaultdict(list)

    @staticmethod
    def print_stats():
        print("Stats:")
        for k, v in Stats.stat_m.items():
            print(f"{k}: mean: {np.mean(v):.3f}, {len(v)} items")


def get_camera_params(reconstruction, colmap_image, img_np):
    camera = reconstruction.camera(colmap_image.camera_id)
    cmodel = camera.model.name
    assert cmodel == "SIMPLE_RADIAL"

    params = camera.params
    dist_coeffs = (params[3], 0.0, 0.0, 0.0)

    camera_params = [cmodel, camera.camera_id]

    if camera.width == img_np.shape[0] and camera.height == img_np.shape[1]:
        # SWAPPED
        c_width = camera.height
        c_height = camera.width
        calib_matrix = np.array([[params[0], 0.0, params[2]],
                                 [0.0, params[0], params[1]],
                                 [0.0, 0.0, 1.0]])
        Stats.stat_m["swapped"].append(1)
        Stats.stat_m["inconsistent dimensions"].append(0)
        return c_width, c_height, calib_matrix, camera_params, dist_coeffs, True

    elif camera.width == img_np.shape[1] and camera.height == img_np.shape[0]:
        # OK
        c_width = camera.width
        c_height = camera.height
        calib_matrix = np.array([[params[0], 0.0, params[1]],
                                 [0.0, params[0], params[2]],
                                 [0.0, 0.0, 1.0]])

        Stats.stat_m["swapped"].append(0)
        Stats.stat_m["inconsistent dimensions"].append(0)
        return c_width, c_height, calib_matrix, camera_params, dist_coeffs, False

    else:
        Stats.stat_m["inconsistent dimensions"].append(1)
        print(f"WARNING: inconsistent dimensions: {camera.width=}, {camera.height=}, {img_np.shape[:2]=}")
        return None, None, None, None, None, None


def run_for_dir(in_dir,
                out_dir_data,
                out_dir_images,
                orig_out_dir_images,
                append: bool,
                rel_out_dir,
                undistort=True,
                max_items=None,
                read_again=False,
                write=True,
                old_file_keys=False):
    """
    FIXME update this
    How it works:

    * provide int/out_dir or scene (e.g. in => megascenes.../scene/... out => Marigold/data/sfm/{scene})
    * the .npz files contain keys 'uv' and 'camera_coords'

    * layout:

    {marigold_dir}/data/sfm/

         ./data/{scene}
            ./np/{img.npz} - 'uv' and 'camera_coords' keys
            ./data.txt - items

         ./images/{scene} - symlink to {megascenes_dir}/{scene_imgs}

         ./undistorted_images/{scene}

         ./{split_file_name}.txt

    :param in_dir: e.g., ../datasets/megascenes/{scene}/reconstruct/colmap/0
    :param out_dir_data: e.g., ../Marigold/data/sfm/data/{scene}  or ./data/sfm/one_off
    :param split:
    :param append:
    :param write:
    :param read_again:
    :return: error (in str) or None
    """

    conflict_names = set()

    assert not read_again or write
    np_out_dir = os.path.join(out_dir_data, "np")
    Path(np_out_dir).mkdir(parents=True, exist_ok=True)
    if rel_out_dir:
        rel_out_dir = os.path.join(rel_out_dir, "np")
    else:
        rel_out_dir = os.path.join(out_dir_data, "np")
    txt_file = os.path.join(out_dir_data, f"data.txt")

    print(f"{in_dir} => {np_out_dir}")
    print(f"rel out dir (np): {rel_out_dir}")
    print(f"out dir images: {out_dir_images}")
    print(f"will write to: {txt_file}")
    with open(txt_file, "a" if append else "w") as ds_file:

        reconstruction = pycolmap.Reconstruction(in_dir)
        print("Reconstruction summary:")
        print(reconstruction.summary())

        valid_points = 0
        image_items = list(reconstruction.images.items())
        if max_items:
            image_items = image_items[:max_items]
        print(f"Number of images: {len(image_items)}")

        marigold_rel_path = "../Marigold"
        # marigold_rel_path = "../marigold_private"

        for _, colmap_image in tqdm(image_items):

            print(f"processing image: {colmap_image.name}")

            ds_rgb_path = str(pathlib.Path(colmap_image.name).relative_to(pathlib.Path("commons")))
            ds_rgb_path = os.path.join(orig_out_dir_images, ds_rgb_path)
            eff_rgb_path = os.path.join(f"{marigold_rel_path}/data/sfm", ds_rgb_path)

            # IMHO this looks to the symlink
            if not os.path.exists(eff_rgb_path):
                if len(Log.output) == 0:
                    Log.output.append(f"In reconstruction dir {in_dir} missing path {eff_rgb_path}")
                Log.output.append(in_dir)
                return "missing"
            img_np = cv.imread(eff_rgb_path)
            c_width, c_height, calib_matrix, camera_params, dist_coeffs, swapped = get_camera_params(reconstruction,
                                                                                                     colmap_image,
                                                                                                     img_np)
            if c_width is None:
                print(f"WARNING skipping image")
                continue

            uv_l = []
            camera_coords_l = []
            # iterate through (valid 2D points)
            for p2d in colmap_image.points2D:

                # NOTE: p2d.point3D_id = 18446744073709551615 (== -1 with the right arithmetic)
                if p2d.point3D_id > 10 ** 11 or p2d.point3D_id == -1:
                    continue

                # world coords => camera coords
                xyz_w = reconstruction.points3D[p2d.point3D_id].xyz
                camera_coords = colmap_image.cam_from_world * xyz_w
                assert camera_coords[2] > 0

                uv_l.append(p2d.xy)
                camera_coords_l.append(camera_coords)

            # NOTE - generally I restrain from doing some filtering or normalization here just because it can
            #  always be done later / on the fly
            #  I can
            # a) filter on minimum number of points in image
            # b) if there is a conflict (uv[i1].int() == uv[i2].int()) - handle it later
            # c) some normalization - e.g., image size

            uv = np.array(uv_l)
            if uv.shape[0] == 0:
                Stats.stat_m["no uv points"].append(1)
                print("WARNING - no uv points, skipping image")
                continue
            else:
                Stats.stat_m["no uv points"].append(0)
            if swapped:
                # print(f"{uv.shape}")
                uv = uv[:, [1, 0]]

            camera_coords = np.array(camera_coords_l)

            print(f"max: uv: {uv[:, 0].max()},{uv[:, 1].max()} vs sizes: {c_width}x{c_height}")
            if uv.min() < 0.0 or uv[:, 0].max() > c_width - 1 or uv[:, 1].max() > c_height - 1:
                Stats.stat_m["out_of_image"].append(1)
                # print("WARNING does not fit")
                mask = np.min(uv, axis=1) >= 0
                mask = np.logical_and(mask, uv[:, 0] <= c_width - 1)
                mask = np.logical_and(mask, uv[:, 1] <= c_height - 1)
                uv = uv[mask]
                camera_coords = camera_coords[mask]
            else:
                Stats.stat_m["out_of_image"].append(0)

            if undistort:
                try:
                    calib_matrix, img_np, uv = undistort_image(calib_matrix, dist_coeffs, uv, img_np)
                    print(f"after undistortion max: uv: {uv[:, 0].max()},{uv[:, 1].max()} vs sizes: {c_width}x{c_height}")
                except Exception:
                    print(traceback.format_exc())
                    print("WARNING skipping image")
                    print("DATA:")
                    print(f"{calib_matrix=}")
                    print(f"{dist_coeffs=}")
                    print(f"{uv[:100]=}")
                    print(f"{eff_rgb_path=}")
                    Stats.stat_m["exception"].append(1)
                    continue
                Stats.stat_m["exception"].append(0)

                # model camera_id width height f cx cy
                camera_params = ["SIMPLE_PINHOLE",
                                 camera_params[1],
                                 img_np.shape[1],
                                 img_np.shape[0],
                                 calib_matrix[0, 0],
                                 calib_matrix[0, 2],
                                 calib_matrix[1, 2]]
            else:
                # NOTE this will soon be obsolete as everything for undistort == False
                # model camera_id width height f cx cy d
                camera_params = [camera_params[0],
                                 camera_params[1],
                                 img_np.shape[1],
                                 img_np.shape[0],
                                 calib_matrix[0, 0],
                                 calib_matrix[0, 2],
                                 calib_matrix[1, 2],
                                 dist_coeffs[0]]

            camera_params = " ".join([str(i) for i in camera_params])
            valid_points += len(uv_l)
            if write:
                depth_data_file_key = get_data_file_key(colmap_image.name, conflict_names, old_file_keys)
                depth_data_path = os.path.join(np_out_dir, f"{depth_data_file_key}.npz")
                np.savez_compressed(depth_data_path, uv=uv, camera_coords=camera_coords)

                if undistort:
                    ds_rgb_path = os.path.join(out_dir_images, f"{depth_data_file_key}.png")
                    save_to = os.path.join(f"{marigold_rel_path}/data/sfm", ds_rgb_path)
                    cv.imwrite(save_to, img_np)

                depth_data_rel_path = os.path.join(rel_out_dir, f"{depth_data_file_key}.npz")
                ds_file.write(f"{ds_rgb_path} {depth_data_rel_path} {camera_params}\n")

            if read_again:
                read_back = np.load(depth_data_path)
                uv_back = read_back["uv"]
                assert np.all(uv == uv_back)
                camera_coords_back = read_back["camera_coords"]
                assert np.all(camera_coords == camera_coords_back)

                camera_params_back = camera_params.split()
                if undistort:
                    assert camera_params[0] == "SIMPLE_PINHOLE"
                else:
                    assert camera_params[0] == "SIMPLE_RADIAL"

                items = [float(i) for i in camera_params[4:7]]
                # numpy, not torch
                calib_matrix = np.array([[items[0], 0.0, items[1]],
                                              [0.0, items[0], items[2]],
                                              [0.0, 0.0, 1.0]])
                print("debug me")

    Stats.stat_m["valid_points"].append(valid_points)
    return None


def run_for_scenes(scenes,
                   max_items,
                   undistort,
                   only_info,
                   megascenes_data_path,
                   marigold_rel_path,
                   old_file_keys,
                   write):

    def get_all_scenes():
        return sorted([pathlib.Path(i).parent.parent.name for i in glob.glob(f"{megascenes_data_path}/*/reconstruct/colmap")])

    if scenes == ["all"]:
        scenes = get_all_scenes()
        print(f"all scenes: {scenes}")
    elif scenes == ["new"]:
        all_scenes = get_all_scenes()
        ls_scenes = set(os.listdir(f"{marigold_rel_path}/data/sfm/data"))
        scenes = list(set(all_scenes) - ls_scenes)
        print(f"all scenes: {all_scenes}")
        print(f"new scenes: {scenes}")

    print("create the symlinks")
    print(f"pushd {marigold_rel_path}/data/sfm")
    print("mkdir -p ./images")
    for scene in scenes:
        print(f"ln -s ../../../../datasets/megascenes/{scene}/images/commons ./images/{scene}")
    print("popd")

    print("create dirs for undistorted imgs")
    for scene in scenes:
        print(f"mkdir -p {marigold_rel_path}/data/sfm/undistorted_images/{scene}")

    for scene in scenes:
        print(f"Processing scene {scene} ...")
        rec_root = os.path.join(megascenes_data_path, scene, "reconstruct/colmap/")
        reconstruction_dirs = sorted(list(Path(rec_root).glob("*")))
        reconstruction_dirs = [str(rec_dir) for rec_dir in reconstruction_dirs if rec_dir.is_dir()]
        dir_list = "\n".join([r for r in reconstruction_dirs])
        print(f"Found reconstruction directories: {dir_list}\n")

        output_dir_data = f"{marigold_rel_path}/data/sfm/data/{scene}"
        output_dir_imgs = f"undistorted_images/{scene}" if undistort else f"images/{scene}"
        orig_dir_imgs = f"images/{scene}"
        rel_out_dir = str(pathlib.Path(output_dir_data).relative_to(pathlib.Path(f"{marigold_rel_path}/data/sfm")))

        read_again = False
        print("arguments: ")
        print("reconstruction_dirs:")
        print("\n".join(reconstruction_dirs))
        print(f"{output_dir_data=}")
        print(f"{output_dir_imgs=}")
        print(f"{orig_dir_imgs=}")
        print(f"{rel_out_dir=}")
        print(f"{undistort=}")
        print(f"{max_items=}")
        print(f"{write=}")
        print(f"{read_again=}")
        print(f"{old_file_keys=}")

        if only_info:
            continue

        for i, rec_dir in enumerate(reconstruction_dirs):
            error = run_for_dir(rec_dir,
                                output_dir_data,
                                output_dir_imgs,
                                orig_dir_imgs,
                                i != 0,
                                rel_out_dir,
                                undistort=undistort,
                                max_items=max_items,
                                read_again=read_again,
                                write=write,
                                old_file_keys=old_file_keys)
            if error is not None:
                break # to the outer loop (= process next scene)


class Log:
    output = []


if __name__ == '__main__':

    s = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes",
                        nargs="+",
                        help="scenes (e.g., prasna_brana karluv most), or 'all' or 'new'",
                        required=False, default=["all"])
    parser.add_argument("--max_items", help="max images per reconstruction", type=int, required=False, default=None)
    parser.add_argument("--undistort", action=argparse.BooleanOptionalAction, required=False, default=True)
    parser.add_argument("--only_info", action=argparse.BooleanOptionalAction, required=False, default=False)
    parser.add_argument("--old_file_keys", action=argparse.BooleanOptionalAction, required=False, default=False)
    parser.add_argument("--write", action=argparse.BooleanOptionalAction, required=False, default=False)

    # new RCI
    # megascenes_data_path = "/mnt/personal/vavravac/datasets/megascenes"
    # old RCI
    # megascenes_data_path = "../datasets/megascenes"
    # local
    # megascenes_data_path = "../../Desktop/mount_rci/datasets/megascenes"
    parser.add_argument("--megascenes_data_path",
                        type=str,
                        required=False,
                        default="/mnt/personal/vavravac/datasets/megascenes")

    # old CMP
    # marigold_rel_path = "../Marigold"
    # RCI
    # marigold_rel_path = "../marigold_private"
    # local
    # marigold_rel_path = "../../Desktop/mount_rci/marigold_private"
    parser.add_argument("--marigold_rel_path",
                        type=str,
                        required=False,
                        default="../marigold_private")


    args = parser.parse_args()

    run_for_scenes(args.scenes,
                   args.max_items,
                   args.undistort,
                   args.only_info,
                   args.megascenes_data_path,
                   args.marigold_rel_path,
                   args.old_file_keys,
                   args.write)
    e = time.time()
    Stats.print_stats()
    print(f"Elapsed time: {e-s:.03f} s.")

    for line in Log.output:
        print(line)
