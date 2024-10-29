import os.path
import pathlib
from pathlib import Path

import numpy as np
import argparse

import pycolmap
from tqdm import tqdm


def get_data_file_key(image, conflict_names):
    depth_data_file_key = Path(image.name).name[:-4]
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


# TODO account for distortion
# TODO better split
# TODO depth in meters?
# TODO check the track length etc...
def run(in_dir,
        out_dir_data,
        out_dir_images,
        append,
        conflict_names,
        rel_out_dir,
        max_items=None,
        read_again=False,
        write=True):
    """
    How it works:

    * provide int/out_dir or scene (e.g. in => megascenes.../scene/... out => Marigold/data/sfm/{scene})
    * the .npz files contain keys 'uv' and 'camera_coords'

    * creates:

    Marigold/data/sfm/{scene}/
          ---> commons -> link to commons with images
          ---> val
               data --->
                    img1.npz
                    img2.npz
                    img3.npz
               val.txt
          ---> train
               data --->
                    img1.npz
                    img2.npz
                    img3.npz
               train.txt
          ---> eval
               data --->
                    img1.npz
                    img2.npz
                    img3.npz
               eval.txt

    :param in_dir: e.g., ../datasets/megascenes/{scene}/reconstruct/colmap/0
    :param out_dir_data: e.g., ../Marigold/data/sfm/data/{scene}  or ./data/sfm/one_off
    :param split:
    :param append:
    :param write:
    :param read_again:
    :return:
    """
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
        pycolmap.undistort_images()
        print("Reconstruction summary:")
        print(reconstruction.summary())

        valid_points = 0
        image_items = list(reconstruction.images.items())
        if max_items:
            image_items = image_items[:max_items]
        print(f"Number of images: {len(image_items)}")
        for _, image in tqdm(image_items):

            camera = reconstruction.camera(image.camera_id)
            cmodel = camera.model.name
            assert cmodel == "SIMPLE_RADIAL"
            # model camera_id width height f cx cy k
            camera_params = f"{cmodel} {camera.camera_id} {camera.width} {camera.height} {' '.join([str(p) for p in reconstruction.camera(image.camera_id).params])}"

            uv_l = []
            camera_coords_l = []
            # iterate through (valid 2D points)
            for p2d in image.points2D:

                # NOTE: p2d.point3D_id = 18446744073709551615 (== -1 with the right arithmetic)
                if p2d.point3D_id > 10 ** 11:
                    continue

                # world coords => camera coords
                xyz_w = reconstruction.points3D[p2d.point3D_id].xyz
                camera_coords = image.cam_from_world * xyz_w
                assert camera_coords[2] > 0

                uv_l.append(p2d.xy)
                camera_coords_l.append(camera_coords)

            depth_data_file_key = get_data_file_key(image, conflict_names)
            depth_data_path = os.path.join(np_out_dir, f"{depth_data_file_key}.npz")
            depth_data_rel_path = os.path.join(rel_out_dir, f"{depth_data_file_key}.npz")
            uv = np.array(uv_l)
            valid_points += len(uv_l)
            camera_coords = np.array(camera_coords_l)
            if write:
                np.savez_compressed(depth_data_path, uv=uv, camera_coords=camera_coords)
                rgb_path = str(pathlib.Path(image.name).relative_to(pathlib.Path("commons")))
                rgb_path = os.path.join(out_dir_images, rgb_path)
                ds_file.write(f"{rgb_path} {depth_data_rel_path} {camera_params}\n")
            # TEST
            if read_again:
                read_back = np.load(depth_data_path)
                uv_back = read_back["uv"]
                assert np.all(uv == uv_back)
                camera_coords_back = read_back["camera_coords"]
                assert np.all(camera_coords == camera_coords_back)

    print(f"Valid points: {valid_points}")


def run_for_scene(args):
    """
    ... or for one-off input dir
    :param args:
    :return:
    """

    if args.input_dir is None:
        for scene in args.scenes:
            print(f"Processing scene {scene} ...")
            rec_root = os.path.join(f"../datasets/megascenes", scene, "reconstruct/colmap/")
            reconstruction_dirs = sorted(list(Path(rec_root).glob("*")))
            reconstruction_dirs = [str(rec_dir) for rec_dir in reconstruction_dirs if rec_dir.is_dir()]
            dir_list = "\n".join([r for r in reconstruction_dirs])
            print(f"Found reconstruction directories: {dir_list}\n")

            conflict_names = set()
            output_dir_data = f"../Marigold/data/sfm/data/{scene}"
            output_dir_imgs = f"images/{scene}"
            rel_out_dir = str(pathlib.Path(output_dir_data).relative_to(pathlib.Path("../Marigold/data/sfm")))
            for i, rec_dir in enumerate(reconstruction_dirs):
                run(rec_dir,
                    output_dir_data,
                    output_dir_imgs,
                    i != 0,
                    conflict_names,
                    rel_out_dir,
                    args.max_items,
                    write=True,
                    read_again=True)

        print("create the symlinks")
        print("cd /datagrid/personal/vavravac/Marigold/data/sfm")
        for scene in args.scenes:
            print(f"ln -s ../../../../datasets/megascenes/{scene}/images/commons ./images/{scene}")

    else:
        run(args.input_dir,
            args.output_dir,
            args.output_dir,
            False,
            set(),
            None,
            args.max_items,
            write=True,
            read_again=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", help="scene (e.g., prasna_brana)", required=False)
    parser.add_argument("--input_dir", help="path to input model dir", required=False)
    parser.add_argument("--output_dir", help="path to output dir", required=False)
    parser.add_argument("--max_items", help="max images per reconstruction", type=int, required=False, default=None)
    args = parser.parse_args()

    if args.scenes is None and (args.input_dir is None or args.output_dir is None):
        raise ValueError("Missing --scene and (--input_dir or --output_dir)")

    if args.scenes is not None and (args.input_dir is not None or args.output_dir is not None):
        raise ValueError("--scene and (--input_dir or --output_dir) both present")

    run_for_scene(args)
