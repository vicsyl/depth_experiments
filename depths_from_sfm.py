import os.path
from pathlib import Path

import numpy as np
import argparse

import pycolmap
from tqdm import tqdm


# FIXME account for distortion
# FIXME better split
# FIXME depth in meters?
# TODO calibration matrix
def run(in_dir, out_dir, split, append, write, read_again=False):
    """
    How it works:

    * provide int/out_dir or scene (in => megascenes.../scene/... out => Marigold/data/sfm/{scene})
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

    :param in_dir:
    :param out_dir:
    :param split:
    :param append:
    :param write:
    :param read_again:
    :return:
    """
    split_dir = os.path.join(out_dir, split)
    split_data_dir = os.path.join(out_dir, split, "data")
    Path(split_data_dir).mkdir(parents=True, exist_ok=True)

    ds_txt_file = os.path.join(split_dir, f"{split}.txt")
    print(f"Will write to {ds_txt_file}")
    with open(ds_txt_file, "a" if append else "w") as ds_file:

        reconstruction = pycolmap.Reconstruction(in_dir)
        print("Reconstruction summary:")
        print(reconstruction.summary())

        valid_points = 0
        image_items = list(reconstruction.images.items())
        print(f"Number of images: {len(image_items)}")
        for image_id, image in tqdm(image_items):

            uv_l = []
            camera_coords_l = []
            for p2d in image.points2D:
                # NOTE: p2d.point3D_id = 18446744073709551615 (== -1 with the right arithmetic)
                if p2d.point3D_id > 10 ** 11:
                    continue
                xyz_w = reconstruction.points3D[p2d.point3D_id].xyz
                uv_l.append(p2d.xy)
                # TODO depth in meters?
                camera_coords = image.cam_from_world * xyz_w
                assert camera_coords[2] > 0
                camera_coords_l.append(camera_coords)
                camera = reconstruction.camera(image.camera_id)
                cmodel = camera.model.name
                assert cmodel == "SIMPLE_RADIAL"
                # model camera_id width height f cx cy k
                camera_params = f"{cmodel} {camera.camera_id} {camera.width} {camera.height} {' '.join([str(p) for p in reconstruction.camera(image.camera_id).params])}"
                # camera_params_back = [float(i) for i in camera_params.split()[1:]]

            depth_data_path = os.path.join(split_data_dir, f"{Path(image.name).name[:-4]}.npz")
            uv = np.array(uv_l)
            valid_points += len(uv_l)
            camera_coords = np.array(camera_coords_l)
            if write:
                np.savez_compressed(depth_data_path, uv=uv, camera_coords=camera_coords)
                jpg_data_path = os.path.join("..", image.name)
                ds_file.write(f"{jpg_data_path} data/{Path(image.name).name[:-4]}.npz {camera_params}\n")
            if read_again:
                read_back = np.load(depth_data_path)
                uv_back = read_back["uv"]
                assert np.all(uv == uv_back)
                camera_coords_back = read_back["camera_coords"]
                assert np.all(camera_coords == camera_coords_back)
        print()

    print(f"Valid points: {valid_points}")


if __name__ == '__main__':

    # (input_dir, output_dir) => easy, for testing
    # scenes -> list of scenes
    #

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", help="scene (e.g., prasna_brana)", required=False)
    parser.add_argument("--input_dir", help="path to input model dir", required=False)
    parser.add_argument("--output_dir", help="path to output dir", required=False)
    args = parser.parse_args()

    if args.scene is None and (args.input_dir is None or args.output_dir is None):
        raise ValueError("Missing --scene and (--input_dir or --output_dir)")

    if args.output_dir is None:
        output_dir = f"../Marigold/data/sfm/{args.scene}"
    else:
        output_dir = args.output_dir
    print(f"Output dir: {output_dir}")

    if args.input_dir is None:
        rec_root = os.path.join(f"../datasets/megascenes", args.scene, "reconstruct/colmap/")
        rec_dirs = sorted(list(Path(rec_root).glob("*")))
        rec_dirs = [str(rec_dir) for rec_dir in rec_dirs if rec_dir.is_dir()]
        dir_list = "\n".join([r for r in rec_dirs])
        print(f"Found reconstruction directories for scene '{args.scene}': {dir_list}")
    else:
        rec_dirs = [args.input_dir]
        print(f"Will use reconstruction directory: {rec_dirs}")

    # TODO find a better split strategy
    val_split = int(len(rec_dirs) * 0.2)
    for i, rec_dir in enumerate(rec_dirs):
        split = "val" if i < val_split else "train"
        run(rec_dir, output_dir, split, len(rec_dirs) != 1, write=False, read_again=True)
