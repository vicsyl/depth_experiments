import os.path
from pathlib import Path

import numpy as np
import argparse

import pycolmap


def run(in_dir, out_dir, append):

    data_dir = os.path.join(out_dir, "data")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    ds_text_file = "train.txt"
    with open(os.path.join(out_dir, ds_text_file), "a" if append else "w") as ds_file:

        reconstruction = pycolmap.Reconstruction(in_dir)
        print(reconstruction.summary())

        image_to_read = 10
        for image_id, image in list(reconstruction.images.items())[:image_to_read]:

            data_xy = []
            data_d = []
            for p2d in image.points2D:
                # TODO magic -> see the APU
                if p2d.point3D_id > 10 ** 11:
                    continue
                xyz_w = reconstruction.points3D[p2d.point3D_id].xyz
                data_xy.append(p2d.xy)
                data_d.append((image.cam_from_world * xyz_w)[2])

            depth_data = {"xy": np.array(data_xy), "d": np.array(data_d)}

            # TODO invalid path
            rgb_file = image.name
            depth_data_path = os.path.join(data_dir, f"{Path(rgb_file).name[:-4]}")
            np.save(depth_data_path, depth_data)

            ds_file.write(f"{rgb_file}, {depth_data_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", help="scene (prasna_brana)", required=False)
    parser.add_argument("--input_dir", help="path to input model dir", required=False)
    parser.add_argument("--output_dir", help="path to output dir", required=False)
    args = parser.parse_args()

    if args.scene is None and (args.input_dir is None or args.output_dir is None):
        raise ValueError("Missing --scene and --input_dir and --output_dir")

    if args.output_dir is None:
        output_dir = f"../marigold/data/sfm/{args.sceme}"
    else:
        output_dir = args.output_dir

    if args.input_dir is None:
        rec_root = os.path.join(f"../datasets/megascenes", args.scene, "reconstruct/colmap/")
        rec_dirs = Path(rec_root).glob("*")

        print("Found {} reconstruction directories".format(len(rec_dirs)))
        # break
    else:
        # FIXME already exists
        rec_dirs = [args.input_dir]

    print(f"will call for in: {rec_dirs} and out: {output_dir}")

    # datasets/megascenes/{scene}/images/commons/
    # output_dir = "../marigold/data/sfm/{key}"
    for rec_dir in rec_dirs:
        run(rec_dir, output_dir, len(rec_dirs) != 1)
