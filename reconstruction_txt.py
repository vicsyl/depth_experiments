import glob
import os
import pathlib
from pathlib import Path

import pycolmap


def find_all_scenes():
    return sorted([pathlib.Path(i).parent.name for i in glob.glob(f"../datasets/megascenes/*/reconstruct")])


def print_all_scenes():
    scenes = find_all_scenes()
    print(f"all scenes: {scenes}")


def write_all_ini_files():
    scenes = find_all_scenes()
    for scene in scenes:
        reconstruction_dirs = get_reconstrucion_dirs(scene)
        for rd in reconstruction_dirs:
            project_file = rd / "project.ini"
            with open(project_file, "r") as f:
                lines = f.readlines()
                print(f"{project_file=}")
                print("".join(lines))
                print()


def get_reconstrucion_dirs(scene):
    rec_root = os.path.join(f"../datasets/megascenes", scene, "reconstruct/colmap/")
    reconstruction_dirs = sorted(list(Path(rec_root).glob("*")))
    reconstruction_dirs = [rec_dir for rec_dir in reconstruction_dirs if rec_dir.is_dir()]
    return reconstruction_dirs


# FIXME, not done, the actual transform is still to be done
def transform_to_txt(scenes):

    for scene in scenes:
        print(f"Processing scene {scene} ...")

        reconstruction_dirs = get_reconstrucion_dirs(scene)

        dir_list = "\n".join([str(r) for r in reconstruction_dirs])
        print(f"Found reconstruction directories: {dir_list}")

        for rec_dir in reconstruction_dirs:
            out_dir = f"output/reconstructions/{scene}/{rec_dir.name}"
            print(f"reconstruction from {rec_dir} will be exported to: {out_dir}")
            reconstruction = pycolmap.Reconstruction(str(rec_dir))
            print("Reconstruction summary:")
            print(reconstruction.summary())
            # reconstruction.export(out_dir, output_type="txt")


if __name__ == "__main__":
    # print_all_scenes()
    # transform_to_txt(["arc_de_triomf_barcelona"])
    write_all_ini_files()
