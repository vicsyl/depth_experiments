import glob
import os
import pathlib


def get_reconstructions():

    scenes = sorted([pathlib.Path(i).parent.name for i in glob.glob(f"../datasets/megascenes/*/reconstruct")])
    print(f"all scenes: {scenes}")

    for scene in scenes:
        rec_root = os.path.join(f"../datasets/megascenes", scene, "reconstruct/colmap/")
        reconstruction_dirs = sorted(list(pathlib.Path(rec_root).glob("*")))
        reconstruction_dirs = [str(rec_dir) for rec_dir in reconstruction_dirs if rec_dir.is_dir()]
        #dir_list = "\n".join([r for r in reconstruction_dirs])
        print(f"{scene}: found reconstruction directories: {reconstruction_dirs}\n")


if __name__ == "__main__":
    get_reconstructions()