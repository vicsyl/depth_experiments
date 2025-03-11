import glob
import logging
import os.path
import pathlib
import random

import numpy as np
import argparse

from tqdm import tqdm
from sfm_dataset.utils import config_logging


# FIXME centralize
def get_line_items(line):
    rgb_rel_path = line[0]
    npy_rel_path = line[1]

    # model camera_id width height f cx cy
    camera_model = line[2]
    assert camera_model == "SIMPLE_PINHOLE"
    camera_id = int(line[3])

    # width height f cx cy
    camera_params = [float(i) for i in line[4:]]

    calib_matrix = np.array([[camera_params[2], 0.0, camera_params[3]],
                                 [0.0, camera_params[2], camera_params[4]],
                                 [0.0, 0.0, 1.0]])[None]

    w, h = camera_params[:2]

    return w, h, camera_model, calib_matrix, rgb_rel_path, npy_rel_path


# ('sacre_couer', 14)
# ('tycho_brahe_johannes_kepler_prague', 15)
# ('letohradek_kralovny_anny', 29)
# ('saint_ignatius_church_prague', 51)
# ('church_of_the_sacred_heart_prague', 104)
# ('arc_de_triomf_barcelona', 227)
# ('prasna_brana', 388)
# ('sydney_opera', 617)
# ('statue_of_liberty', 2171)
# ('Santa_Maria_del_Fiore_Florence', 2548)
# ('karluv_most', 2643)
# ('Hagia_Sophia', 2871)
# ('Brooklyn_Bridge', 2881)
# ('Old_Town_Hall_Prague', 4293)
# ('sagrada_familia', 5166)
# ('Dom_Luis_I_bridge_Porto', 5838)
# ('notre_dame_paris', 7777)
def get_scenes_info(scenes=["all"]):

    # CMP
    # marigold_rel_path = "../Marigold"
    # RCI
    # marigold_rel_path = "../marigold_private"
    marigold_rel_path = "../../Desktop/mount_rci/marigold_private"

    if scenes[0].lower() == "all":
        paths = sorted(list(glob.glob(f"{marigold_rel_path}/data/sfm/data/*/data.txt")))
        scenes_paths = [(pathlib.Path(p).parent.name, p) for p in paths]
    else:
        scenes_paths = [(scene, f"{marigold_rel_path}/data/sfm/data/{scene}/data.txt") for scene in scenes]

    m = []
    for scene, path in scenes_paths:
        with open(path, "r") as f:
            l = len(f.readlines())
            m.append((scene, l))

    m = sorted(m, key=lambda x: x[1])
    print("\n".join([str(i) for i in m]))

    return scenes_paths


def generate_splits_for_scene(splits_scenes_limits,
                              marigold_rel_path,
                              max_per_scene,
                              prefix,
                              shuffle,
                              max_dimension,
                              min_points):
    """
    Opens all {scene}/data.txt and just copy the lines to {prefix}_{splits}.txt, BUT
        - if maximal dimension of the image is < max_dimension
        - if there are at leas min_points points
    @param scenes_train:
    @param scenes_eval:
    @param scenes_test:
    @param max_per_scene:
    @param prefix:
    @param shuffle:
    @param max_dimension:
    @param min_points:
    @return:
    """

    dataset_dir = os.path.join(marigold_rel_path, "data/sfm")

    for split, scenes, split_limit in splits_scenes_limits:
        logging.info(f"processing split: {split} - scenes: {scenes}")

        all_lines = []
        paths = [f"{marigold_rel_path}/data/sfm/data/{scene}/data.txt" for scene in scenes]
        for path, scene in zip(paths, scenes):
            logging.info(f"processing scene: {scene}")

            with open(path, "r") as f:
                lines = f.readlines()

            if max_per_scene:
                lines = lines[:max_per_scene]

            for line in tqdm(lines):

                line_in = line.strip().split(" ")
                line_in = [l.strip() for l in line_in]
                w, h, camera_model, calib_matrix, rgb_rel_path, npy_rel_path = get_line_items(line_in)

                # maximal dimension
                if max(w, h) > max_dimension:
                    continue

                npy_path = os.path.join(dataset_dir, npy_rel_path)
                arrays = np.load(npy_path, allow_pickle=True)

                # uv: [self.max_items, 2 = (x, y)]
                uv = arrays["uv"]

                mmin = np.min(uv, axis=1)

                # FIXME: dataset generation
                uv[mmin >= 0]
                if uv.shape[0] < min_points:
                    continue

                all_lines.append(line)

        if shuffle:
            random.shuffle(all_lines)

        if split_limit:
            all_lines = all_lines[:split_limit]

        with open(os.path.join(dataset_dir, f"{prefix}_{split}.txt"), "w") as fw:
            fw.write(f"# This is {split} for {max_per_scene=}, {max_dimension=}, {min_points=}, {shuffle=}\n")
            fw.write("".join(all_lines))
            fw.write("\n")


def run_for_scene_random(scenes, max_per_scene, prefix):
    """
    old - not use anymore
    opens all {scene}/data.txt and just copy the lines to {prefix}_{splits}.txt
    @param scenes:
    @param max_per_scene:
    @param prefix:
    @return:
    """

    # CMP
    # marigold_rel_path = "../Marigold"
    # RCI
    marigold_rel_path = "../marigold_private"

    if scenes[0].lower() == "all":
        paths = sorted(list(glob.glob(f"{marigold_rel_path}/data/sfm/data/*/data.txt")))
    else:
        paths = [f"{marigold_rel_path}/data/sfm/data/{scene}/data.txt" for scene in scenes]

    print("Paths:\n", "\n".join(paths))

    # AI: split so that e.g., a scene is only in one split
    split_map = {"train": 0.8, "eval": 0.1, "test": 0.1}
    assert np.isclose(sum([split_map[k] for k in split_map.keys()]), 1.0)
    print(f"input split map: {split_map}")

    cum = 0.0
    for k in split_map.keys():
        cum += split_map[k]
        f = open(os.path.join(f"{marigold_rel_path}/data/sfm", f"{prefix}_{k}.txt"), "w")
        split_map[k] = (cum, f)

    print(f"cumulative map with file handlers: {split_map}")

    for path in paths:
        print(f"reading {path}")
        with open(path, "r") as f:
            lines = f.readlines()
            if max_per_scene is not None:
                lines = lines[:max_per_scene]
            for line in tqdm(lines):
                r = random.random()
                for split in split_map.keys():
                    if r <= split_map[split][0]:
                        split_map[split][1].write(line)
                        break

    for split in split_map.keys():
        split_map[split][1].close()


def main():

    config_logging()

    # scenes_paths = get_scenes_info()
    # return

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, help="splits prefix (i.e. all)", required=True)
    parser.add_argument("--scenes_train",
                        nargs="+",
                        help="train scenes (e.g., 'prasna_brana karluv_most')",
                        required=False,
                        default=None)
    parser.add_argument("--scenes_eval",
                        nargs="+",
                        help="eval scenes (e.g., 'prasna_brana karluv_most')",
                        required=False,
                        default=None)
    parser.add_argument("--scenes_test",
                        nargs="+",
                        help="test scenes (e.g., 'prasna_brana karluv_most')",
                        required=False,
                        default=None)

    parser.add_argument("--max_per_scene", type=int, help="max items per scene", required=False, default=None)
    parser.add_argument("--train_split_limit", type=int, help="max items for a train split", required=False, default=None)
    parser.add_argument("--test_split_limit", type=int, help="max items for a test split", required=False, default=None)
    parser.add_argument("--eval_split_limit", type=int, help="max items for an eval split", required=False, default=None)
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_dimension", type=int, help="max dimension", required=False, default=3000)
    parser.add_argument("--min_points", type=int, help="min points", required=False, default=100)

    # CMP
    # marigold_rel_path = "../Marigold"
    # RCI
    # marigold_rel_path = "../marigold_private"
    # from local
    # marigold_rel_path = "../../Desktop/mount_rci/marigold_private"
    parser.add_argument("--marigold_rel_path", type=str, required=False, default="../marigold_private")

    args = parser.parse_args()

    scenes_train = args.scenes_train
    if not scenes_train:
        scenes_train = ['statue_of_liberty', 'karluv_most',
                        'Brooklyn_Bridge', 'Old_Town_Hall_Prague',
                        'sagrada_familia', 'Dom_Luis_I_bridge_Porto',
                        'notre_dame_paris', 'Santa_Maria_del_Fiore_Florence',
                        'Hagia_Sophia']

    scenes_eval = args.scenes_eval
    if not scenes_eval:
        scenes_eval = ['sacre_couer',
                       'tycho_brahe_johannes_kepler_prague',
                       'letohradek_kralovny_anny',
                       'saint_ignatius_church_prague',
                       'church_of_the_sacred_heart_prague']

    scenes_test = args.scenes_test
    if not scenes_test:
        scenes_test = ['arc_de_triomf_barcelona', 'prasna_brana', 'sydney_opera']

    splits_scenes_limits = (("train", scenes_train, args.train_split_limit),
                            ("test", scenes_test, args.test_split_limit),
                            ("eval", scenes_eval, args.eval_split_limit))

    generate_splits_for_scene(splits_scenes_limits,
                              args.marigold_rel_path,
                              max_per_scene=args.max_per_scene,
                              prefix=args.prefix,
                              shuffle=args.shuffle,
                              max_dimension=args.max_dimension,
                              min_points=args.min_points)


if __name__ == '__main__':
    main()
