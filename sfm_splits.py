import glob
import os.path
import random

import numpy as np
import argparse

from tqdm import tqdm


def run_for_scene(scenes):
    if scenes[0].lower() == "all":
        paths = sorted(list(glob.glob("../Marigold/data/sfm/data/*/data.txt")))
    else:
        paths = [f"../Marigold/data/sfm/data/{scene}/data.txt" for scene in scenes]

    print("Paths:\n", "\n".join(paths))

    # TODO for now only random splits....
    prefix = "big_"
    split_map = {"train": 0.8, "eval": 0.1, "test": 0.1}
    assert np.isclose(sum([split_map[k] for k in split_map.keys()]), 1.0)
    print(f"input split map: {split_map}")

    cum = 0.0
    check_1 = False
    for k in split_map.keys():
        cum += split_map[k]
        f = open(os.path.join("../Marigold/data/sfm", f"{prefix}{k}.txt"), "w")
        split_map[k] = (cum, f)
        assert cum < 1.000001
        if np.isclose(cum, 1.0):
            check_1 = True

    assert check_1
    print(f"cumulative map with file handlers: {split_map}")

    for path in paths:
        print(f"reading {path}")
        with open(path, "r") as f:
            for line in tqdm(f.readlines()):
                r = random.random()
                for split in split_map.keys():
                    if r <= split_map[split][0]:
                        split_map[split][1].write(line)
                        break

    for split in split_map.keys():
        split_map[split][1].close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", help="scene (e.g., 'prasna_brana karluv most')", required=True)
    args = parser.parse_args()
    run_for_scene(args.scenes)
