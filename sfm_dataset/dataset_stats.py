import argparse
import json
import logging
import os
import pathlib
import re
import subprocess
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from tqdm import tqdm

from utils import config_logging

img_suffixes = set([".jpg", ".jpeg", ".png"])


class Log:

    stats = defaultdict(list)
    log = []

    @staticmethod
    def add_log(m):
        Log.log.append(m)
        logging.info(m)


def get_key_from_short_index(short_i):
    short_i = str(short_i)
    i = "0" * (6 - len(short_i)) + short_i
    i = i[:3] + "/" + i[3:]
    logging.debug(f"{short_i} => {i}")
    return i


def run_subprocess(cmd: List[str], log=True):
    logging.info(f"running: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if log:
        logging.info(f"stdout: {stdout}")
        logging.info(f"stderr: {stderr}")
    return process.returncode, stdout


def download_reconstruction(short_i, just_remove=False):
    """
    @param short_i:
    @return: success, to_dir
    """
    to_dir = "data/sfm_work/reconstruction"
    rm_dir = f"{to_dir}/*"
    scene_key = get_key_from_short_index(short_i)
    # testing
    # scene_key = "000/001"

    # remove the previous working data
    command = ["rm", "-rf", rm_dir]
    returncode, stdout = run_subprocess(command)
    if returncode != 0 or just_remove:
        return False, None

    command = ["s5cmd", "--no-sign-request", "cp", f"s3://megascenes/reconstruct/{scene_key}/colmap/*", to_dir]
    returncode, stdout = run_subprocess(command)
    if returncode != 0:
        return False, None

    return True, to_dir


#
def get_lines_from_reconstruct(scene_key):
    """
    E.g.:
    s5cmd --no-sign-request ls s3://megascenes/reconstruct/000/001/colmap/*
    2024/05/30 07:20:30             11264  0/cameras.bin
    2024/05/30 07:20:42          34265398  0/images.bin
    2024/05/30 07:20:33           3465260  0/points3D.bin
    2024/05/30 07:20:30              1761  0/project.ini
    """
    return get_lines_for_path(f"reconstruct/{scene_key}/colmap/*")


def get_lines_for_images_dir(scene_key):
    return get_lines_for_path(f"images/{scene_key}/*")


def get_lines_for_path(path):
    params = ["s5cmd", "--no-sign-request", "ls", f"s3://megascenes/{path}"]
    logging.info(f"running: {params}")
    cp = subprocess.run(params, capture_output=True)
    output = cp.stdout.decode("utf-8")
    lines = output.split("\n")
    return lines


def get_image_stats(name, short_i, data):

    images, sizes = get_images_sizes(short_i)
    images_c = len(images)
    del images
    avg_size = np.asarray(sizes).mean()

    data[name]["avg_size"] = avg_size
    data[name]["images"] = images_c
    logging.info(f"{name}: {images_c} images, average size: {avg_size}")


def get_images_sizes(short_i):

    images = []
    sizes = []

    i = get_key_from_short_index(short_i)
    lines = get_lines_for_images_dir(i)

    for l in lines:
        if len(l.strip()) == 0:
            continue
        l_s = re.split(r' +', l)
        size_entry = int(l_s[2])
        file_entry = l_s[3]
        p = pathlib.Path(file_entry)
        if set(p.parts).__contains__("pictures"):
            if not p.suffix.lower() in img_suffixes:
                Log.stats["uknown_suffix"].append(p)
                Log.add_log(f"Unknown suffix in pictures dir: {p}")
            else:
                logging.debug(f"image in {p}")
                images.append(str(p))
                sizes.append(size_entry)

        elif p.suffix in img_suffixes:
            Log.stats["img_not_in_pictures_dir"].append(str(p))
            Log.add_log(f"Image not in pictures dir: {p}")

    return images, sizes


def reconstruction_stats(name, short_i, data):

    scene_key = get_key_from_short_index(short_i)

    data[short_i] = {}
    data[short_i]["name"] = name
    data[short_i]["short_i"] = short_i
    data[short_i]["scene_key"] = scene_key

    def generate_hists(points_counts, dimensions, bns=100, show=False):

        max_dims = [max(d1, d2) for d1, d2 in dimensions]
        areas = [(d1 * d2) for d1, d2 in dimensions]
        points_densities = [p / a for a, p in zip(areas, points_counts)]

        fig, axs = plt.subplots(1, 4, figsize=(10, 4))
        fig.suptitle(f"{scene_key}_{name}_{dir} - {len(points_counts)} images")
        counts, bins = np.histogram(points_counts, bins=bns)
        axs[0].set_title("points counts")
        axs[0].stairs(counts, bins)

        counts, bins = np.histogram(points_densities, bins=bns)
        axs[1].set_title("points densities")
        axs[1].stairs(counts, bins)

        counts, bins = np.histogram(areas, bins=bns)
        axs[2].set_title("areas")
        axs[2].stairs(counts, bins)

        counts, bins = np.histogram(max_dims, bins=bns)
        axs[3].set_title("max dimension")
        axs[3].stairs(counts, bins)

        save_img_file = f'data/img_gallery/{scene_key.replace("/", "_")}_{dir}_{name.replace("/", "_")}.png'
        os.makedirs(pathlib.Path(save_img_file).parent, exist_ok=True)
        fig.savefig(save_img_file)
        th = 100
        if len(points_counts) > th:
            save_img_file = f'data/img_gallery_{th}/{scene_key.replace("/", "_")}_{dir}_{name.replace("/", "_")}.png'
            os.makedirs(pathlib.Path(save_img_file).parent, exist_ok=True)
            fig.savefig(save_img_file)
        if show:
            fig.show()

    success, out_dir = download_reconstruction(short_i)
    if not success:
        logging.info(f"could not download reconstruction {name}")
        data[short_i]["success"] = 0
        return

    data[short_i]["success"] = 1
    data[short_i]["reconstructions"] = []
    for dir in os.listdir(out_dir):

        in_dir = os.path.join(out_dir, dir)
        points_counts, dimensions = one_reconstruction_stats(in_dir)

        out_dir_s = "data/sfm_stats_data"
        os.makedirs(out_dir_s, exist_ok=True)
        np.savez_compressed(f"{out_dir_s}/{short_i}_{dir}_stats.npz",
                            points_counts=points_counts,
                            dimensions=dimensions)

        generate_hists(points_counts, dimensions)
        data[short_i]["reconstructions"].append({"name": dir, "imgs": len(points_counts)})

    download_reconstruction(short_i, just_remove=True)


def one_reconstruction_stats(in_dir):

    reconstruction = pycolmap.Reconstruction(in_dir)
    logging.info("Reconstruction summary:")
    logging.info(reconstruction.summary())

    image_items = list(reconstruction.images.items())
    logging.info(f"Number of images: {len(image_items)}")

    points_counts = []
    dimensions = []

    for _, colmap_image in tqdm(image_items):
        logging.debug(f"processing image: {colmap_image.name}")
        camera = reconstruction.camera(colmap_image.camera_id)
        cmodel = camera.model.name
        assert cmodel == "SIMPLE_RADIAL"
        w_est = camera.params[1]
        w_est = w_est * 2 + 1
        h_est = camera.params[2]
        h_est = h_est * 2 + 1
        points_c = len([p2d for p2d in colmap_image.points2D if p2d.point3D_id <= 10 ** 11 and p2d.point3D_id != -1])
        points_counts.append(points_c)
        dimensions.append((h_est, w_est))

    return points_counts, dimensions


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--read_previous", action=argparse.BooleanOptionalAction, required=False, default=True)
    parser.add_argument("--max_scenes", type=int, required=False, default=None)
    parser.add_argument("--save_period", type=int, required=False, default=1000)
    args = parser.parse_args()

    if args.read_previous:
        from data.stats_read import data
        data_l = data
    else:
        data_l = {}

    with open("./data/sfm_work/categories.json", "rb") as fd:
        js = json.load(fd)

    ids_names = list(js.items())
    if args.max_scenes is not None:
        ids_names = ids_names[:args.max_scenes]

    def save_data(fp="./data/stats.py"):
        with open(fp, "w") as fd:
            fd.write("data = {}\n")
            for k, v in data_l.items():
                fd.write(f"data[{k}] = {v}\n")
        logging.info(f"data saved to {fp}")

    for i, (name, short_i) in tqdm(enumerate(ids_names)):
        if short_i in data_l:
            logging.info(f"skipping {name} as it is {short_i=} is already in the map")
            continue
        #get_image_stats(name, short_i, data)
        reconstruction_stats(name, short_i, data_l)
        if (i - 10) % args.save_period == 0:
            save_data()

    save_data()
    logging.info("Done")


def stats_filter():
    from data.stats import data
    print(f"{len(data)=}")
    data = {k: v for k, v in data.items() if v["success"] == 1}
    print(f"sucesses: {len(data)=}")

    all_images = []
    for k, v in data.items():
        s = 0
        for i in v['reconstructions']:
            s += i['imgs']
        all_images.append((k, s, v['name']))

    all_images = sorted(all_images, key=lambda x: x[1])
    print("\n".join([str(i) for i in all_images]))


if __name__ == '__main__':
    config_logging()
    main()
    #stats_filter()
