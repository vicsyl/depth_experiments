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

from depths_from_sfm import get_camera_params
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
        logging.info(f"stdout: \n {stdout}")
        logging.info(f"stderr: \n {stderr}")
    return process.returncode, stdout


def download_reconstruction(short_i, just_remove=False):
    """
    @param short_i:
    @return: success, to_dir
    """
    to_dir = "sfm_dataset/work/reconstruction"
    rm_dir = f"{to_dir}/*"
    scene_key = get_key_from_short_index(short_i)
    scene_key = "000/001"

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

    def generate_hists(points_counts, points_densities, bns=100, show=False):
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle(f"{scene_key}_{name}_{dir} - {len(points_counts)} images")
        counts, bins = np.histogram(points_counts, bins=bns)
        axs[0].set_title("points counts")
        axs[0].stairs(counts, bins)  # IMHO works better than plt.axis.hist
        axs[1].set_title("points densities")
        counts, bins = np.histogram(points_densities, bins=bns)
        axs[1].stairs(counts, bins)  # IMHO works better than plt.axis.hist

        save_img_file = f'data/img_gallery/{scene_key.replace("/", "_")}_{dir}_{name}.png'
        os.makedirs(pathlib.Path(save_img_file).parent, exist_ok=True)
        fig.savefig(save_img_file)
        if show:
            fig.show()

    success, out_dir = download_reconstruction(short_i)
    if not success:
        logging.info(f"could not download reconstruction {name}")
        data[short_i]["success"] = 0
        return
    for dir in os.listdir(out_dir):
        in_dir = os.path.join(out_dir, dir)
        points_counts, dimensions = one_reconstruction_stats(in_dir)

        out_dir_s = "data/sfm_stats_data"
        os.makedirs(out_dir_s, exist_ok=True)
        np.savez_compressed(f"{out_dir_s}/{short_i}_{dir}_stats.npz",
                            points_counts=points_counts,
                            dimensions=dimensions)

        points_densities = [p / (d1 * d2) for (d1, d2), p in zip(dimensions, points_counts)]
        generate_hists(points_counts, points_densities)
        data[short_i]["success"] = 1
        data[short_i]["imgs"] = len(points_counts)

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

    #read_n = 2
    data = {}

    with open("./sfm_dataset/categories.json", "rb") as fd:
        js = json.load(fd)

    #for name, short_i in list(js.items())[:read_n]:
    for name, short_i in list(js.items()):
        #get_image_stats(name, short_i, data)
        reconstruction_stats(name, short_i, data)

    logging.info("data:")
    logging.info(data)
    with open("./data/stats.py", "w") as fd:
        fd.write("data = {}\n")
        for k, v in data.items():
            fd.write(f"data[{k}] = {v}\n")


if __name__ == '__main__':
    config_logging()
    main()