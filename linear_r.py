import glob
from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from gallery import HtmlGallery


def name_map(itr):
    r = {}
    for i in sorted(list(itr)):
        name = Path(i).name
        name = name[:name.rfind(".")]
        name = name[:name.rfind(".")]
        r[name] = i
    return r


def create_html_gallery(apt, room):
    dir = f"./data/gallery/{apt}_{room}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    HtmlGallery(columns_by_width=True).write_gallery_file(in_folder=dir,
                                     target_folder=dir,
                                     width_pc=19)


def run():
    for apt, room in [('apt1', 'kitchen'), ('apt1', 'living'), ('apt2', 'bed'), ('apt2', 'kitchen'), ('apt2', 'living'), ('apt2', 'luke'), ('office2', '5a'), ('office2', '5b')]:
    # for apt, room in [('apt1', 'kitchen')]:
        mses, coeffs, shifts = run_apt_room(apt, room, show=False)
        create_hist(apt, room, mses, coeffs, shifts, show=False)
        create_html_gallery(apt, room)
    create_hist_gallery()


def create_hist_gallery():
    dir = "./data/histograms"
    HtmlGallery().write_gallery_file(in_folder=dir,
                                     target_folder=dir,
                                     width_pc=28)


def create_hist(apt, room, mses, coeffs, shifts, show=True):

    plt.figure()
    plt.title(f"Scale coefficients for {apt}/{room}")
    plt.xlabel("no unit")
    plt.ylabel("# data points")

    # max_dist_value = 100
    # coeffs = np.clip(coeffs, a_min=0.0, a_max=coeffs.max())
    bins = np.linspace(0, coeffs.max() + 0.01, 123)
    plt.hist(coeffs, bins, alpha=0.5)

    path = f"./data/histograms/{apt}_{room}.coeffs.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.title(f"Shifts coefficients for {apt}/{room}")
    plt.xlabel("shift[m.]")
    plt.ylabel("# data points")

    # shifts = np.clip(coeffs, a_min=0.0, a_max=shifts.max())
    bins = np.linspace(0, shifts.max() + 0.01, 123)
    plt.hist(shifts, bins, alpha=0.5)

    path = f"./data/histograms/{apt}_{room}.shifts.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close()

    plt.figure()
    plt.title(f"MSE of linear regression for {apt}/{room}")
    plt.xlabel("MSE [m^2]")
    plt.ylabel("# data points")

    # mses = np.clip(coeffs, a_min=0.0, a_max=mses.max())
    bins = np.linspace(0, mses.max() + 0.01, 123)
    plt.hist(mses, bins, alpha=0.5)

    path = f"./data/histograms/{apt}_{room}.mses.png"
    plt.savefig(path, bbox_inches='tight')
    print(f"Saved to {path}")
    if show:
        plt.show()
    plt.close()


def run_apt_room(apt, room, show=True):

    max_files = 100000

    print(f"Running apt {apt} room {room}")

    ts_in = f"../twelve_scenes/data/ds/{apt}/{room}/data"

    color_files = set(glob.glob(f"{ts_in}/*.color.jpg"))
    color_files = name_map(color_files)
    print(f"color_files length: {len(color_files)}")

    depth_files = set(glob.glob(f"{ts_in}/*.depth.png"))
    depth_files = name_map(depth_files)
    print(f"depth_files length: {len(depth_files)}")

    mge_in = f"./data/work_ts/{apt}_{room}"

    depth_ests = set(glob.glob(f"{mge_in}/*.color.npy"))
    depth_ests = name_map(depth_ests)
    print(f"depth_est length: {len(depth_ests)}")

    depth_est_vs = set(glob.glob(f"{mge_in}/*.color.jpg"))
    depth_est_vs = name_map(depth_est_vs)
    print(f"depth_est_v length: {len(depth_est_vs)}")

    mses = []
    coeffs = []
    shifts = []

    for idx, (color_file, color_file_path) in enumerate(list(color_files.items())[:max_files]):

        if (idx + 1) % 100 == 0:
            print(f"{idx}/{len(color_files)}")

        depth_gt_file = depth_files[color_file]
        depth_est = depth_ests[color_file]
        depth_est_v = depth_est_vs[color_file]

        color_img = cv.imread(color_file_path)
        depth_est_img = np.load(depth_est)

        depth_gt_img = cv.imread(depth_gt_file, cv.IMREAD_ANYDEPTH)
        depth_gt_img = depth_gt_img / 1000.0
        # FIXME - TODO
        depth_gt_img[depth_gt_img < 0.1] = depth_gt_img.mean()

        depth_gt_img = cv.resize(depth_gt_img, depth_est_img.shape[::-1], interpolation=cv.INTER_LINEAR)
        depth_est_v_img = cv.imread(depth_est_v)

        rows = 2
        cols = 2
        f, axs = plt.subplots(rows, cols, figsize=(7, 6))

        axs[0, 0].title.set_text("original")
        axs[0, 0].imshow(color_img)
        axs[0, 1].title.set_text("depth GT")
        axs[0, 1].imshow(depth_gt_img)
        axs[1, 0].title.set_text("depth estimate")
        axs[1, 0].imshow(depth_est_img)
        axs[1, 1].title.set_text("depth estimate inverted depth")
        axs[1, 1].imshow(depth_est_v_img[:, :, 0])
        dir = f"./data/gallery/{apt}_{room}"
        Path(dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{dir}/{color_file}.jpg", bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        y = depth_gt_img.flatten()
        x = depth_est_img.flatten().reshape(-1, 1)
        reg = LinearRegression().fit(x, y)

        y_hat = x @ reg.coef_ + reg.intercept_
        residual = y - y_hat
        mse = (residual ** 2).sum() / residual.shape[0]
        coeff = reg.coef_.item()
        shift = reg.intercept_.item()

        mses.append(mse)
        coeffs.append(coeff)
        shifts.append(shift)

    return np.array(mses), np.array(coeffs), np.array(shifts)


if __name__ == "__main__":
    run()
