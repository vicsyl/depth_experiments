import argparse
import time

import PIL.Image
import numpy as np
import open3d
import cv2 as cv


def read_f():
    return 10


def vis(width, height, f, depth_img):
    camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height,
                                                            f, f,
                                                            width / 2, height / 2)

    camera_extrinsic = np.eye(4, dtype=np.float32)
    m = camera_intrinsic.intrinsic_matrix
    print(f"Intrinsic matrix: {m}")
    print(f"Extrinsic matrix: {camera_extrinsic}")

    # camera_extrinsic superfluous
    pc_d = open3d.geometry.PointCloud.create_from_depth_image(depth_img, camera_intrinsic, camera_extrinsic)
    vis2 = open3d.visualization.Visualizer()

    # ctr = vis2.get_view_control()
    # camera_params = ctr.convert_to_pinhole_camera_parameters()
    # print(f"{camera_params=}")
    # print(f"{camera_params.extrinsic=}")
    #  ctr.convert_from_pinhole_camera_parameters(parameters)

    # open3d.visualization.ViewControl()
    # p = PinholeCameraParameters()
    # vis2.get_view_control() convert_from_pinhole_camera_parameters(camera_intrinsic)
    vis2.create_window(window_name='pointcloud', width=width, height=height)
    vis2.add_geometry(pc_d)

    ctr = vis2.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    print(f"{camera_params.extrinsic=}")
    camera_params.extrinsic = np.eye(4, dtype=np.float32)

    camera_params.intrinsic = open3d.camera.PinholeCameraIntrinsic(width, height,
                                                                   f, f,
                                                                   width / 2, height / 2)

    print(f"{camera_params=}")
    print(f"{camera_params.intrinsic=}")
    print(f"{camera_params.extrinsic=}")

    # parameters = open3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-10-30-13-07-27.json")
    # b = ctr.convert_from_pinhole_camera_parameters(parameters)

    b = ctr.convert_from_pinhole_camera_parameters(camera_params)
    print(f"success: {b}")
    #  ctr.convert_from_pinhole_camera_parameters(parameters)
    vis2.update_renderer()

    ctr = vis2.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    print(f"2 {camera_params.extrinsic=}")
    vis2.update_renderer()

    vis2.run()
    return vis2


def visualize(depth_file, rgb_file, f):
    d_cv0 = cv.imread("data/depths/depth_simple/depth.png")

    if depth_file.endswith(".npy"):
        npy_np = np.load(depth_file)
    elif depth_file.endswith(".npz"):
        npy_np = np.load(depth_file)["depth"]
    else:
        raise ValueError("depth_file must end with .npy or .npz")

    if len(npy_np.shape) == 4:
        assert npy_np.shape[0] == npy_np.shape[1] == 1
        npy_np = npy_np.squeeze()

    show_inverse = False
    if show_inverse:
        npy_np = 1 / np.clip(npy_np, a_min=1.0, a_max=300)

    print(f"loaded none: {npy_np is None}")
    print(f"shape: {npy_np.shape}")
    height, width = npy_np.shape[:2]

    vs = []

    npy_np[npy_np == np.inf] = 0.1
    #npy_np = (npy_np * 255).astype(np.uint16)[..., None] + 100
    npy_np = (npy_np * 255).astype(np.uint16)[..., None]
    npy_np = np.ascontiguousarray(npy_np)

    depth_img = open3d.geometry.Image(npy_np)
    vs.append(vis(width, height, f, depth_img))

    # depth_img = open3d.geometry.Image(((1 / npy_np) * 255).astype(np.uint16)[..., None])
    # vs.append(vis(width, height, f, depth_img))

    for v in vs:
        v.destroy_window()

    # THIS (i.e. add rgb) doesn't work somehow
    # rgb_img = PIL.Image.open(rgb_file)
    # rgb_img = rgb_img.resize((width, height), resample=PIL.Image.Resampling.LANCZOS)
    # #rgb_img = open3d.io.read_image(rgb_file)
    # rgb_img = open3d.geometry.Image(np.asarray(rgb_img))
    # rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img)
    # pc_rgbs = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    # vis = open3d.visualization.Visualizer()
    # vis.create_window(window_name='pointcloud', width=1200, height=800)
    # vis.add_geometry(pc_rgbs)
    # vis.run()

    # while True:
    #     time.sleep(2)



if __name__ == '__main__':
    # depth_file = "data/depths/in-the-wild_example_fixed_new/depth_bw/example_1_pred.png"

    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_file', type=str, required=False, default="data/depths/depth_simple/depth.png")
    parser.add_argument('--f', type=float, required=False, default=1000.0)
    args = parser.parse_args()

    # rgb_file = "data/depths/input/example_0.jpg"
    # rgb_file = "../output_loss_rci/glise_St_Ignace_Loyola_Prague_17png_depth_pred.npy"
    # rgb_file = "../Marigold/output_vis/depths_train_infer_Notre_dame._Paris.png.npy"
    # depth_file = "data/depths/depth_simple/depth.png"
    # --depth_file ../../../dev/marigold_local/output_loss_rci/glise_St_Ignace_Loyola_Prague_17png_depth_pred.npy
    # --depth_file ../../mount_rci/marigold_private/output_vis/depths_train_glise_St_Ignace_Loyola_Prague_17.png.npy
    # --depth_file ../../../dev/marigold_local/output_loss_rci/The_Old_Praha_Gate_Powder_Tower__panoramio__Andrej_Kunieykpng_depth_pred.npy

    depth_file = args.depth_file
    depth_file = "../../../dev/marigold_local/output_loss_rci/The_Old_Praha_Gate_Powder_Tower__panoramio__Andrej_Kunieykpng_depth_pred.npy"
    #depth_file = "../../../dev/marigold_local/output_vis_rci/depths_train_Sydney_(AU),_Opera_House_--_2019_--_2994.png.npy"
    depth_file = "../../../dev/marigold_local/output_loss_rci/glise_St_Ignace_Loyola_Prague_17png_depth_pred.npy"
    #depth_file = "../../mount_rci/marigold_private/output_vis/depths_train_glise_St_Ignace_Loyola_Prague_17.png.npy"

    visualize(depth_file, rgb_file=None, f=args.f)
