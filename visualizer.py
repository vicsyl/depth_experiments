import argparse
import time

import PIL.Image
import numpy as np
import open3d
import cv2 as cv


def read_f():
    return 1000

def visualize(depth_file, rgb_file, f):
    d_cv = cv.imread(depth_file)
    height, width = d_cv.shape[:2]

    depth_img = open3d.io.read_image(depth_file)

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

    #ctr = vis2.get_view_control()
    #camera_params = ctr.convert_to_pinhole_camera_parameters()
    #print(f"{camera_params=}")
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
    vis2.destroy_window()

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

    rgb_file = "data/depths/input/example_0.jpg"
    # depth_file = "data/depths/depth_simple/depth.png"
    visualize(args.depth_file, rgb_file, args.f)
