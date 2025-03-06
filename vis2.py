import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from math import radians

if __name__ == '__main__':
    def get_intrinsic(width, height):
        return o3d.core.Tensor([[1, 0, width * 0.5],
                                [0, 1, height * 0.5],
                                [0, 0, 1]])


    def get_extrinsic(x=0, y=0, z=0, rx=0, ry=0, rz=0):
        extrinsic = np.eye(4)
        extrinsic[:3, 3] = (x, y, z)
        extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([radians(rx), radians(ry), radians(rz)])
        return extrinsic


    def compute_show_reprojection(pcd, width, height, intrinsic, extrinsic):
        depth_reproj = pcd.project_to_depth_image(width,
                                                  height,
                                                  intrinsic,
                                                  extrinsic,
                                                  depth_scale=5000.0,
                                                  depth_max=10.0)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(np.asarray(depth.to_legacy()))
        axs[1].imshow(np.asarray(depth_reproj.to_legacy()))
        plt.show()


    width, height = 640, 480
    intrinsic = get_intrinsic(width, height)
    extrinsic = get_extrinsic()
    # original example data
    tum_data = o3d.data.SampleTUMRGBDImage()
    depth = o3d.t.io.read_image(tum_data.depth_path)
    pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth,
                                                            intrinsic,
                                                            extrinsic,
                                                            depth_scale=5000.0,
                                                            depth_max=10.0)

    compute_show_reprojection(pcd, width, height, intrinsic, get_extrinsic(z=1, rz=-45))

    # testing a differen point cloud
    pcd_data = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_data.path)
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    c = pcd.get_center().numpy()
    minb = pcd.get_min_bound().numpy()
    maxb = pcd.get_max_bound().numpy()
    print('point cloud center', c)
    print('min', minb)
    print('max', maxb)
    compute_show_reprojection(pcd, width, height, intrinsic, get_extrinsic(c[0], c[1] + 1, c[2] + 1))