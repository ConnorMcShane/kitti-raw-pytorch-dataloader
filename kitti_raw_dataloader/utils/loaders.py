"""Methods to load in data files"""

import cv2
import numpy as np
import copy
import os
import open3d as o3d


class Loaders():
    """Class for loading different data modalities."""

    def __init__(self, cfg: object) -> None:
        """Initialize Loaders class.
        Args:
            cfg (config): stores all the data configurations;
        """
        self.cfg = cfg
        self.load_dict = {
            "image": self.load_image,
            "odometry": self.load_odometry,
            "pointcloud": self.load_pointcloud,
            "pointcloud_bev": self.load_pointcloud_bev,
            "pointcloud_depthmap": self.load_pointcloud_depthmap,
            "calibration": self.load_calibration,
        }
        self.intrinsics, self.extrinsics = None, None


    @staticmethod
    def load_image(path: str, to_rgb=False) -> np.ndarray:
        """Load image from path.
        Args:
            path (str): path to image;
        Returns:
            np.ndarray: image;
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    @staticmethod
    def load_odometry(path: str) -> np.ndarray:
        """Load odometry from path.
        Args:
            path (str): path to odometry text file;
        Returns:
            np.ndarray: odometry;
        """
        return np.loadtxt(path)


    @staticmethod
    def load_pointcloud(path: str) -> np.ndarray:
        """Load pointcloud from path.
        Args:
            path (str): path to pointcloud file;
        Returns:
            np.ndarray: pointcloud;
        """
        if path.endswith(".bin"):
            point_cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        elif path.endswith(".pcd"):
            point_cloud = np.asarray(o3d.io.read_point_cloud(path).points)
        else:
            raise NotImplementedError("Point cloud format not supported!")
        return point_cloud


    def load_pointcloud_bev(self, path: str, height: int, width: int, x_range: tuple, y_range: tuple, z_max: float) -> np.ndarray:
        """Load pointcloud from path.
        Args:
            path (str): path to pointcloud file;
        Returns:
            np.ndarray: pointcloud;
        """
        # load point cloud
        point_cloud = self.load_pointcloud(path)
        
        # remove points outside of range
        point_cloud = point_cloud[(point_cloud[:, 0] >= x_range[0]) & (point_cloud[:, 0] <= x_range[1])]
        point_cloud = point_cloud[(point_cloud[:, 1] >= y_range[0]) & (point_cloud[:, 1] <= y_range[1])]

        # Calculate the step sizes for x and y
        x_step = (x_range[1] - x_range[0]) / width
        y_step = (y_range[1] - y_range[0]) / height
        
        # Calculate the indices for each point's location in the BEV image
        x_indices = ((point_cloud[:, 0] - x_range[0]) / x_step).astype(int)
        y_indices = ((point_cloud[:, 1] - y_range[0]) / y_step).astype(int)
        z_values = point_cloud[:, 2]
        
        # Create an empty BEV image
        bev_image = np.zeros((height, width))
        
        # Assign the z values to the BEV image based on the max value for each pixel
        bev_image[y_indices, x_indices] = z_values
        np.maximum.at(bev_image, (y_indices, x_indices), z_values)

        # Normalize the BEV image
        bev_image = bev_image / z_max
        ground_offset = 0.5 # offsets points above the ground to make them more visible, therefore the true range will be between 0.5 and 1.0
        bev_image[bev_image != 0] += (ground_offset / (1 - ground_offset))
        bev_image /= ((ground_offset / (1 - ground_offset)) + 1.0)
        bev_image = np.clip(bev_image, 0, 1)

        return bev_image
    

    def load_pointcloud_depthmap(self, path: str, camera: str = "image_02") -> np.ndarray:
        # load point cloud
        point_cloud = self.load_pointcloud(path)

        # reproject lidar points to image
        lidar2cam = self.extrinsics["lidar2cam"][camera]

        # project lidar points to image
        point_cloud = point_cloud[:, :3].T
        point_cloud = np.concatenate([point_cloud, np.ones([1, point_cloud.shape[1]])], axis=0)
        img_points = lidar2cam @ point_cloud
        depths = img_points[2]
        img_points = img_points[:2] / img_points[2]
        img_points = img_points.T.astype(np.int32)

        # filter points outside image
        img_size = self.intrinsics["size"][camera]
        mask = np.logical_and(img_points[:, 0] >= 0, img_points[:, 0] < img_size[0])
        mask = np.logical_and(mask, img_points[:, 1] >= 0)
        mask = np.logical_and(mask, img_points[:, 1] < img_size[1])
        img_points = img_points[mask]
        depths = depths[mask]

        # save depthmap image
        depthmap = np.zeros([img_size[1], img_size[0]])
        depthmap[img_points[:, 1], img_points[:, 0]] = depths

        return depthmap


    @staticmethod
    def load_calibration(path: str) -> np.ndarray:
        """Load extrinsics from path.
        Args:
            path (str): path to calibration files;
        Returns:
            np.ndarray: extrinsics;
        """
        # read in calibration files
        calib =  {
            "cam_to_cam":os.path.join(path, "calib_cam_to_cam.txt"),
            "velo_to_cam":os.path.join(path, "calib_velo_to_cam.txt"),
            "imu_to_velo":os.path.join(path, "calib_imu_to_velo.txt"),
        }
        for calib_name, calib_path in calib.items():
            if not os.path.exists(calib_path):
                raise FileNotFoundError(f"{calib_name} not found at {calib_path}")
            with open(calib_path, "r") as f:
                calib_data = f.readlines()
            calib_data = {line.split(" ")[0].replace(":", ""):np.array(line.split(" ")[1:], dtype=np.float32) for line in calib_data[1:]}
            calib[calib_name] = calib_data

        # process camera intrinsics and extrinsics
        cam_to_cam00 = calib["cam_to_cam"]["P_rect_00"].reshape([3, 4])
        cam_to_cam01 = calib["cam_to_cam"]["P_rect_01"].reshape([3, 4])
        cam_to_cam02 = calib["cam_to_cam"]["P_rect_02"].reshape([3, 4])
        cam_to_cam03 = calib["cam_to_cam"]["P_rect_03"].reshape([3, 4])
        cam_to_cam00 = np.concatenate([cam_to_cam00, np.array([[0., 0., 0., 1.]])], axis=0)
        cam_to_cam01 = np.concatenate([cam_to_cam01, np.array([[0., 0., 0., 1.]])], axis=0)
        cam_to_cam02 = np.concatenate([cam_to_cam02, np.array([[0., 0., 0., 1.]])], axis=0)
        cam_to_cam03 = np.concatenate([cam_to_cam03, np.array([[0., 0., 0., 1.]])], axis=0)

        # calculate lidar to camera extrinsics
        rect = calib["cam_to_cam"]["R_rect_00"].reshape([3, 3])
        rect_4x4 = np.eye(4, dtype=rect.dtype)
        rect_4x4[:3, :3] = rect
        lidar2cam = np.zeros([3, 4], dtype=rect.dtype)
        lidar2cam[:3, :3] = calib["velo_to_cam"]["R"].reshape([3, 3])
        lidar2cam[:, 3] = calib["velo_to_cam"]["T"].T
        lidar2cam = np.concatenate([lidar2cam, np.array([[0., 0., 0., 1.]])], axis=0)
        lidar2cam = rect_4x4 @ lidar2cam

        # intrinsics
        intrinsics = {}

        # cam intrinsics only
        intrinsics["matrix"] = {
            "image_00": calib["cam_to_cam"]["P_rect_00"].reshape(3, 4)[:, :3],
            "image_01": calib["cam_to_cam"]["P_rect_01"].reshape(3, 4)[:, :3],
            "image_02": calib["cam_to_cam"]["P_rect_02"].reshape(3, 4)[:, :3],
            "image_03": calib["cam_to_cam"]["P_rect_03"].reshape(3, 4)[:, :3],
        }

        # cam image size
        intrinsics["size"] = {
            "image_00": (int(calib["cam_to_cam"]["S_rect_00"][0]), int(calib["cam_to_cam"]["S_rect_00"][1])),
            "image_01": (int(calib["cam_to_cam"]["S_rect_01"][0]), int(calib["cam_to_cam"]["S_rect_01"][1])),
            "image_02": (int(calib["cam_to_cam"]["S_rect_02"][0]), int(calib["cam_to_cam"]["S_rect_02"][1])),
            "image_03": (int(calib["cam_to_cam"]["S_rect_03"][0]), int(calib["cam_to_cam"]["S_rect_03"][1])),
        }
        
        # extrinsics
        extrinsics = {}

        # cam extrinsics and intrinsics
        extrinsics["cam2cam"] = {
            "image_00": cam_to_cam00,
            "image_01": cam_to_cam01,
            "image_02": cam_to_cam02,
            "image_03": cam_to_cam03,
        }
        
        # lidar to cam extrinsics and intrinsics
        extrinsics["lidar2cam"] = {
            "image_00": cam_to_cam00 @ lidar2cam,
            "image_01": cam_to_cam01 @ lidar2cam,
            "image_02": cam_to_cam02 @ lidar2cam,
            "image_03": cam_to_cam03 @ lidar2cam,
        }

        # imu to velo extrinsics
        extrinsics["imu2velo"] = calib["imu_to_velo"]["R"].reshape(3, 3)
        extrinsics["imu2velo"] = np.concatenate([extrinsics["imu2velo"], calib["imu_to_velo"]["T"].reshape(3, 1)], axis=1)
        extrinsics["imu2velo"] = np.concatenate([extrinsics["imu2velo"], np.array([[0., 0., 0., 1.]])], axis=0)

        return intrinsics, extrinsics
