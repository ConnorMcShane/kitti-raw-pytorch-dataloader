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
            "calib": self.load_calib,
        }


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

        # # create empty bev image
        # point_cloud_bev = np.zeros((height, width), dtype=np.float32)

        # point_cloud_bev = point_cloud_bev / z_max
        # point_cloud_bev = np.clip(point_cloud_bev, 0, 1)

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


    @staticmethod
    def load_calib(path: str) -> np.ndarray:
        """Load calibration from path.
        Args:
            path (str): path to calibration file;
        Returns:
            np.ndarray: calibration;
        """
        return np.load(path)
