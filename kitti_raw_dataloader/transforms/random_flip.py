"""RandomFlip class for transforms."""

from typing import Dict, Any, List
import random
import numpy as np


class RandomFlip():
    """RandomFlip class for transforms."""

    def __init__(self, params: Dict[str, Any], cfg: Dict) -> None:
        """Initialize the transform class.
        Args:
            params (dict): parameters for the transform
        """
        self.cfg = cfg
        self.params: Dict[str, Any] = params

    def __call__(self, data: dict) -> dict:
        """Apply the transform to the data.
        Args:
            data (dict): data to apply the transform to

        Returns:
            data (dict): data with the transform applied
        """
        return self.apply_transform(data)

    def apply_transform(self, data: Dict) -> Dict:
        """Apply the transform to the data.
        Args:
            data (dict): data to apply the transform to

        Returns:
            data (dict): data with the transform applied
        """
        if self.params["flip_prob"] > 0.0:
            if random.random() < self.params["flip_prob"]:
                
                # images
                data = self._flip_image(data)

                # calib
                data = self._flip_calib(data)

                # lidar
                data = self._flip_lidar(data)

                # odometry
                data = self._flip_odometry(data)


        return data
    

    def _flip_image(self, data: dict) -> dict:
        """Flip the image
        
        Args:
            data (dict): data to apply the transform to
            
        Returns:
            data (dict): data with the transform applied
        """
        image_keys: List[str] = ["image", "depthmap", "bev"]

        for key, value in data.items():
            if any([k in key for k in image_keys]):
                if isinstance(value, np.ndarray):
                    if any([k in key for k in image_keys]):
                        data[key] = np.array(np.fliplr(value))
                elif isinstance(value, list) and isinstance(value[0], np.ndarray):
                    if any([k in key for k in image_keys]):
                        data[key] = [np.array(np.fliplr(v)) for v in value]
                else:
                    raise ValueError("value is not a numpy array")
        
        return data


    def _flip_calib(self, data: dict) -> dict:
        """Flip the calibration
        
        Args:
            data (dict): data to apply the transform to
            
        Returns:
            data (dict): data with the transform applied
        """
        
        for cam in data["intrinsics"]["matrix"].keys():
            img_width, _ = data["intrinsics"]["size"][cam][0], data["intrinsics"]["size"][cam][1]

            data["intrinsics"]["matrix"][cam][0, 2] = img_width - data["intrinsics"]["matrix"][cam][0, 2]
            data["extrinsics"]["cam2cam"][cam][:3, :3] = data["intrinsics"]["matrix"][cam][:3, :3]
            data["extrinsics"]["lidar2cam"][cam] = data["extrinsics"]["cam2cam"][cam] @ data["extrinsics"]["lidar2cam"]["original"]

        return data


    def _flip_lidar(self, data: dict) -> dict:
        """Flip the lidar points.
        
        Args:
            data (dict): data to apply the transform to
            
        Returns:
            data (dict): data with the transform applied
        """
        # flip lidar points

        assert isinstance(data["lidar_points"], np.ndarray), "value is not a numpy array"

        if self.params["flip_lidar_in_cam_frame"]:
            data["lidar_points"] = (data["extrinsics"]["lidar2cam"]["original"] @ data["lidar_points"].T).T
            data["lidar_points"][:, 0] = -data["lidar_points"][:, 0]
            data["lidar_points"] = (np.linalg.inv(data["extrinsics"]["lidar2cam"]["original"]) @ data["lidar_points"].T).T
        else:
            data["lidar_points"][:, 1] = -data["lidar_points"][:, 1]

        return data


    def _flip_odometry(self, data: dict) -> dict:
        """Flip the odometry
        
        Args:
            data (dict): data to apply the transform to
            
        Returns:
            data (dict): data with the transform applied
        """
        # TODO: implement

        return data
