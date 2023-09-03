"""Crop class for transforms."""

from typing import Dict, Any, List
import numpy as np

class Crop():
    """Crop class for transforms."""

    def __init__(self, params: Dict[str, Any], cfg: Dict) -> None:
        """Initialize the transform class.
        Args:
            params (dict): parameters for the transform
        """
        self.cfg = cfg
        self.params: Dict[str, Any] = params


    def _calc_pixels(self) -> None:
        """Calculate the number of pixels to crop."""
        self.left = self.params["left"]
        self.right = 1 - self.params["right"]
        self.top = self.params["top"]
        self.bottom = 1 - self.params["bottom"]


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

        self._calc_pixels()
        assert self.left < self.right, "left must be smaller than right"
        assert self.top < self.bottom, "top must be smaller than bottom"

        # crop image
        data = self._crop_image(data)

        # crop calib
        data = self._crop_calib(data)

        return data

    def _crop_image(self, data: Dict) -> Dict:
        """Crop the image to the given shape.
        Args:
            image (Image): image to resize

        Returns:
            image (Image): resized image
        """
        image_keys: List[str] = ["image", "depthmap"]

        for key, value in data.items():
            if any([k in key for k in image_keys]):
                img_height, img_width = value.shape[0], value.shape[1]

                if isinstance(value, list):
                    assert isinstance(value[0], np.ndarray), "value is not a numpy array"
                    left_pxl = int(self.left * img_width)
                    right_pxl = int(self.right * img_width)
                    top_pxl = int(self.top * img_height)
                    bottom_pxl = int(self.bottom * img_height)
                    data[key] = [v[top_pxl:bottom_pxl, left_pxl:right_pxl] for v in value]
                else:
                    assert isinstance(value, np.ndarray), "value is not a numpy array"
                    left_pxl = int(self.left * img_width)
                    right_pxl = int(self.right * img_width)
                    top_pxl = int(self.top * img_height)
                    bottom_pxl = int(self.bottom * img_height)
                    data[key] = value[top_pxl:bottom_pxl, left_pxl:right_pxl]
        return data


    def _crop_calib(self, data: Dict) -> Dict:
        """Crop the calibration to the given shape.
        Args:
            data (dict): data to resize the calibration of

        Returns:
            data (dict): data with the calibration resized
        """
        for cam in data["intrinsics"]["matrix"].keys():
            img_width, img_height = data["intrinsics"]["size"][cam][0], data["intrinsics"]["size"][cam][1]

            left_pxl = int(self.left * img_width)
            right_pxl = int(self.right * img_width)
            top_pxl = int(self.top * img_height)
            bottom_pxl = int(self.bottom * img_height)

            data["intrinsics"]["matrix"][cam][0, 2] -= left_pxl
            data["intrinsics"]["matrix"][cam][1, 2] -= top_pxl
            data["intrinsics"]["size"][cam] = (right_pxl - left_pxl, bottom_pxl - top_pxl)
            data["extrinsics"]["cam2cam"][cam][:3, :3] = data["intrinsics"]["matrix"][cam][:3, :3]
            data["extrinsics"]["lidar2cam"][cam] = data["extrinsics"]["cam2cam"][cam] @ data["extrinsics"]["lidar2cam"]["original"]

        return data

