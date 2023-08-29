"""Resize class for transforms."""

from typing import Dict, Any, List
import cv2
import numpy as np

class Resize():
    """Resize class for transforms."""

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

        # resize image
        data = self._resize_image(data)

        # resize depthmap
        data = self._resize_depthmap(data)

        # resize calib
        data = self._resize_calib(data)

        return data
    

    def _resize_image(self, data: Dict) -> Dict:
        """Resize the image to the given shape.
        Args:
            image (Image): image to resize

        Returns:
            image (Image): resized image
        """
        image_keys: List[str] = ["image"]
        shape = [self.params["height"], self.params["width"]]
        cv2_interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }

        assert self.params["interpolation"] in cv2_interpolation.keys(), f"interpolation must be in {cv2_interpolation.keys()}"
        interpolation = cv2_interpolation[self.params["interpolation"]]

        for key, value in data.items():
            if any([k in key for k in image_keys]):
                if isinstance(value, list):
                    assert isinstance(value[0], np.ndarray), "image is not a numpy array"
                    data[key] = [cv2.resize(v, (shape[1], shape[0]), interpolation=interpolation) for v in value]
                else:
                    assert isinstance(value, np.ndarray), "image is not a numpy array"
                    data[key] = cv2.resize(value, (shape[1], shape[0]), interpolation=interpolation)

        return data
    

    def _resize_depthmap(self, data: Dict) -> Dict:
        """
        Resizes depth map preserving all valid depth pixels
        Multiple downsampled points can be assigned to the same pixel.
        
        Args:
            old_shape: The old shape of the image.
            new_shape: The new shape of the image.
        """
        depthmap_keys: List[str] = ["depthmap"]
        new_shape = [self.params["height"], self.params["width"]]

        for key, value in data.items():
            if any([k in key for k in depthmap_keys]):

                if isinstance(value, list):
                    value = [self._resize_depthmap_preserve(v, new_shape) for v in value]
                else:
                    value = self._resize_depthmap_preserve(value, new_shape)
                data[key] = value

        return data
    

    @staticmethod
    def _resize_depthmap_preserve(depthmap: np.ndarray, new_shape: list) -> np.ndarray:
        """
        Resizes depth map preserving all valid depth pixels
        """
        assert isinstance(depthmap, np.ndarray), "depthmap is not a numpy array"

        # Store dimensions and reshapes to single column
        depthmap = np.squeeze(depthmap)
        h, w = depthmap.shape
        x = depthmap.reshape(-1)
        # Create coordinate grid
        uv = np.mgrid[:h, :w].transpose(1, 2, 0).reshape(-1, 2)
        # Filters valid points
        idx = x > 0
        crd, val = uv[idx], x[idx]
        # Downsamples coordinates
        crd[:, 0] = (crd[:, 0] * (new_shape[0] / h)).astype(np.int32)
        crd[:, 1] = (crd[:, 1] * (new_shape[1] / w)).astype(np.int32)
        # Filters points inside image
        idx = (crd[:, 0] < new_shape[0]) & (crd[:, 1] < new_shape[1])
        crd, val = crd[idx], val[idx]
        # Creates downsampled depthmap image and assigns points
        depthmap = np.zeros(new_shape)
        depthmap[crd[:, 0], crd[:, 1]] = val
        # Return resized depthmap map
        depthmap = np.expand_dims(depthmap, axis=2)
        
        return depthmap


    def _resize_calib(self, data: Dict) -> Dict:
        """Resize the calibration to the given shape.
        Args:
            data (dict): data to resize the calibration of

        Returns:
            data (dict): data with the calibration resized
        """
        new_shape = [self.params["height"], self.params["width"]]

        for cam in data["intrinsics"]["matrix"].keys():
            img_width, img_height = data["intrinsics"]["size"][cam][0], data["intrinsics"]["size"][cam][1]
            data["intrinsics"]["matrix"][cam][0, :3] *= new_shape[1] / img_width
            data["intrinsics"]["matrix"][cam][1, :3] *= new_shape[0] / img_height
            data["intrinsics"]["size"][cam] = (new_shape[1], new_shape[0])
            data["extrinsics"]["cam2cam"][cam][:3, :3] = data["intrinsics"]["matrix"][cam][:3, :3]
            data["extrinsics"]["lidar2cam"][cam] = data["extrinsics"]["cam2cam"][cam] @ data["extrinsics"]["lidar2cam"]["original"]

        return data
