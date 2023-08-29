"""Normalise transform."""

from typing import Dict, List, Any
import cv2
import numpy as np

class Normalise():
    """Normalise transform."""

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

        Example:
            >>> transform = Normalise()
            >>> data = {'image': image}
            >>> data = transform(data)
        """
        image_keys: List[str] = ["image"]

        for key, value in data.items():
            if any([k in key for k in image_keys]):
                if isinstance(value, list):
                    data[key] = [self._normalise_image(v) for v in value]
                else:
                    data[key] = self._normalise_image(value)
        return data


    def _normalise_image(self, image: np.ndarray) -> np.ndarray:
        """Normalise the image.
        Args:
            image (np.ndarray): image to normalise

        Returns:
            image (np.ndarray): normalised image
        """
        assert isinstance(image, np.ndarray), "image is not a numpy array"
        if self.params["to_rgb"]:
            # Convert the image to RGB if necessary
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalise each color channel using its mean and standard deviation
        image = (image - self.params["mean"]) / self.params["std"]

        return image

