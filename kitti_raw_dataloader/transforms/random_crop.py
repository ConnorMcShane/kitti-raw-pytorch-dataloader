"""RandomCrop the image"""

from typing import Dict, Any
import random

from .crop import Crop


class RandomCrop(Crop):
    """RandomCrop the image"""

    def __init__(self, params: Dict[str, Any], cfg: dict) -> None:
        """Initialize the RandomCrop transform class.
        Args:
            params (dict): parameters for the transform
        """
        super().__init__(params, cfg)


    def _calc_pixels(self) -> None:
        """Calculate the number of pixels to crop."""
        # get random crop params
        self.left = float(random.uniform(0, 1) * self.params["left"])
        self.right = float(1 - (random.uniform(0, 1) * self.params["right"]))
        self.top = float(random.uniform(0, 1) * self.params["top"])
        self.bottom = float(1 - (random.uniform(0, 1) * self.params["bottom"]))
