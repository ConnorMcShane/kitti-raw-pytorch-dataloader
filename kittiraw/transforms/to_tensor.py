"""ToTensor class for transforms."""

from typing import Dict, Any
import numpy as np
import torch
from torchvision.transforms import ToTensor as TorchToTensor

class ToTensor():
    """ToTensor class for transforms."""

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
        torchtotensor = TorchToTensor()
        tensortype =  "torch.FloatTensor"
        # Convert single items and list items
        img_keys = ["image", "depthmap", "bev"]
        list_keys = ["context"]
        for key in data.keys():
            if any([k in key for k in list_keys]):
                if any([k in key for k in img_keys]):
                    data[key] = [torchtotensor(k).type(tensortype) for k in data[key]]
            elif any([k in key for k in img_keys]):
                data[key] = torchtotensor(data[key]).type(tensortype)
            
            if isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key]).type(tensortype)
            elif isinstance(data[key], list):
                if len(data[key]) > 0 and isinstance(data[key][0], np.ndarray):
                    data[key] = [torch.from_numpy(k).type(tensortype) for k in data[key]]

        return data
