"""KittiRaw Dataset"""

from typing import Dict
import os
import copy

import torch
import numpy as np
import pickle
import json
import cv2

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transforms import TRANSFORMS
from .utils.loaders import Loaders

class KittiRawDatatset(Dataset):
    """KittiRaw Dataset Class"""


    def __init__(self, cfg: Dict, mode: str = "train") -> None:
        """KittiRaw Dataset
        
        Args:
            cfg (Dict): config
            mode (str, optional): train or val. Defaults to "train".
        """

        self._assertations(cfg, mode)

        self.cfg = cfg
        self.mode = mode
        self.split = self.cfg["splits"][self.mode]
        self.loaders = Loaders(self.cfg)
        self._set_transforms()
        self._set_samples()

        print(len(self), self.mode, "samples available")


    def __len__(self) -> int:
        """__len__"""
        return len(self.samples)
    

    def _assertations(self, cfg: Dict, mode: str) -> None:
        """assertations to set limits of the dataset
        
        Args:
            cfg (Dict): config
            mode (str, optional): train or val. Defaults to "train".
        """
        assert mode in ["train", "val", "test"], "mode must be train or val or test"
        assert isinstance(cfg, dict), "cfg must be a dict"


    def _set_transforms(self) -> None:
        """Initialise transforms"""
        # setup transforms dict
        self.transforms_dict = {}

        # initialise transforms
        for transform, params in self.cfg["transforms"][self.mode].items():
            self.transforms_dict[transform] = TRANSFORMS[transform](params)

        # compose transforms
        self.transforms: Compose = Compose(self.transforms_dict.values())


    def _set_samples(self) -> None:
        """Set samples"""
        # load samples
        with open(os.path.join(self.cfg["root"], self.split), "r") as f:
            self.samples = f.read().splitlines()

        # shuffle samples
        if self.cfg["shuffle"]:
            np.random.shuffle(self.samples)


    def __getitem__(self, idx: int) -> Dict:
        """return sample"""
        sample_path = self.samples[idx].split(" ")[0]
        sample_id = self.samples[idx].split(" ")[1]

        sample = {}
        sample["id"] = sample_id

        for sensor in self.cfg["input_sensors"]:
            for sensor_name, sensor_params in self.cfg["input_sensors"][sensor].items():
                if sensor_params["load"]:
                    filepath = os.path.join(self.cfg["root"], sample_path, sensor_params["folder"], sample_id + sensor_params["format"])
                    if "params" in sensor_params:
                        sample[sensor_name] = self.loaders.load_dict[sensor_params["loader"]](filepath, **sensor_params["params"])
                    else:
                        sample[sensor_name] = self.loaders.load_dict[sensor_params["loader"]](filepath)

        # transform sample
        sample = self.transforms(sample)
        
        return sample