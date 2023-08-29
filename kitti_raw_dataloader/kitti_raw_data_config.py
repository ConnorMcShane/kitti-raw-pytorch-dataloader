"""Configuration for the debugger."""
from typing import Dict, OrderedDict
import copy

class KittiRawConfig(object):
    """Config for the KittiRaw dataloader."""

    def __init__(self):
        """Initialize the KittiRawConfig class."""

        super().__init__()

        # Name of the dataset.
        self.dataset: str = "KittiRawDataset"

        # Path to the directory where the data is stored.
        self.root: str = "./example_data/kitti_raw"

        # Split files
        self.splits: Dict[str, str] = {
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
        }

        # Shuffle.
        self.shuffle: bool = True

        # Batch size.
        self.batch_size: int = 1

        # Number of workers.
        self.num_workers: int = 1

        # sensors.
        self.input_sensors: Dict[str, dict] = {
            "camera": {
                "image_00": {
                    "load": False,
                    "folder": "image_00/data",
                    "format": ".png",
                    "loader": "image",
                },
                "image_01": {
                    "load": False,
                    "folder": "image_01/data",
                    "format": ".png",
                    "loader": "image",
                },
                "image_02": {
                    "load": True,
                    "folder": "image_02/data",
                    "format": ".png",
                    "loader": "image",
                },
                "image_03": {
                    "load": False,
                    "folder": "image_03/data",
                    "format": ".png",
                    "loader": "image",
                },
            },
            "lidar": {
                "lidar_points": {
                    "load": True,
                    "folder": "velodyne_points/data",
                    "format": ".bin",
                    "loader": "pointcloud",
                },
                "lidar_bev": {
                    "load": True,
                    "folder": "velodyne_points/data",
                    "format": ".bin",
                    "loader": "pointcloud_bev",
                    "params": {"height": 1024, "width": 1024, "x_range": (-50, 50), "y_range": (-50, 50), "z_max": 5.0},
                },
                "lidar_depthmap": {
                    "load": True,
                    "folder": "velodyne_points/data",
                    "format": ".bin",
                    "loader": "pointcloud_depthmap",
                    "params": {"camera": "image_02"},
                }
            },
            "odometry": {
                "odometry": {
                    "load": True,
                    "folder": "oxts/data",
                    "format": ".txt",
                    "loader": "odometry",
                },
            },
        }

        # calibration.
        self.load_calib: bool = True

        # Number of images for forward and backward context.
        self.image_context_stride: int = 1
        self.image_back_context: int = 1
        self.image_forward_context: int = 1

        # transformations.
        self.transforms: Dict = {}

        # train transforms
        self.transforms["train"]: OrderedDict = OrderedDict()
        self.transforms["train"]["Crop"] = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        self.transforms["train"]["RandomCrop"] = {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        self.transforms["train"]["RandomFlip"] = {"flip_prob": 0.5}
        self.transforms["train"]["Resize"] = {"height": 375, "width": 1242, "interpolation": "area"}
        self.transforms["train"]["Normalise"] = {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "to_rgb": False}
        self.transforms["train"]["ToTensor"] = {}

        # val transforms
        self.transforms["val"]: OrderedDict = copy.deepcopy(self.transforms["train"])
        if "RandomCrop" in self.transforms["val"]:
            del self.transforms["val"]["RandomCrop"]
        if "RandomFlip" in self.transforms["val"]:
            del self.transforms["val"]["RandomFlip"]

        # test transforms
        self.transforms["test"]: OrderedDict = copy.deepcopy(self.transforms["val"])
