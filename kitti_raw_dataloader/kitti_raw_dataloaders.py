"""KittiRaw dataloaders."""
from typing import Dict

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .kitti_raw_dataset import KittiRawDatatset
from .kitti_raw_data_config import KittiRawConfig

data_config = KittiRawConfig().__dict__


def get_dataloaders() -> Dict[str, DataLoader]:
    """Get dataloaders."""
    dataloaders = {}
    modes = ["train", "val", "test"]
    for mode in modes:
        dataloaders[mode] = get_dataloader(mode)
    return dataloaders


def get_dataloader(mode: str) -> DataLoader:
    """Get dataloader."""
    dataset = get_dataset(data_config, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=data_config["shuffle"],
        num_workers=data_config["num_workers"],
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )
    return dataloader


def get_dataset(cfg: Dict, mode: str) -> Dataset:
    """Get dataset."""
    dataset = KittiRawDatatset(cfg, mode=mode)
    return dataset
