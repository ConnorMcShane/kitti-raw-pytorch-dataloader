import cv2
import numpy as np
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from kitti_raw_dataloader.kitti_raw_dataloaders import get_dataloaders

dataloaders = get_dataloaders()
train_loader = dataloaders["train"]
val_loader = dataloaders["val"]

loader = train_loader

means = loader.dataset.cfg["transforms"]["train"]["Normalise"]["mean"]
stds = loader.dataset.cfg["transforms"]["train"]["Normalise"]["std"]

for i, batch in enumerate(loader):

    # print all keys and their types and shapes
    for key, value in batch.items():
        print(key, type(value))

    # save binary image
    image = batch["lidar_bev"][0].cpu().squeeze().numpy() * 255
    image = image.astype(np.uint8)
    cv2.imwrite(f"temp/bev/{str(i).zfill(3)}_image.png", image)

    # save rgb image
    image = batch["image_02"][0].detach().permute(1, 2, 0).cpu().numpy()
    image = (image * stds) + means
    cv2.imwrite(f"temp/image_02/{str(i).zfill(3)}_image.png", image)

    # save depthmap image
    depthmap = batch["lidar_depthmap"][0].cpu().squeeze().numpy()
    depthmap = depthmap * 255 / depthmap.max()
    cv2.imwrite(f"temp/depthmap/{str(i).zfill(3)}_image.png", depthmap)
    
    break
