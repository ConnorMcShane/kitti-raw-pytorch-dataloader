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

loader = val_loader

for i, batch in enumerate(loader):

    # save binary image
    image = batch["lidar_bev"][0].cpu().squeeze().numpy() * 255
    image = image.astype(np.uint8)
    cv2.imwrite(f"temp/bev/{str(i).zfill(3)}_image.png", image)

    # save rgb image
    image = batch["image_02"][0].cpu().numpy()
    cv2.imwrite(f"temp/image_02/{str(i).zfill(3)}_image.png", image)

    # save depthmap image
    depthmap = batch["lidar_depthmap"][0].cpu().squeeze().numpy()
    depthmap = depthmap * 255 / depthmap.max()
    cv2.imwrite(f"temp/depthmap/{str(i).zfill(3)}_image.png", depthmap)

    # print odometry
    print(batch["odometry"][0].cpu().numpy())

    break
    if i > 5:
        break
