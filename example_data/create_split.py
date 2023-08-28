"""create split for train, val, test for raw kitti dataset. Samples will be split by drive id."""

import os
import random
import argparse


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('--root', type=str, default='./example_data/kitti_raw', help='path to the dataset root directory')
    parser.add_argument('--test_split', type=float, default=0.2, help='test split') # 20% of data for testing, 80% for training and validation
    parser.add_argument('--val_split', type=float, default=0.3, help='validation split') # 30% of non-test data for validation, 70% for training
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the dataset')
    parser.add_argument('--save_file', type=bool, default=True, help='save the split files')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0])
    args = parser.parse_args()
    return args


def create_split(root, test_split, val_split, shuffle=True, save_file=True, seed=42):
    """create split for train, val, test for raw kitti dataset. Samples will be split by drive id."""

    random.seed(seed)
    
    collection_days = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
    drive_paths = []
    for i, day in enumerate(collection_days):
        daily_drive_ids = [x for x in os.listdir(os.path.join(root, day)) if os.path.isdir(os.path.join(root, day, x))]
        for drive_id in daily_drive_ids:
            drive_paths.append(os.path.join(day, drive_id))

    if shuffle:
        random.shuffle(drive_paths)
    
    trainval_paths = drive_paths[int(len(drive_paths)*test_split):]
    train_paths = trainval_paths[int(len(trainval_paths)*val_split):]
    val_paths = trainval_paths[:int(len(trainval_paths)*val_split)]
    test_paths = drive_paths[:int(len(drive_paths)*test_split)]

    train_ids = []
    val_ids = []
    test_ids = []
    trainval_ids = []

    for train_path in train_paths:
        image_02_path = os.path.join(root, train_path, "image_02", "data")
        path_samples = [x.split(".")[0] for x in os.listdir(image_02_path) if x.endswith(".png")]
        train_path_ids = [train_path + " " + x for x in path_samples]
        train_ids.extend(train_path_ids)

    for val_path in val_paths:
        image_02_path = os.path.join(root, val_path, "image_02", "data")
        path_samples = [x.split(".")[0] for x in os.listdir(image_02_path) if x.endswith(".png")]
        val_path_ids = [val_path + " " + x for x in path_samples]
        val_ids.extend(val_path_ids)
    
    for test_path in test_paths:
        image_02_path = os.path.join(root, test_path, "image_02", "data")
        path_samples = [x.split(".")[0] for x in os.listdir(image_02_path) if x.endswith(".png")]
        test_path_ids = [test_path + " " + x for x in path_samples]
        test_ids.extend(test_path_ids)
    
    for trainval_path in trainval_paths:
        image_02_path = os.path.join(root, trainval_path, "image_02", "data")
        path_samples = [x.split(".")[0] for x in os.listdir(image_02_path) if x.endswith(".png")]
        trainval_path_ids = [trainval_path + " " + x for x in path_samples]
        trainval_ids.extend(trainval_path_ids)

    if save_file:
        with open(os.path.join(root, "train.txt"), "w") as f:
            for train_id in train_ids:
                f.write(train_id+"\n")

        with open(os.path.join(root, "val.txt"), "w") as f:
            for val_id in val_ids:
                f.write(val_id+"\n")

        with open(os.path.join(root, "trainval.txt"), "w") as f:
            for trainval_id in trainval_ids:
                f.write(trainval_id+"\n")

        with open(os.path.join(root, "test.txt"), "w") as f:
            for test_id in test_ids:
                f.write(test_id+"\n")
    
    print(f"train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    return train_ids, val_ids, test_ids


if __name__ == '__main__':
    args = parse_args()
    create_split(
        args.root, 
        args.test_split,
        args.val_split,
        args.shuffle,
        args.save_file,
        args.seed
    )
