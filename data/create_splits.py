import os
from pathlib import Path
import argparse
import h5py
import math
import numpy as np
import json


def write_json(splits, save_path):
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(splits, f, indent=4)


def split_random(keys, train_ratio):
    num_videos = len(keys)
    num_train = int(math.ceil(num_videos * train_ratio))
    train_idxs = np.random.choice(range(num_videos), size=num_train, replace=False)

    train_keys, test_keys = [], []
    for key_idx, key in enumerate(keys):
        if key_idx in train_idxs:
            train_keys.append(key)
        else:
            test_keys.append(key)

    return train_keys, test_keys


def create(args):
    print(f"==========\nArgs: {args}\n==========")
    num_splits = args.num_splits
    split_ratio = args.split_ratio
    dataset_path = args.dataset
    print(f"Goal: randomly split data for {num_splits} times, {int(split_ratio * 100)}% for training and the rest for testing")

    print(f"Loading dataset from {dataset_path}")
    dataset = h5py.File(dataset_path, "r")
    keys = dataset.keys()

    num_videos = len(keys)
    num_train = int(math.ceil(num_videos * split_ratio))
    num_test = num_videos - num_train
    print(f"# total videos: {num_videos}")
    print(f"# train videos: {num_train}")
    print(f"# test videos: {num_test}")

    splits = []
    for split_idx in range(num_splits):
        train_keys, test_keys = split_random(keys, split_ratio)
        splits.append({"train_keys": train_keys, "test_keys": test_keys})

    save_path = Path(args.splits_dir, f"{args.file_name}.json")
    write_json(splits, save_path)
    print(f"Splits have been saved to {save_path}")

    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to create splits in json form")
    parser.add_argument("dataset", type=str, help="path to h5 dataset")
    parser.add_argument("--splits-dir", type=str, default='./splits', help="path to save output json file (default: './splits')")
    parser.add_argument("--file-name", type=str, default="splits", help="name to save as, excluding extension (default: 'splits')")
    parser.add_argument("--num-splits", type=int, default=5, help="how many splits to generate (default: 5)")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="percentage of training data (default: 0.8)")
    args = parser.parse_args()

    create(args)
