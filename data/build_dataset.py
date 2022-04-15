import sys
from pathlib import Path

_package_path = Path(__file__).parent.absolute()
_package_search_path = _package_path.parent
sys.path.append(str(_package_search_path))

import os
import argparse
from data import DatasetBuilder
import csv
import h5py
import numpy as np


def build(video_path, save_path):
    builder = DatasetBuilder(video_path, save_path)
    builder.build()
    builder.close()


def merge(datasets_path, save_path, video_name_dict):
    merged_file = h5py.File(save_path, "w")

    num_video = 0
    for root, _, files in os.walk(datasets_path):
        for file in files:
            filePath = Path(root, file)
            if filePath.suffix.lower() != ".h5":
                continue

            file = h5py.File(filePath, "r")
            for key in file:
                data = file[key]
                if not data.keys():
                    continue

                num_video += 1
                if video_name_dict:
                    video_name = str(np.array(data["video_name"]))
                    merge_key = video_name_dict[video_name]
                else:
                    merge_key = f"video_{num_video}"

                merged_file.copy(data, merge_key)

    merged_file.close()
    print(f"{num_video} video files have been merged.")


parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="path of a video file or a folder having video or h5 files.")
parser.add_argument("--save_path", type=str, default="./dataset.h5")
parser.add_argument("--merge", action="store_true", help="merge h5 files in the data_path")
parser.add_argument("--labels_path", type=str, default=None)
args = parser.parse_args()

if args.merge:
    video_name_dict = {}
    if args.labels_path:
        with open(args.labels_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            for (i, row) in enumerate(reader):
                if i == 0:  # header
                    continue
                video_name = row[0]
                video_name_dict[video_name] = f"video_{i}"
    merge(args.data_path, args.save_path, video_name_dict)
else:
    build(args.data_path, args.save_path)
