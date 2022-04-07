import sys
from pathlib import Path

_package_path = Path(__file__).parent.absolute()
_package_search_path = _package_path.parent
sys.path.append(str(_package_search_path))

import argparse
from data import DatasetBuilder

parser = argparse.ArgumentParser()
parser.add_argument("video_path", type=str, help="path of a video file or a folder having video files.")
parser.add_argument("--save_path", type=str, default="./dataset.h5")
args = parser.parse_args()

video_path = args.video_path
save_path = args.save_path

builder = DatasetBuilder(video_path, save_path)
builder.build()
builder.close()
