from pathlib import Path
import h5py
import numpy as np
import time

import sys
from pathlib import Path

package_path = Path(__file__).parent.absolute()
package_search_path = package_path.parent
sys.path.append(str(package_search_path))

from data import vsum_tool, SumMeVideo, TVSumVideo, VSUMMVideo
import torch


dataset_dir = package_path / "../../datasets"

# summe_dir = dataset_dir / "SumMe"
# video = SumMeVideo("Air_Force_One", summe_dir)  # video_1
# print(video.filename)
# print(f"    number of frames: {video.nframes} {video.fps}")
# ground_truth = video.ground_truth()
# print(f"    ground truth: {type(ground_truth)} {ground_truth.dtype} {ground_truth.shape} {ground_truth.min()}~{ground_truth.max()}")

tvsum_dir = dataset_dir / "ydata-tvsum50-v1_1"
video = TVSumVideo("xxdtq8mxegs", tvsum_dir)    # video_15
print(video.filename)
print(f"    number of frames: {video.nframes} {video.fps}")
ground_truth = video.ground_truth()
print(f"    ground truth: {type(ground_truth)} {ground_truth.dtype} {ground_truth.shape} {ground_truth.min()}~{ground_truth.max()}")
ground_truth_0 = (ground_truth[0]).astype(np.uint8)
print(f"ground_truth_0: {ground_truth_0.shape} {ground_truth_0.min()}~{ground_truth_0.max()}")

# vsumm_dir = dataset_dir / "VSUMM"
# video = VSUMMVideo("v21", vsumm_dir)
# print(video.filename)
# print(f"    number of frames: {video.nframes} {video.fps}")
# ground_truth = video.ground_truth()
# print(f"    ground truth: {type(ground_truth)} {ground_truth.dtype} {ground_truth.shape} {ground_truth.min()}~{ground_truth.max()}")

# filePath = package_path / "./SumMe/eccv16_dataset_summe_google_pool5.h5"
filePath = package_path / "./TVSum/eccv16_dataset_tvsum_google_pool5.h5"
print(filePath)
hdf = h5py.File(filePath, "r")

keys = list(hdf.keys())
print(f"video list: {len(keys)} videos")
# print(keys)
# key = keys[0]
key = "video_15"
print(f"video name: {key}")
data = hdf[key]
print(f"video dict keys: {list(data.keys())}")

change_points = np.array(data["change_points"])
features = np.array(data["features"])
# n_frame_per_seg = np.array(data["n_frame_per_seg"])
n_frames = np.array(data["n_frames"])
n_steps = np.array(data["n_steps"])
picks = np.array(data["picks"])
user_summary = np.array(data["user_summary"])

print(f"change_points: {change_points.dtype} {change_points.shape}")
# print(change_points)
print(f"features: {features.dtype} {features.shape} {features.min()}~{features.max()}", features[0, :].sum(), features[:, 0].sum())
# print(f"n_frame_per_seg: {n_frame_per_seg.shape} {n_frame_per_seg}")
print(f"n_frames: {n_frames}")
print(f"n_steps: {n_steps}")
print(f"picks: {picks.shape}")
zero_or_one = ((0 < user_summary[0]) & (user_summary[0] < 1)).sum() == 0
print(f"user_summary: {user_summary.dtype} {user_summary.shape} {user_summary.min()}{'|'if zero_or_one else '~'}{user_summary.max()}")
user_summary = user_summary.astype(np.uint8)
# print(list(user_summary[0]))
print("=> ground_truth_0 == user_summary[0]:", np.array_equal(ground_truth_0, user_summary[0]),
      (ground_truth_0 == user_summary[0]).sum())
if "video_name" in data:
    video_name = np.array(data["video_name"])
    print(f"video_name: {video_name}")

hdf.close()

filePath = dataset_dir / "xxdtq8mxegs_dataset.h5"
print(filePath)
hdf = h5py.File(filePath, "r")
keys = list(hdf.keys())
print(f"video list2: {len(keys)} videos")
# print(keys)
key = keys[0]
data = hdf[key]
print(f"video dict keys2: {list(data.keys())}")
change_points2 = np.array(data["change_points"])
features2 = np.array(data["features"])
n_frames2 = np.array(data["n_frames"])
# n_frame_per_seg2 = np.array(data["n_frame_per_seg"])
picks2 = np.array(data["picks"])
picks2 = np.array(range(n_frames2))
print(f"change_points2: {change_points2.dtype} {change_points2.shape}")
# print(change_points2)
print(f"features2: {features2.dtype} {features2.shape} {features2.min()}~{features2.max()}", features2[0, :].sum(), features2[:, 0].sum())
# print(f"n_frame_per_seg2: {n_frame_per_seg2.shape} {n_frame_per_seg2}")
print(f"n_frames2: {n_frames2}")
# print(f"n_steps2: {n_steps2}")
print(f"picks2: {picks2.shape}")
print("=> change_points == change_points2:", np.array_equal(change_points, change_points2))
print("=> features == features2:", np.array_equal(features, features2))
print("=> n_frames == n_frames2:", np.array_equal(n_frames, n_frames2))
# print("=> n_frame_per_seg == n_frame_per_seg2:", np.array_equal(n_frame_per_seg, n_frame_per_seg2))
print("=> picks == picks2:", np.array_equal(picks, picks2))

# for vsum_tool.generate_summary
probs = ground_truth_0
cps = change_points2.tolist()
cps = change_points.tolist()
n_frames3 = n_frames2
positions = picks2

repeat = 300
start_time = time.time()
for i in range(repeat):
    user_summary3_0, _, _, _ = vsum_tool.generate_summary(probs, cps, n_frames3, positions)
print(f"vsum_tool.generate_summary() took {time.time() - start_time:.2f} secs.")
print("=> user_summary3_0 == user_summary[0]:", np.array_equal(user_summary3_0, user_summary[0]),
      np.equal(user_summary3_0, user_summary[0]).sum())

hdf.close()
