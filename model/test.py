import sys
from pathlib import Path

package_path = Path(__file__).parent.absolute()
package_search_path = package_path.parent
sys.path.append(str(package_search_path))

import numpy as np
import cv2
from PIL import Image
import h5py
from tqdm import tqdm
import json

from model.layers import FeatureExtractor

from model.configs import get_config
from model.solver import Solver
from model.data_loader import get_loader

from data import TVSumVideo, DatasetBuilder, vsum_tool


dataset_dir = package_path / "../../datasets"

# real_name: indexed_name
with open(dataset_dir / "tvsum_mapped_video_names.json") as f:
    _name_dict = json.load(f)
    tvsum_name_dict = {_name_dict[key]: key for key in _name_dict}

tvsum_dir = dataset_dir / "ydata-tvsum50-v1_1"
video = TVSumVideo("xxdtq8mxegs", tvsum_dir)    # video_15

video_path = Path(video.filename)
video = None
video_path = dataset_dir / "e3f6787b-ab57-4d07-8492-b4a7e51f71b2.avi"
print(video_path)
video_name = video_path.stem

# featureExtractor = FeatureExtractor()     # googlenet
# # print(featureExtractor)

# video_capture = cv2.VideoCapture(str(video_path))

# print("Extracting features ...")
# decimation_factor = 15
# features = []
# picks = []
# n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# for frame_idx in tqdm(range(n_frames)):
#     success, frame = video_capture.read()
#     if not success:
#         break

#     pick_frame = (frame_idx % decimation_factor) == 0
#     if (not pick_frame):
#         continue

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = Image.fromarray(frame)
#     feature = featureExtractor(frame)
#     feature = feature.cpu().detach().numpy()

#     features.append(feature)
#     picks.append(frame_idx)
# features = np.asarray(features)
# picks = np.asarray(picks)

save_path = dataset_dir / f"{video_name}_dataset.h5"
if not save_path.is_file():
    builder = DatasetBuilder(video_path, save_path)
    builder.build()
    builder.close()

hdf = h5py.File(save_path, 'r')
keys = list(hdf.keys())
print(f"video list: {len(keys)} videos")
# print(keys)
key = keys[0]
print("video name:", key)
data = hdf[key]
print("video dict keys:", list(data.keys()))
change_points = np.array(data["change_points"])
features = np.array(data["features"])
n_frame_per_seg = np.array(data["n_frame_per_seg"])
n_frames = np.array(data["n_frames"])
picks = np.array(data["picks"])
hdf.close()

print(f"change_points: {change_points.dtype} {change_points.shape}")
# print(change_points)
print(f"features: {features.dtype} {features.shape} {features.min()}~{features.max()}")
# print(features)
print(f"n_frame_per_seg: {n_frame_per_seg}")
print(f"n_frames: {n_frames}")
print(f"picks: {picks.dtype} {picks.shape}")
features_0 = features[0]
print(f"features_0: {features_0.shape} {features_0}\n")

filePath = Path("../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5")
print(filePath)
hdf2 = h5py.File(filePath, 'r')
key2 = "video_15" # xxdtq8mxegs
print("video name:", key2)
data2 = hdf2[key2]
print("video dict keys:", list(data2.keys()))
change_points2 = np.array(data2["change_points"])
print(f"change_points2: {change_points2.dtype} {change_points2.shape}")
# print(change_points)
features2 = np.array(data2["features"])
print(f"features2: {features2.dtype} {features2.shape} {features2.min()}~{features2.max()}")
# picks2 = np.array(data["picks"])
# print(f"picks2: {picks2.dtype} {picks.shape}")
user_summary = np.array(data2["user_summary"])
print(f"user_summary: {user_summary.dtype} {user_summary.shape} {user_summary.min()}~{user_summary.max()}")
features2_0 = features2[0]
print(f"features2_0: {features2_0.shape} {features2_0}")
diff_count = (1e-8 < (features_0 - features2_0)).sum()
print("features_0 == features2_0:", diff_count == 0, diff_count, "\n")
hdf2.close()

# Result of running evaluation/pipeline.sh
# Once you've run the pipeline you can simply get the result again as below:
#   cd evaluation
#   python choose_best_epoch.py ../exp1 TVSum
sigma = 0.8
split_index = 4
epoch = 79

config = get_config(mode="test", regularization_factor=sigma, split_index=split_index)
print(f"config: mode={config.mode}, sigma={config.regularization_factor},",
      f"split_index={config.split_index}, action_state_size={config.action_state_size}")

train_loader = None
test_loader = get_loader(config.mode, config.split_index)
solver = Solver(config, train_loader, test_loader)

solver.build()
solver.load_model_state(epoch)
print("model loaded!")

# solver.evaluate(-1)
# solver.evaluate(-2)

if video:   # use public dataset
    features = features2
    change_points = change_points2

print("Predicting scores ...")
num_prediction = 1000
scores = []
for i in tqdm(range(num_prediction)):
    scores.append(solver.predict(features))
scores = np.asarray(scores).mean(axis=0)
print("scores:", type(scores), scores.dtype, scores.shape)
# print(scores)

change_points = change_points.tolist()
pred_summary, keyshots, keyframes, duration = vsum_tool.generate_summary(
    scores, change_points, n_frames, picks,
    save_as_video=True, video_path=video_path)
num_keyshots = len(keyshots)
print(f"summary: {duration} seconds, {num_keyshots} keyshots")

if video:
    ground_truth_summary = video.ground_truth().mean(axis=0)
    # print(ground_truth_summary.dtype, ground_truth_summary.shape, ground_truth_summary.min(), ground_truth_summary.max())
    show = False
    save_path = dataset_dir / f"{video.name}_result.png"
    vsum_tool.plot_result(video.name, pred_summary, ground_truth_summary, show=show, save_path=save_path)
