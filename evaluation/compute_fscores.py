import os
import sys
from pathlib import Path

_package_path = Path(__file__).parent.absolute()
_package_search_path = _package_path.parent
sys.path.append(str(_package_search_path))

import json
import numpy as np
import h5py

from data.vsum_tool import generate_summary, evaluate_summary

# example usage: python compute_fscores.py <results_dir> TVSum avg
results_dir = Path(sys.argv[1])
dataset = sys.argv[2]
eval_method = sys.argv[3]
print("results_dir:", results_dir)
print("dataset:", dataset)
print("eval_method:", eval_method)

# dataset prefix: {SumMe | TVSum}_
DS_PREFIX_LEN = 6


def epochFromFileName(fileName):
    # file name format: {SumMe | TVSum}_{epoch}.json
    try:
        return int(fileName[DS_PREFIX_LEN:-5])
    except:
        return -1


results = os.listdir(results_dir)
results.sort(key=epochFromFileName)

HOME_PATH = _package_path / "../data"
DATASET_PATH = HOME_PATH / dataset / f"eccv16_dataset_{dataset.lower()}_google_pool5.h5"

# for each epoch, read the results' file and compute the f_score
f_score_epochs = []
for epoch in results:
    print(epoch)
    if epochFromFileName(epoch) < 0:
        print("    Invalid epoch!")
        continue

    all_user_summary, all_summaries = [], []
    with open(results_dir / epoch) as f:
        epoch_results = json.loads(f.read())

        with h5py.File(DATASET_PATH, "r") as hdf:
            video_names = list(epoch_results.keys())
            for video_name in video_names:
                scores = np.asarray(epoch_results[video_name])

                data = hdf[video_name]

                user_summary = np.array(data["user_summary"])
                change_points = np.array(data["change_points"])
                n_frames = np.array(data["n_frames"])
                picks = np.array(data["picks"])

                summary, _, _, _ = generate_summary(scores, change_points, n_frames, picks)

                all_user_summary.append(user_summary)
                all_summaries.append(summary)

    all_f_scores = []
    # compare the resulting summary with the ground truth one, for each video
    for video_index in range(len(all_summaries)):
        summary = all_summaries[video_index]
        user_summary = all_user_summary[video_index]
        f_score, _, _ = evaluate_summary(summary, user_summary, eval_method)
        all_f_scores.append(f_score)

    f_score_epochs.append(np.mean(all_f_scores))
    print("    f_score: ", np.mean(all_f_scores))

with open(results_dir / "f_scores.txt", "w") as outfile:
    json.dump(f_score_epochs, outfile)
