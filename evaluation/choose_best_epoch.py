from pathlib import Path
import csv
import json
import sys
import torch
import numpy as np

"""
Chooses the best F-score (among 100 epochs) based on a criterion (Reward & Actor_loss).
Takes as input the path to .csv file with all the loss functions and a .txt file with the F-Scores (for each split).
Prints a scalar that represents the average best F-score value.
"""


def use_logs(logs_file, f_scores):
    losses = {}
    losses_names = []

    with open(logs_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                for col in range(len(row)):
                    losses[row[col]] = []
                    losses_names.append(row[col])
            else:
                for col in range(len(row)):
                    losses[losses_names[col]].append(float(row[col]))

    # criterion: Reward & Actor_loss
    actor = losses["actor_loss_epoch"]
    reward = losses["reward_epoch"]

    actor_t = torch.tensor(actor)
    reward_t = torch.tensor(reward)

    # Normalize values
    actor_t = abs(actor_t)
    actor_t = actor_t / max(actor_t)
    reward_t = reward_t / max(reward_t)

    product = (1 - actor_t) * reward_t

    epoch = torch.argmax(product)

    return np.round(f_scores[epoch], 2), epoch


# example usage: python choose_best_epoch.py <exp_dir> TVSum
exp_dir = Path(sys.argv[1])
dataset = sys.argv[2]

NUM_SPLITS = 5
NUM_SIGMAS = 10

# For each "sigma" value, compute the best F-Score of each split based on the criterion
all_fscores = np.zeros((NUM_SPLITS, NUM_SIGMAS, 2), dtype=float)    # fscore, epoch
for i in range(NUM_SIGMAS):
    sigma = 0.1 * (i + 1)
    # change this path if you use different structure for your directories inside the experiment
    path = exp_dir / f"{dataset}/sigma{sigma:.1f}"
    for split in range(NUM_SPLITS):
        split_dir = f"split{split}"
        results_file = path / "results" / split_dir / "f_scores.txt"
        logs_file = path / "logs" / split_dir / "scalars.csv"

        # read F-Scores
        with open(results_file) as f:
            f_scores = json.loads(f.read())  # list of F-Scores

        # best F-Score based on train logs
        all_fscores[split, i] = use_logs(logs_file, f_scores)

all_fscore_epochs = all_fscores[:, :, 1].astype(int)
all_fscores = all_fscores[:, :, 0]
print("All F1 Scores:\n", all_fscores)
print("=> epoch:\n", all_fscore_epochs)

best_index_per_split = np.argmax(all_fscores, axis=1)
best_per_split = all_fscores[range(NUM_SPLITS), best_index_per_split]
best_sigma_per_split = (best_index_per_split + 1) * 0.1
best_epoch_per_split = all_fscore_epochs[range(NUM_SPLITS), best_index_per_split]
# best_per_split = np.max(all_fscores, axis=1)
print("Best F1 Score per Split:", best_per_split)
print("=> index:", best_index_per_split)
print("=> sigma:", best_sigma_per_split)
print("=> epoch:", best_epoch_per_split)

best_index = np.argmax(best_per_split)
best_fscore = best_per_split[best_index]
# best_fscore = np.mean(best_per_split)
print("Best F1 Score:", best_fscore)
print("=> sigma, split, epoch:", best_sigma_per_split[best_index], best_index, best_epoch_per_split[best_index])
print("Mean F1 Score:", np.mean(best_per_split))
