from .KTS import cpd_auto
from .knapsack import knapsack

import os
import numpy as np
import math
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def get_change_points(video_feat, n_frames, fps,
                      decimation_factor=1, min_cp_interval=5):
    kernel = np.matmul(video_feat, video_feat.T)
    ncp_1cpps = n_frames / fps  # number of cp at a rate of 1 cp per sec
    max_num_cp = int(math.floor(ncp_1cpps / min_cp_interval))

    change_points, _ = cpd_auto(kernel, max_num_cp, 1)
    change_points *= decimation_factor

    change_points = np.concatenate(([0], change_points, [n_frames]))
    begin_points = change_points[:-1]
    end_points = change_points[1:]

    change_points = np.vstack((begin_points, end_points - 1)).T
    n_frame_per_seg = end_points - begin_points

    return change_points, n_frame_per_seg


def generate_summary(importance_scores, change_points, n_frames, picks, proportion=0.15,
                     save_as_video=False, video_path=None, summary_dir=None,
                     save_frames=False, save_keyframes=True):
    """
    Generate keyshot-based video summary. i.e. a binary vector

    Args:
        importance_scores: predicted importance scores.
        change_points: 2D matrix, each row contains a segment.
        n_frames: original number of frames.
        picks: positions of subsampled frames in the original video.
        proportion: length of video summary (compared to original video length).
    """
    assert(importance_scores.shape == picks.shape)

    picks = picks.astype(int)
    if picks[-1] != n_frames:
        picks = np.concatenate([picks, [n_frames]])

    # Compute the importance scores for the initial frame sequence (not the subsampled one)
    frame_scores = np.zeros(n_frames)
    for i in range(len(picks) - 1):
        score = importance_scores[i] if i < len(importance_scores) else 0
        frame_scores[picks[i]:picks[i + 1]] = score

    # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
    seg_scores = []
    nfps = []
    for segment in change_points:
        seg_begin, seg_end = segment + np.asarray((0, 1))
        seg_score = frame_scores[seg_begin:seg_end].mean()
        seg_scores.append(float(seg_score))
        nfps.append(int(seg_end - seg_begin))

    # Select the best shots
    limit = int(math.floor(n_frames * proportion))
    keyshots = knapsack(seg_scores, nfps, limit)
    # print("keyshots:", keyshots)

    # Select all frames from each selected shot (by setting their value in the summary vector to 1)
    summary = np.zeros(n_frames, dtype=np.uint8)
    for seg_idx in keyshots:
        seg_begin, seg_end = change_points[seg_idx] + np.asarray((0, 1))
        summary[seg_begin:seg_end] = 1

    keyshots = np.asarray(change_points)[keyshots]
    keyframes = [begin + np.argmax(frame_scores[begin:end]) for begin, end in keyshots + (0, 1)]
    # print("keyframes:", keyframes)

    if save_as_video:
        summary_duration = _save_summary(
            video_path, summary, keyframes, summary_dir, save_frames, save_keyframes)
    else:
        summary_duration = None

    return summary, keyshots, keyframes, summary_duration


def evaluate_summary(machine_summary, user_summary, eval_method="avg"):
    """
    Compare machine summary with user summary (Keyshot-based).

    Args:
        machine_summary: summary by machine
        user_summary: summary by user (annotation)
        eval_method: {'avg', 'max'}
            'avg' : average results of comparing multiple human summaries.
            'max' : takes the maximum(best) out of multiple comparisons.
    """

    pred_len = len(machine_summary)
    num_users, n_frames = user_summary.shape
    assert(pred_len == n_frames)
    # if n_frames < pred_len:
    #     machine_summary = machine_summary[:n_frames]
    # elif pred_len < n_frames:
    #     zero_padding = np.zeros(n_frames - pred_len)
    #     machine_summary = np.concatenate([machine_summary, zero_padding])

    if eval_method not in ["avg", "max"]:
        print("Unsupported evaluation method:", eval_method)
        eval_method = "avg"

    # binarization
    machine_summary = (0 < machine_summary).astype(int)
    user_summary = (0 < user_summary).astype(int)

    epsilon = 1e-8
    f_scores = []
    precisions = []
    recalls = []
    for user_idx in range(num_users):
        gt_summary = user_summary[user_idx]
        overlapped = (machine_summary * gt_summary).sum()

        precision = overlapped / (machine_summary.sum() + epsilon)
        recall = overlapped / (gt_summary.sum() + epsilon)
        if (precision == 0) and (recall == 0):
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall) * 100

        f_scores.append(f_score)
        precisions.append(precision)
        recalls.append(recall)

    if eval_method == "avg":
        final_f_score = np.mean(f_scores)
        final_precision = np.mean(precisions)
        final_recall = np.mean(recalls)
    elif eval_method == "max":
        max_idx = np.argmax(f_scores)
        final_f_score = f_scores[max_idx]
        final_precision = precisions[max_idx]
        final_recall = recalls[max_idx]

    return final_f_score, final_precision, final_recall


def _save_summary(video_path, summary, keyframes, summary_dir=None,
                  save_frames=False, save_keyframes=True):
    if not video_path:
        return

    video_name = video_path.stem
    if not summary_dir:
        summary_dir = video_path.parent / f"{video_name}_summary"
    frames_dir = summary_dir / "frames"
    keyframes_dir = summary_dir / "keyframes"
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(keyframes_dir, exist_ok=True)

    summary_path = summary_dir / "summary.avi"

    reader = cv2.VideoCapture(str(video_path))
    n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    fps = reader.get(cv2.CAP_PROP_FPS)
    frame_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(summary_path), fourcc, fps, (frame_width, frame_height))

    print("saving summary ...")
    n_frames_summary = 0
    for frame_idx in tqdm(range(n_frames)):
        success, frame = reader.read()
        if not success:
            break

        if save_frames:
            decimation_factor = 15
            if (frame_idx % decimation_factor) == 0:
                frame_path = frames_dir / f"Frame{frame_idx}.jpeg"
                cv2.imwrite(str(frame_path), frame)

        if not summary[frame_idx]:
            continue

        # reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # success, frame = reader.read()
        # assert(success)

        if save_keyframes and (frame_idx in keyframes):
            keyframe_path = keyframes_dir / f"Frame{frame_idx}.jpeg"
            cv2.imwrite(str(keyframe_path), frame)

        writer.write(frame)
        n_frames_summary += 1

    writer.release()
    reader.release()

    summary_duration = n_frames_summary / fps
    return summary_duration


def plot_result(video_name, pred_summary, ground_truth_summary,
                show=True, save_path=None):
    assert(len(pred_summary) == len(ground_truth_summary))
    frame_indexes = list(range(len(ground_truth_summary)))

    sns.set()
    plt.title(video_name)

    colors = ["lightseagreen" if i == 0 else "orange" for i in pred_summary]
    plt.bar(x=frame_indexes, height=ground_truth_summary, color=colors,
            edgecolor=None, linewidth=0)

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)
