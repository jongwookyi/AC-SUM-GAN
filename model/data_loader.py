import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, split_index, dataset_file_path, splits_file_path):
        self.mode = mode
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        self.dataset_file_path = dataset_file_path
        self.hdf = h5py.File(dataset_file_path, "r")

        self.splits_file_path = splits_file_path
        with open(splits_file_path) as f:
            splits = json.loads(f.read())
            for i, split in enumerate(splits):
                if i == split_index:
                    self.split = split
                    self.video_names = split[f"{mode}_keys"]
                    self.len = len(self.video_names)
                    break

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        video_name = self.video_names[index]    # gets the current video name
        frame_features = torch.tensor(np.array(self.hdf[f"{video_name}/features"]))
        return frame_features, video_name


def get_loader(config):
    mode = config.mode
    split_index = config.split_index
    dataset_file_path = config.dataset_path
    splits_file_path = config.splits_path

    dataset = VideoData(mode, split_index, dataset_file_path, splits_file_path)
    if mode.lower() == "train":
        return DataLoader(dataset, batch_size=1, shuffle=True)
    else:
        return dataset


if __name__ == "__main__":
    pass
