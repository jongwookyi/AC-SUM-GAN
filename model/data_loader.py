from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

DATASET_NAME_SUMME = "SumMe"
DATASET_NAME_TVSUM = "TVSum"


class VideoData(Dataset):
    def __init__(self, mode, split_index, dataset_name=DATASET_NAME_TVSUM):
        self.mode = mode
        self.split_index = split_index  # it represents the current split (varies from 0 to 4)

        assert(dataset_name in [DATASET_NAME_SUMME, DATASET_NAME_TVSUM])
        self.dataset_name = dataset_name
        self.dataset_file_path = Path(
            f"../data/{dataset_name}/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5"
        )
        self.hdf = h5py.File(self.dataset_file_path, "r")

        self.splits_file_path = Path(f"../data/splits/{dataset_name.lower()}_splits.json")
        with open(self.splits_file_path) as f:
            splits = json.loads(f.read())
            for i, split in enumerate(splits):
                if i == self.split_index:
                    self.split = split
                    self.video_names = split[f"{self.mode}_keys"]
                    break

    def __len__(self):
        self.len = len(self.video_names)
        return self.len

    def __getitem__(self, index):
        video_name = self.video_names[index]    # gets the current video name
        frame_features = torch.tensor(np.array(self.hdf[f"{video_name}/features"]))
        return frame_features, video_name


def get_loader(mode, split_index):
    if mode.lower() == "train":
        vd = VideoData(mode, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, split_index)


if __name__ == "__main__":
    pass
