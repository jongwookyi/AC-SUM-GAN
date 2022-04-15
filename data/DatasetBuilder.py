"""
    DatasetBuilder

    1. Converting video to frames
    2. Extracting features
    3. Getting change points
"""
import torch
import torch.nn as nn
import torchvision as tv

import os
from pathlib import Path

from . import vsum_tool

from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import h5py


class DatasetBuilder:
    def __init__(self, video_path, save_path, save_frames=False,
                 deep_feature_model="googlenet", use_gpu=True):
        save_path = Path(save_path)
        self.save_frames = save_frames
        self.frames_dir = save_path.parent / "frames"
        self.h5_file = h5py.File(save_path, "w")

        self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self._set_deep_feature_model(deep_feature_model)

        self._set_video_list(video_path)

    def _set_deep_feature_model(self, model_name):
        # alexnet, resnet50, resnet152, googlenet
        model_name = model_name.lower()
        if not hasattr(tv.models, model_name):
            print(f"Unsupported model {model_name}!")
            model_name = "googlenet"

        print(f"deep feature model: {model_name}")
        model = getattr(tv.models, model_name)(pretrained=True)
        # print(model)

        pool_index = -3 if model_name == "googlenet" else -2
        layers = list(model.children())[:pool_index + 1]
        # print(layers)

        self.model = nn.Sequential(*layers).float().eval().to(self._device)
        self.preprocess = tv.transforms.Compose([
            tv.transforms.Resize([224, 224]),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _set_video_list(self, video_path):
        video_path = Path(video_path)
        self.video_path = video_path
        if video_path.is_dir():
            self.video_list = list(video_path.iterdir())
        else:
            self.video_list = [video_path]

        for idx, file_name in enumerate(self.video_list):
            key = f"video_{idx + 1}"
            self.h5_file.create_group(key)

    def _extract_feature(self, frame):
        with torch.no_grad():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.preprocess(frame)

            # add a dimension for batch
            batch = frame.unsqueeze(0).to(self._device)

            features = self.model(batch)
            features = features.squeeze()

            features = features.cpu().detach().numpy()
            return features

    def build(self):
        for video_idx, video_path in enumerate(tqdm(self.video_list)):
            if video_path.suffix.lower() == ".h5":
                continue

            if self.save_frames:
                frames_dir = self.frames_dir / video_path.stem
                os.makedirs(frames_dir, exist_ok=True)

            video_capture = cv2.VideoCapture(str(video_path))

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
 
            decimation_factor = 15
            picks = []
            video_feat = []
            video_feat_for_train = []
            for frame_idx in tqdm(range(n_frames)):
                success, frame = video_capture.read()
                if not success:
                    break

                frame_feat = self._extract_feature(frame)

                video_feat.append(frame_feat)
                if (frame_idx % decimation_factor) == 0:
                    picks.append(frame_idx)
                    video_feat_for_train.append(frame_feat)

                if self.save_frames:
                    cv2.imwrite(frames_dir / f"Frame{frame_idx}.jpeg", frame)
            picks = np.asarray(picks)
            video_feat = np.asarray(video_feat)
            video_feat_for_train = np.asarray(video_feat_for_train)

            video_capture.release()

            change_points, n_frame_per_seg = vsum_tool.get_change_points(video_feat, n_frames, fps)

            data = self.h5_file[f"video_{video_idx + 1}"]
            data["change_points"] = change_points
            # data["features"] = video_feat
            data["features"] = video_feat_for_train
            data["fps"] = fps
            data["n_frame_per_seg"] = n_frame_per_seg
            data["n_frames"] = n_frames
            data["picks"] = picks
            data["video_name"] = video_path.stem

    def close(self):
        self.h5_file.close()
