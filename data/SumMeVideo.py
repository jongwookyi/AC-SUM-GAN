from . import BaseVideo

from pathlib import Path
import scipy.io
import numpy as np

# directory structure:
# /SumMe
#   /GT
#   /matlab
#   /python
#   /videos


class SumMeVideo(BaseVideo):
    def __init__(self, name, dataset_dir="./SumMe"):
        file_name = f"{name}.mp4"
        video_dir = Path(dataset_dir, "videos")
        super(SumMeVideo, self).__init__(file_name, video_dir)

    def ground_truth(self):
        file_path = self.video_dir / f"../GT/{self.name}.mat"
        data = scipy.io.loadmat(str(file_path))

        user_score = np.where(0 < data["user_score"], 1, 0).T
        return user_score.astype(np.uint8)
