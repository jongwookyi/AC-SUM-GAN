from . import BaseVideo

from pathlib import Path
import pandas as pd
import numpy as np

# directory structure:
# /SumMe
#   /data
#   /matlab
#   /thumbnail
#   /video


class TVSumVideo(BaseVideo):
    GT_FILE_NAME = "ydata-tvsum50-anno.tsv"
    GT_SCORE_COLUMN = 1

    def __init__(self, name, dataset_dir="./TVSum"):
        file_name = f"{name}.mp4"
        video_dir = Path(dataset_dir, "video")
        super(TVSumVideo, self).__init__(file_name, video_dir)

    def ground_truth(self):
        file_path = self.video_dir / f"../data" / self.GT_FILE_NAME
        data = pd.read_csv(file_path, sep="\t", header=None, index_col=0)

        user_score = data.loc[self.name].values[:, self.GT_SCORE_COLUMN]
        user_score = np.asarray([np.fromstring(csv, dtype=np.uint8, sep=",") for csv in user_score])
        return user_score
