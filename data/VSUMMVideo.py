from . import BaseVideo

from pathlib import Path
import os
import numpy as np

# directory structure:
# /VSUMM
#   /database
#   /UserSummary
#       /user1
#       ...
#       /user5


class VSUMMVideo(BaseVideo):
    NUM_USERS = 5
    SUMMARY_NAME_PREFIX = "Frame"

    def __init__(self, name, dataset_dir="./VSUMM"):
        file_name = f"{name}.mpg"
        video_dir = Path(dataset_dir, "database")
        super(VSUMMVideo, self).__init__(file_name, video_dir)

    def ground_truth(self):
        data_dir = self.video_dir / f"../UserSummary/{self.name}"

        user_score = np.zeros((self.NUM_USERS, self.nframes), dtype=np.uint8)
        for user_index in range(0, self.NUM_USERS):
            user_dir = data_dir / f"user{user_index + 1}"
            for summary in user_dir.iterdir():
                summary_name = summary.stem
                if not summary_name.startswith(self.SUMMARY_NAME_PREFIX):
                    print("Not a summary file:", summary)
                    continue
                frame_index = int(summary_name[len(self.SUMMARY_NAME_PREFIX):])
                user_score[user_index, frame_index] = 1

        return user_score
