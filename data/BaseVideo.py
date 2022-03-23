import abc
from pathlib import Path
import cv2


class BaseVideo(abc.ABC):
    def __init__(self, file_name, video_dir, **kwargs):
        self.video_dir = video_dir
        self.filename = str(video_dir / file_name)
        self.reader = cv2.VideoCapture(self.filename)

    @property
    def name(self):
        return Path(self.filename).stem

    @property
    def nframes(self):
        return int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return int(self.reader.get(cv2.CAP_PROP_FPS))

    # May cause out of memory error
    # def frames(self):
    #     return list(self.iter_frames())

    def iter_frames(self):
        while True:
            success, frame = self.reader.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
            else:
                break

    @abc.abstractmethod
    def ground_truth(self):
        # shape: (num_users, n_frames)
        pass
