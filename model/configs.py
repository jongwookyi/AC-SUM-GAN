import os
from pathlib import Path
import argparse
import pprint
import math
import torch

_package_path = Path(__file__).parent.absolute()

DEVICE_CUDA = "cuda"
DEVICE_CPU = "cpu"

_default_dataset = "TVSum"
_dataset_dir = (_package_path / "../data").resolve()
_default_dataset_path = _dataset_dir / _default_dataset / f"eccv16_dataset_{_default_dataset.lower()}_google_pool5.h5"
_default_splits_path = _dataset_dir / "splits" / f"{_default_dataset.lower()}_splits.json"
_default_save_dir = (_package_path / "../exp1").resolve()


def str2bool(v):
    """string to boolean"""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.device.startswith(DEVICE_CUDA) and (not torch.cuda.is_available()):
            print("CUDA is not available, uses CPU!")
            self.device = DEVICE_CPU
        self.device = torch.device(self.device)
        self.termination_point = math.floor(0.15 * self.action_state_size)
        self.save_dir = Path(self.save_dir)
        self.set_dataset_dir(self.dataset)

    def set_dataset_dir(self, dataset):
        sigma = f"sigma{self.regularization_factor}"
        split = f"split{self.split_index}"

        save_dir = self.save_dir / dataset / sigma

        self.log_dir = save_dir / "logs" / split
        self.score_dir = save_dir / "results" / split
        self.model_dir = save_dir / "models" / split

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.score_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = "Configurations\n"
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Mode
    parser.add_argument("--mode", type=str, default="train", help="running mode")
    parser.add_argument("--device", type=str, default=DEVICE_CUDA, help="torch device")
    parser.add_argument("--verbose", type=str2bool, default="true", help="show verbose logs")
    parser.add_argument("--dataset", type=str, default=_default_dataset, help="name of dataset")
    parser.add_argument("--dataset_path", type=str, default=str(_default_dataset_path), help="dataset file path")
    parser.add_argument("--splits_path", type=str, default=str(_default_splits_path), help="splits file path")

    # Model
    parser.add_argument("--input_size", type=int, default=1024, help="input size")
    parser.add_argument("--hidden_size", type=int, default=512, help="hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--regularization_factor", type=float, default=0.5, help="sigma")
    parser.add_argument("--entropy_coef", type=float, default=0.1, help="entropy coefficient")

    # Train
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size")
    parser.add_argument("--clip", type=float, default=5.0, help="clip")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--discriminator_lr", type=float, default=1e-5, help="discriminator learning rate")
    parser.add_argument("--split_index", type=int, default=0, help="split index")
    parser.add_argument("--action_state_size", type=int, default=60, help="action state size")
    parser.add_argument("--save_dir", type=str, default=str(_default_save_dir), help="save dir")

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == "__main__":
    config = get_config()
    import ipdb
    ipdb.set_trace()
