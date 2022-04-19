import sys
from pathlib import Path

_package_path = Path(__file__).parent.absolute()
_package_search_path = _package_path.parent
sys.path.append(str(_package_search_path))

from model.configs import get_config
from model.solver import Solver
from model.data_loader import get_loader


if __name__ == "__main__":
    config = get_config(mode="train")
    test_config = get_config(mode="test")

    print("train config:", config)
    print("test config:", test_config)

    train_loader = get_loader(config.mode, config.split_index)
    test_loader = get_loader(test_config.mode, test_config.split_index)
    solver = Solver(config, train_loader, test_loader)

    # evaluates the summaries generated using the initial random weights of the network
    solver.evaluate(-1)

    solver.train()
