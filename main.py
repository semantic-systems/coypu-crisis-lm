import logging
import sys
import os
import warnings

import hydra
from omegaconf import DictConfig

from src.train import train


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    #logger = logging.getLogger(__name__)
    #logger.setLevel(logging.INFO)
    #warnings.filterwarnings("ignore")
    if cfg.mode.name == "train":
        train(cfg)
    elif cfg.mode.name == "eval":
        pass
    elif cfg.mode.name == "predict":
        pass


if __name__ == "__main__":
    run()
