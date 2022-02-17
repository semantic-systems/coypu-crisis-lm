import logging
import sys
import os
import warnings

import mlflow
import hydra
from omegaconf import DictConfig

from src.train_mlm import train
from src.utils import get_project_root


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    # Define where to store mlflow runs (centralized)
    # Otherwise they would be stored separately for each hydra run
    mlruns_folder = os.path.join(get_project_root(), cfg.mlruns_dir)
    mlflow.set_tracking_uri(f"file:{mlruns_folder}")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    #warnings.filterwarnings("ignore")

    if cfg.mode.name == "train":
        train(cfg, logger)
    elif cfg.mode.name == "eval":
        pass
    elif cfg.mode.name == "predict":
        pass


if __name__ == "__main__":
    run()
