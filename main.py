import logging
import sys
import os
import warnings

import mlflow
import hydra
from omegaconf import DictConfig

from src.train import train
from src.mlm_inference import run_mlm_inference
from src.utils import get_project_root


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    # Define where to store mlflow runs (centralized)
    # Otherwise they would be stored separately for each hydra run
    mlruns_folder = os.path.join(get_project_root(), cfg.mlruns_dir)
    mlflow.set_tracking_uri(f"file:{mlruns_folder}")
    mlflow.set_experiment(f"model={cfg.model.name}_task={cfg.task}_modelseed={cfg.model_seed}")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    #warnings.filterwarnings("ignore")

    if cfg.mode.name == "train":
        print("Launching train mode:")
        train(cfg, logger)
    elif cfg.mode.name == "mlm_inference":
        print("Launching inference mode:")
        run_mlm_inference(cfg, logger)


if __name__ == "__main__":
    run()
