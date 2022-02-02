import logging
import sys
import os
import warnings

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    #logger = logging.getLogger(__name__)
    #logger.setLevel(logging.INFO)
    #warnings.filterwarnings("ignore")
    pass

if __name__ == "__main__":
    run()
