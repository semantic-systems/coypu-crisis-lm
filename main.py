import logging
import sys
import os
import warnings

import hydra
from omegaconf import DictConfig

from datasets import load_dataset
from src.utils import download_data_from_url, unzip_tar_file


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    #logger = logging.getLogger(__name__)
    #logger.setLevel(logging.INFO)
    #warnings.filterwarnings("ignore")

    # download crisisbench
    data_dir = hydra.utils.to_absolute_path(os.path.join(cfg.data_path, cfg.data_subfolder))
    if not os.path.isdir(data_dir):
        zipfile_location = download_data_from_url(cfg)
        unzip_tar_file(zipfile_location)
    dataset = load_dataset(hydra.utils.to_absolute_path("src/custom_datasets.py"),
                           name="informativeness")
    print(dataset)
    
if __name__ == "__main__":
    run()
