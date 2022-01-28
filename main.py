import logging
import sys
import warnings

import hydra
from omegaconf import DictConfig

from src.custom_datasets import CrisisBenchDataset


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    #logger = logging.getLogger(__name__)
    #logger.setLevel(logging.INFO)
    #warnings.filterwarnings("ignore")

    if cfg.task == "informativeness":
        class_labels = ['informative', 'not_informative']
    elif cfg.task == "humanitarian":
        class_labels = ['affected_individual', 'caution_and_advice', 'displaced_and_evacuations', 'donation_and_volunteering', 'infrastructure_and_utilities_damage', 'injured_or_dead_people', 'missing_and_found_people', 'not_humanitarian', 'requests_or_needs', 'response_efforts', 'sympathy_and_support']
    else:
        sys.exit("Please specify a known task in your config.yaml: Either 'humanitarian' or "
                 "'informativeness'")

    dataset = CrisisBenchDataset()

    print(dataset._info())

if __name__ == "__main__":
    run()
