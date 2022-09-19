import os
import sys

import hydra
from datasets import load_dataset

from transformers import DataCollatorForTokenClassification

from src.custom_data_collator_mlm import CustomDataCollatorForLanguageModeling
from src.utils import unzip_tar_file, download_data_from_url, get_project_root


def get_data_collator(architecture, tokenizer, mlm_probability=None, uniform_masking=False):
    if architecture in ["mlm", "adap_mlm"]:
        data_collator = CustomDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                                                              mlm_probability=mlm_probability,
                                                              uniform_masking=uniform_masking)
    elif architecture in ["seq", "adap_seq"]:
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    else:
        sys.exit(
            "Architecture not implemented. Please check your config.yaml.")
    return data_collator


def get_data(cfg):
    # Get and prepare data
    data_dir = hydra.utils.to_absolute_path(os.path.join(cfg.data_path, cfg.data_subfolder))
    if not os.path.isdir(data_dir):
        unzip_tar_file(download_data_from_url(cfg))
    if cfg.debugging_mode:
        if cfg.architecture in ["mlm", "adap_mlm"]:
            config_name = "debugging_mlm"
        else:
            config_name = "debugging_seq"
    else:
        config_name = cfg.task
    dataset = load_dataset(os.path.join(get_project_root(), "src/custom_datasets.py"),
                           name=config_name)
    print("Loaded dataset with", dataset)
    return dataset