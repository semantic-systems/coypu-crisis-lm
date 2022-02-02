import os
import sys

import hydra
from transformers import DataCollatorForWholeWordMask, DataCollatorForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

from src.model_helpers import get_model_and_tokenizer
from src.utils import download_data_from_url, unzip_tar_file


def get_data_collator(architecture, tokenizer, mlm_probability):
    if architecture == "mlm":
        data_collator = DataCollatorForWholeWordMask(tokenizer, mlm_probability)
    elif architecture == "seq":
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        sys.exit("Architecture not implemented. Please check your config.yaml and select either 'mlm' or 'seq'.")
    return data_collator


def get_trainer_args(cfg):
    output_dir = hydra.utils.to_absolute_path(cfg.model_dir)
    os.makedirs(output_dir, exist_ok=True)

    if cfg.mode.name == "train":
        training_args = TrainingArguments(output_dir=output_dir,
                                          overwrite_output_dir=cfg.mode.continue_training,
                                          do_train=cfg.mode.do_train,
                                          do_eval=cfg.mode.do_eval,
                                          per_device_train_batch_size=cfg.mode.per_device_train_batch_size,
                                          per_device_eval_batch_size=cfg.mode.per_device_eval_batch_size,
                                          learning_rate=cfg.mode.learning_rate,
                                          weight_decay=cfg.mode.weight_decay,
                                          num_train_epochs=cfg.mode.num_train_epochs,
                                          #load_best_model_at_end=True,
                                          )
    else:
        sys.exit("Run mode not implemented. So far only supporting training.")

    return training_args


def train(cfg):
    num_labels = 2 if cfg.task == "informativeness" else 11
    padding = "max_length" if cfg.mode.pad_to_max_length else False

    model, tokenizer = get_model_and_tokenizer(cfg.model.pretrained_model, cfg.architecture, num_labels)
    training_args = get_trainer_args(cfg)
    data_collator = get_data_collator(cfg.architecture, tokenizer, cfg.mode.mlm_probability)

    data_dir = hydra.utils.to_absolute_path(os.path.join(cfg.data_path, cfg.data_subfolder))
    if not os.path.isdir(data_dir):
        unzip_tar_file(download_data_from_url(cfg))
    dataset = load_dataset(hydra.utils.to_absolute_path("src/custom_datasets.py"),
                           name=cfg.task)

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(examples["text"], padding=padding, truncation=True, max_length=None)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        load_from_cache_file=not cfg.overwrite_data_cache,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"] if training_args.do_train else None,
        eval_dataset=tokenized_dataset["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


