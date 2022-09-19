""" Train LM via huggingface library.
Some helper functions copied from
https://github.com/huggingface/transformers/blob/master/examples/research_projects/mlm_wwm/ """

import os
import sys
from pprint import pprint

from transformers import (
    Trainer,
    AdapterTrainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from src.evaluate import eval_model
from src.load_data import get_data_collator, get_data
from src.load_save_model import get_model_and_tokenizer, get_trainer_args, save_model_state
from src.metrics import classifier_metrics
from src.custom_mlflow_callback import CustomMLflowCallback

artifacts_output_dir = "artifacts"


def _get_last_checkpoint(cfg, training_args, logger):
    """Detect and return last checkpoint or None."""
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif cfg.model.pretrained_model is not None and os.path.isdir(cfg.model.pretrained_model):
        checkpoint = cfg.model.pretrained_model
    else:
        checkpoint = None
    return checkpoint


def train(cfg, logger):
    if cfg.model.name == "bertweet":
        normalization = True
    else:
        normalization = False
    if cfg.architecture in ["mlm", "adap_mlm"]:
        model, tokenizer = get_model_and_tokenizer(cfg.model.pretrained_model, cfg.architecture,
                                                   cfg.mode.freeze_encoder, cfg.model.pretrained_adapter,
                                                   do_train=cfg.mode.do_train, normalization=normalization, task_name=cfg.task)
        data_collator = get_data_collator(cfg.architecture, tokenizer, cfg.model.mlm_probability,
                                          cfg.model.uniform_masking)
        compute_metrics = None
    elif cfg.architecture in ["seq", "adap_seq"]:
        model, tokenizer = get_model_and_tokenizer(cfg.model.pretrained_model, cfg.architecture,
                                                   cfg.mode.freeze_encoder, cfg.model.pretrained_adapter,
                                                   do_train=cfg.mode.do_train, normalization=normalization,
                                                   num_labels=2 if cfg.task == "informativeness" else 11,
                                                   task_name=cfg.task)
        data_collator = None
        compute_metrics = classifier_metrics
    else:
        sys.exit("Architecture style not implemented.")

    training_args = get_trainer_args(cfg, artifacts_output_dir)
    dataset = get_data(cfg)
    dataset = dataset.shuffle(cfg.data_seed)
    print(dataset)

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if
                            len(line) > 0 and not line.isspace()]
        return tokenizer(examples["text"],
                         padding="max_length" if cfg.model.pad_to_max_length else False,
                         truncation=True, max_length=(cfg.model.max_seq_length if
                                                      isinstance(cfg.model.max_seq_length,
                                                                 int) else None))

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        load_from_cache_file=not cfg.overwrite_data_cache,
    )

    print("Tokenized dataset:")
    pprint(tokenized_dataset)

    # Define callbacks
    mlflow_cb = CustomMLflowCallback()
    callbacks = [] #[mlflow_cb]

    if cfg.mode.early_stopping:
        early_stopping_cb = EarlyStoppingCallback(
            early_stopping_patience=cfg.mode.patience
        )
        callbacks += [early_stopping_cb]

    # Setup trainer
    if "adap" in cfg.architecture:
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"] if training_args.do_train else None,
            eval_dataset=tokenized_dataset["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"] if training_args.do_train else None,
            eval_dataset=tokenized_dataset["validation"] if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
    if cfg.mode.continue_training:
        checkpoint = _get_last_checkpoint(cfg, training_args, logger)
    else:
        checkpoint = None
    trainer.train(resume_from_checkpoint=checkpoint)
    train_result = trainer.evaluate(eval_dataset=tokenized_dataset["validation"] if training_args.do_eval else None)
    print(train_result)

    # For development. Check which layers are not frozen:
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    if cfg.mode.save_model:
        trainer.save_model()

    save_model_state(logger, train_result, trainer, training_args)
    results = eval_model(cfg, trainer, training_args, logger)

    # Move all stored artifacts to mlflow run
    #artifacts_dir = get_current_artifacts_dir(cfg)
    #for file_name in os.listdir(tmp_output_dir):
    #    shutil.move(os.path.join(tmp_output_dir, file_name), artifacts_dir)
    #    shutil.copy(os.path.join(".hydra", "config.yaml"),
    #                os.path.join(artifacts_dir, "config.yaml"))
    #os.rmdir(tmp_output_dir)

    return results


