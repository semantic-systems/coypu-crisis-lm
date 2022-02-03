import os
import sys
import math
from pprint import pprint

import hydra
from transformers import (
    DataCollatorForWholeWordMask,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset

from src.model_helpers import get_model_and_tokenizer
from src.utils import download_data_from_url, unzip_tar_file
from src.custom_mlflow_callback import CustomMLflowCallback


def get_data_collator(architecture, tokenizer, mlm_probability):
    if architecture == "mlm":
        data_collator = DataCollatorForWholeWordMask(tokenizer, mlm_probability)
    elif architecture == "seq":
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        sys.exit(
            "Architecture not implemented. Please check your config.yaml and select either 'mlm' or 'seq'.")
    return data_collator


def get_trainer_args(cfg):
    output_dir = cfg.model_dir
    os.makedirs(output_dir, exist_ok=True)

    if cfg.mode.name == "train":
        training_args = TrainingArguments(output_dir=output_dir,
                                          overwrite_output_dir=cfg.mode.continue_training,
                                          do_train=cfg.mode.do_train,
                                          do_eval=cfg.mode.do_eval,
                                          per_device_train_batch_size=cfg.mode.per_device_train_batch_size if not cfg.debugging_mode else 1,
                                          per_device_eval_batch_size=cfg.mode.per_device_eval_batch_size if not cfg.debugging_mode else 1,
                                          optim="adamw_torch",
                                          learning_rate=cfg.mode.learning_rate,
                                          weight_decay=cfg.mode.weight_decay,
                                          num_train_epochs=cfg.mode.num_train_epochs,
                                          evaluation_strategy="steps",
                                          load_best_model_at_end=True,
                                          seed=cfg.seed,
                                          fp16=cfg.gpu.fp16,
                                          fp16_opt_level=cfg.gpu.fp16_opt_level,
                                          half_precision_backend=cfg.gpu.half_precision_backend,
                                          )
    else:
        sys.exit("Run mode not implemented. So far only supporting training.")

    return training_args


def get_last_checkpoint(cfg, training_args, logger):
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


def eval_model(trainer, training_args, logger):
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate on validation set ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            # Relevant for distributed training. Check if this is main process.
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
    return results


def save_model_state(logger, train_result, trainer, training_args):
    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
    if trainer.is_world_process_zero():
        # Relevant for distributed training. Check if this is main process.
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


def train(cfg, logger):
    num_labels = 2 if cfg.task == "informativeness" else 11
    padding = "max_length" if cfg.mode.pad_to_max_length else False

    model, tokenizer = get_model_and_tokenizer(cfg.model.pretrained_model, cfg.architecture,
                                               num_labels)
    training_args = get_trainer_args(cfg)
    data_collator = get_data_collator(cfg.architecture, tokenizer, cfg.mode.mlm_probability)

    # Get and prepare data
    data_dir = hydra.utils.to_absolute_path(os.path.join(cfg.data_path, cfg.data_subfolder))
    if not os.path.isdir(data_dir):
        unzip_tar_file(download_data_from_url(cfg))

    dataset = load_dataset(hydra.utils.to_absolute_path("src/custom_datasets.py"),
                           name=cfg.task if not cfg.debugging_mode else "debugging")
    print("Loaded dataset with", dataset)

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if
                            len(line) > 0 and not line.isspace()]
        return tokenizer(examples["text"], padding=padding, truncation=True, max_length=None)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        load_from_cache_file=not cfg.overwrite_data_cache,
    )

    print("Tokenized dataset:")
    pprint(tokenized_dataset)

    # Define callbacks
    mlflow_cb = CustomMLflowCallback()
    callbacks = [mlflow_cb]

    if cfg.mode.early_stopping:
        early_stopping_cb = EarlyStoppingCallback(
            early_stopping_patience=cfg.mode.patience
        )
        callbacks += [early_stopping_cb]

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"] if training_args.do_train else None,
        eval_dataset=tokenized_dataset["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    checkpoint = get_last_checkpoint(cfg, training_args, logger)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    if cfg.mode.save_model:
        trainer.save_model()

    save_model_state(logger, train_result, trainer, training_args)
    results = eval_model(trainer, training_args, logger)

    return results



