""" Train LM via huggingface library.
Some helper functions copied from
https://github.com/huggingface/transformers/blob/master/examples/research_projects/mlm_wwm/ """

import os
import shutil
import math
import sys
from pprint import pprint

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    classification_report, plot_confusion_matrix
from transformers import (
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

from src.helpers import get_model_and_tokenizer, get_data_collator, get_trainer_args, \
    save_model_state, get_data
from src.utils import get_current_artifacts_dir
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


def _classifier_metrics(p):
    logits, labels = p
    preds = np.argmax(logits, axis=1)
    average = "macro"
    accuracy = accuracy_score(y_true=labels, y_pred=preds)
    recall = recall_score(y_true=labels, y_pred=preds, average=average)
    precision = precision_score(y_true=labels, y_pred=preds, average=average)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def _eval_model(cfg, trainer, training_args, logger):
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate on validation set ***")

        eval_output = trainer.evaluate()
        print(eval_output)
        if cfg.architecture == "mlm":
            perplexity = math.exp(eval_output["eval_loss"])
            results["perplexity"] = perplexity
        elif cfg.architecture == "seq":
            results["accuracy"] = eval_output["eval_accuracy"]
            results["precision"] = eval_output["eval_precision"]
            results["recall"] = eval_output["eval_recall"]
            results["f1"] = eval_output["eval_f1"]
        else:
            sys.exit()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            # Relevant for distributed training. Check if this is main process.
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
    return results


def train(cfg, logger):
    if cfg.model.name == "bertweet":
        normalization = True
    else:
        normalization = False
    if cfg.architecture == "mlm":
        model, tokenizer = get_model_and_tokenizer(cfg.model.pretrained_model, cfg.architecture,
                                                   cfg.mode.freeze_encoder, normalization=normalization)
        data_collator = get_data_collator(cfg.architecture, tokenizer, cfg.model.mlm_probability,
                                          cfg.model.uniform_masking)
        compute_metrics = None
    elif cfg.architecture == "seq":
        model, tokenizer = get_model_and_tokenizer(cfg.model.pretrained_model, cfg.architecture,
                                                   cfg.mode.freeze_encoder, normalization=normalization,
                                                   num_labels=2 if cfg.task == "informativeness" else 11)
        data_collator = None
        compute_metrics = _classifier_metrics
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
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    if cfg.mode.continue_training:
        checkpoint = _get_last_checkpoint(cfg, training_args, logger)
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # For development. Check which layers are not frozen:
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    if cfg.mode.save_model:
        trainer.save_model()

    save_model_state(logger, train_result, trainer, training_args)
    results = _eval_model(cfg, trainer, training_args, logger)

    # Move all stored artifacts to mlflow run
    #artifacts_dir = get_current_artifacts_dir(cfg)
    #for file_name in os.listdir(tmp_output_dir):
    #    shutil.move(os.path.join(tmp_output_dir, file_name), artifacts_dir)
    #    shutil.copy(os.path.join(".hydra", "config.yaml"),
    #                os.path.join(artifacts_dir, "config.yaml"))
    #os.rmdir(tmp_output_dir)

    return results


