import os
import sys

import hydra
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM, \
    DataCollatorForWholeWordMask, DataCollatorForTokenClassification, TrainingArguments
from datasets import load_dataset

from src.utils import unzip_tar_file, download_data_from_url, get_project_root


def get_model_and_tokenizer(pre_trained_model, architecture, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    if architecture == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(pre_trained_model)
    elif architecture == "seq":
        model = AutoModelForSequenceClassification.from_pretrained(pre_trained_model, num_labels=num_labels)
    else:
        sys.exit("Architecture not implemented. Please check your config.yaml and select either 'mlm' or 'seq'.")
    return model, tokenizer


def get_data_collator(architecture, tokenizer, mlm_probability):
    if architecture == "mlm":
        data_collator = DataCollatorForWholeWordMask(tokenizer, mlm_probability)
    elif architecture == "seq":
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        sys.exit(
            "Architecture not implemented. Please check your config.yaml and select either 'mlm' or 'seq'.")
    return data_collator


def get_trainer_args(cfg, output_dir):
    if cfg.mode.name == "train":
        trainer_args = TrainingArguments(output_dir=output_dir,
                                         overwrite_output_dir=cfg.mode.continue_training,
                                         do_train=cfg.mode.do_train,
                                         do_eval=cfg.mode.do_eval,
                                         per_device_train_batch_size=cfg.mode.per_device_train_batch_size if not cfg.debugging_mode else 1,
                                         per_device_eval_batch_size=cfg.mode.per_device_eval_batch_size if not cfg.debugging_mode else 1,
                                         optim="adamw_torch",
                                         learning_rate=cfg.mode.learning_rate,
                                         weight_decay=cfg.mode.weight_decay,
                                         num_train_epochs=cfg.mode.num_train_epochs,
                                         evaluation_strategy=cfg.mode.evaluation_strategy,
                                         eval_steps=cfg.mode.eval_steps if
                                         cfg.mode.evaluation_strategy == "steps" else None,
                                         logging_steps=cfg.mode.logging_steps,
                                         load_best_model_at_end=(cfg.mode.evaluation_strategy ==
                                                                 "steps"),
                                         seed=cfg.seed,
                                         fp16=cfg.gpu.fp16,
                                         fp16_opt_level=cfg.gpu.fp16_opt_level,
                                         half_precision_backend=cfg.gpu.half_precision_backend,
                                         )
    elif cfg.mode.name == "test":
        trainer_args = TrainingArguments(output_dir=output_dir,
                                         do_eval=cfg.mode.do_eval,
                                         per_device_eval_batch_size=cfg.mode.per_device_eval_batch_size if not cfg.debugging_mode else 1,
                                         )
    else:
        sys.exit("Can't load trainer args bc. 'run mode' is neither set to 'train' nor 'test'")

    return trainer_args


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


def get_data(cfg):
    # Get and prepare data
    data_dir = hydra.utils.to_absolute_path(os.path.join(cfg.data_path, cfg.data_subfolder))
    if not os.path.isdir(data_dir):
        unzip_tar_file(download_data_from_url(cfg))
    dataset = load_dataset(os.path.join(get_project_root(), "src/custom_datasets.py"),
                           name=cfg.task if not cfg.debugging_mode else "debugging")
    print("Loaded dataset with", dataset)
    return dataset