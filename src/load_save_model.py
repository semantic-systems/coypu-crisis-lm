import os
import sys

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification,\
    AutoAdapterModel, AdapterConfig, TrainingArguments


def get_model_and_tokenizer(pretrained_model, architecture, freeze_encoder, pretrained_adapter, do_train=True,
                            normalization=False, num_labels=None, task_name=""):
    if normalization:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if architecture == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
    elif architecture == "seq":
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
        # Freeze encoder
        if freeze_encoder:
            for param in model.base_model.parameters():
                param.requires_grad = False
    elif "adap" in architecture:
        model = get_adapter(pretrained_model, architecture, task_name, pretrained_adapter, do_train, num_labels)
    else:
        sys.exit("Architecture not implemented. Please check your config.yaml and select either 'mlm' or 'seq'.")
    return model, tokenizer

def get_adapter(pretrained_model, architecture, task_name, pretrained_adapter, do_train, num_labels=None):
    # Train an Adapter via AdapterHub, https://adapterhub.ml/
    # https://docs.adapterhub.ml/training.html
    # - bottleneck size 48 (compression rate 16 - performance-wise sweet spot
    # - Pfeiffer variant (one bottleneck)
    # https://github.com/adapter-hub/adapter-transformers/blob/master/examples/pytorch/text-classification/run_glue.py
    if architecture == "adap_mlm":
        model = AutoAdapterModel.from_pretrained(pretrained_model)
        model.add_masked_lm_head(task_name) # add head
    elif architecture == "adap_seq":
        model = AutoAdapterModel.from_pretrained(pretrained_model)
        model.add_classification_head(task_name, num_labels=num_labels) # add head
    # task adapter - only add if not existing
    if task_name not in model.config.adapters:
        # resolve the adapter config
        adapter_config = AdapterConfig.load(
            "pfeiffer",
            non_linearity="relu",
            reduction_factor=16,
        )
        # add a new adapter
        model.add_adapter(
            task_name,
            config=adapter_config
        )
    # Freeze all model weights except of those of this adapter
    model.train_adapter(task_name)
    # Set the adapters to be used in every forward pass
    model.set_active_adapters(task_name)
    return model



def get_trainer_args(cfg, output_dir):
    if cfg.mode.name == "train":
        trainer_args = TrainingArguments(output_dir=output_dir,
                                         overwrite_output_dir=cfg.mode.overwrite_output_dir,
                                         do_train=cfg.mode.do_train,
                                         do_eval=cfg.mode.do_eval,
                                         per_device_train_batch_size=cfg.model.per_device_train_batch_size if not cfg.debugging_mode else 1,
                                         per_device_eval_batch_size=cfg.model.per_device_eval_batch_size if not cfg.debugging_mode else 1,
                                         optim="adamw_torch",
                                         learning_rate=cfg.model.learning_rate,
                                         weight_decay=cfg.model.weight_decay,
                                         adam_beta1=cfg.model.adam_beta1,
                                         adam_beta2=cfg.model.adam_beta2,
                                         warmup_steps=cfg.model.warmup_steps,
                                         num_train_epochs=cfg.model.num_train_epochs,
                                         evaluation_strategy=cfg.mode.evaluation_strategy,
                                         eval_steps=cfg.mode.eval_steps if
                                         cfg.mode.evaluation_strategy == "steps" else None,
                                         save_steps=cfg.mode.save_steps,
                                         logging_steps=cfg.mode.logging_steps,
                                         load_best_model_at_end=cfg.mode.load_best_model_at_end,
                                         seed=cfg.model_seed,
                                         fp16=cfg.gpu.fp16,
                                         fp16_opt_level=cfg.gpu.fp16_opt_level if cfg.gpu.fp16 else None,
                                         half_precision_backend=cfg.gpu.half_precision_backend if cfg.gpu.fp16 else None,
                                         )
    elif cfg.mode.name == "test":
        trainer_args = TrainingArguments(output_dir=output_dir,
                                         do_eval=cfg.mode.do_eval,
                                         per_device_eval_batch_size=cfg.model.per_device_eval_batch_size if not cfg.debugging_mode else 1,
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