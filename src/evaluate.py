import math
import os
import sys
from pprint import pprint

from transformers import AutoTokenizer, AutoModel, AutoAdapterModel, TrainingArguments, AdapterTrainer

from evaluate import evaluator, combine

from src.load_data import get_data_collator, get_data
from src.metrics import classifier_metrics

def get_model_and_tokenizer_eval(pretrained_model, pretrained_adapter, normalization=False):
    if normalization:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, normalization=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    if len(pretrained_adapter) > 0:
        model = AutoAdapterModel.from_pretrained(pretrained_model)
        adapter_name = model.load_adapter(pretrained_adapter)
        model.set_active_adapters(adapter_name)
    else:
        model = AutoModel.from_pretrained(pretrained_model)
    return model, tokenizer

def eval_model(cfg, trainer, training_args, logger):
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
            results["f1_weighted"] = eval_output["eval_f1_weighted"]
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

def evaluate(cfg, logger):
    if cfg.model.name == "bertweet":
        normalization = True
    else:
        normalization = False
    model, tokenizer = get_model_and_tokenizer_eval(cfg.model.pretrained_model, cfg.model.pretrained_adapter,
                                                    normalization)

    if cfg.architecture in ["mlm", "adap_mlm"]:
        compute_metrics = None
    elif cfg.architecture in ["seq", "adap_seq"]:
        compute_metrics = classifier_metrics
    else:
        sys.exit("Architecture style not implemented.")

    dataset = get_data(cfg)
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

    training_args = TrainingArguments(
        per_device_eval_batch_size=cfg.model.per_device_eval_batch_size,
        logging_steps=200,
        output_dir="eval_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.evaluate()

    # eval_results = task_evaluator.compute(
    #    model_or_pipeline=model,
    #    tokenizer=tokenizer,
    #    data=tokenized_dataset["test"],
    #    metric=combine(["accuracy", "recall", "precision", "f1"])
    #    #label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
    #)
    #print(eval_results)


