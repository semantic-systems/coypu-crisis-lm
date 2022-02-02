import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM


def get_model_and_tokenizer(pre_trained_model, architecture, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    if architecture == "mlm":
        model = AutoModelForMaskedLM.from_petrained(pre_trained_model)
    elif architecture == "seq":
        model = AutoModelForSequenceClassification(pre_trained_model, num_labels=num_labels)
    else:
        sys.exit("Architecture not implemented. Please check your config.yaml and select either 'mlm' or 'seq'.")
    return tokenizer, model

