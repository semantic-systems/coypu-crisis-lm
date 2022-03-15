import os
import sys
import random
from pprint import pprint

from transformers import pipeline
import hydra

from src.TweetNormalizer import normalizeTweet


def run_mlm_inference(cfg, logger):
    mask_filler_pipeline = pipeline("fill-mask", model=cfg.model.pretrained_model)
    output_path = hydra.utils.to_absolute_path(cfg.mode.output_path)
    os.makedirs(output_path, exist_ok=True)
    if cfg.mode.inference_mode == "file":
        _apply_to_file(mask_filler_pipeline, cfg.mode.input_path, output_path)
    elif cfg.mode.inference_mode == "interactive":
        _apply_to_user_input(mask_filler_pipeline, output_path)
    else:
        sys.exit("Aborting - Please specify inference_mode 'file' or 'interactive'.")


def _apply_to_file(mask_filler_pipeline, filepath, output_path):
    filepath = hydra.utils.to_absolute_path(filepath)
    if filepath.endswith('.tsv'):
        delim = '\t'
        ending = '.tsv'
    elif filepath.endswith('.csv'):
        delim = ','
        ending = '.csv'
    else:
        sys.exit("Unknown file format. Make sure to use a .tsv or .csv dataset file.")

    out_file = os.path.join(output_path, f"inferred_from_file{ending}")
    num_correct, num_total = 0, 0
    with open(filepath, "r", newline=None, encoding='utf-8', errors='replace') as f:
        print(f"Reading {os.path.basename(filepath)} as input to inference pipeline.")
        print("Computing predictions...")
        next(f)  # skip head col
        with open(out_file, "a") as o:
            for i, line in enumerate(f):
                line = line.strip()
                if line == "":
                    continue
                row = line.split(delim)
                text = row[3].strip()
                text = normalizeTweet(text)
                if len(text) > 280:  # Allowed tweet length
                    print(f"Skipping input {i} bc. it is longer than 280 characters.")
                    continue
                words = text.split()
                to_be_masked_word = words[random.randint(0, len(words)-1)]
                masked_input = text.replace(to_be_masked_word, '<mask>', 1)
                model_output = mask_filler_pipeline(masked_input)

                num_total += 1
                top_guess = model_output[0]['token_str'].replace(" ", "")
                if top_guess == to_be_masked_word:
                    num_correct += 1

                top_outputs = [f"({result['token_str']}, {round(result['score'], 3)})" for result in model_output]
                top_outputs_str = delim.join(top_outputs)
                o.write(f"{text}{delim}{masked_input}{delim}{top_outputs_str}\n")
            print("Results are stored in", out_file)
            print(f"Ratio of correct top guesses: {num_correct/num_total}")


def _apply_to_user_input(mask_filler_pipeline, output_path):
    out_file = os.path.join(output_path, f"inferred_from_ui.tsv")
    print("--- MLM demo ---")
    print("In order to try out the mask filler, type in any sentence and replace the word to be guessed with <mask>. "
          "Then press ENTER. Note: Your input must not exceed 280 characters.")
    print("Enter Q to quit.")
    while True:
        with open(out_file, "a") as o:
            user_input = input("Input: ")
            if user_input == "Q":
                print("Closing demo. Results were stored in", out_file)
                break
            user_input = normalizeTweet(user_input)
            model_output = mask_filler_pipeline(user_input)
            print("Output:")
            pprint(model_output)
            top_outputs = [f"({result['token_str']}, {round(result['score'], 3)})" for result in model_output]
            top_outputs_str = "\t".join(top_outputs)
            o.write(f"{user_input}\t{top_outputs_str}\n")

