import sys
from pprint import pprint

from transformers import pipeline


def run_mlm_inference(cfg, logger):
    apply_fill_mask(cfg.model.pretrained_model, cfg.mode.inference_mode, cfg.mode.file_path)


def apply_fill_mask(model, inference_mode, file_path=""):
    mask_filler_pipeline = pipeline("fill-mask", model=model)
    if inference_mode == "file":
        _apply_to_file(mask_filler_pipeline, file_path)
    elif inference_mode == "interactive":
        _apply_to_user_input(mask_filler_pipeline)
    else:
        sys.exit("Aborting - Please specify inference_mode 'file' or 'interactive'.")


def _apply_to_file(mask_filler_pipeline, path):
    pass


def _apply_to_user_input(mask_filler_pipeline):
    print("--- MLM demo ---")
    print("In order to try out the mask filler, type in any sentence and replace the word to be guessed with <mask>. Then press ENTER.")
    user_input = input("Input: ")
    model_output = mask_filler_pipeline(user_input)
    print("Output:")
    pprint(model_output)
