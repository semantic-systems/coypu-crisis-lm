# crisis-lm
WIP: Development of a crisis-tweet LM 

## Data
When running `python main.py` the CrisisBench dataset is automatically downloaded to a data 
folder within this project (if you don't already have it there). From this, a [huggingface 
dataset](https://huggingface.co/docs/datasets/access.html) is created.  


## Fine-tuning approaches
This repository will experiment with different LM fine-tuning approaches.

* Replication of CrisisBERT via sequence classification learning with DistilBERT
* Unsupervised MLM objective
* [Adapter-based fine-tuning](https://neurips2021-nlp.github.io/papers/9/CameraReady/NeurIPS2021_UDA_with_adapter.pdf)
* 