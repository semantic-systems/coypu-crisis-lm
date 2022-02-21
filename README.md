# crisis-lm
WIP: Development of a crisis-tweet LM 

## Data
When running `python main.py` the CrisisBench dataset is automatically downloaded to a data 
folder within this project (if you don't already have it there). From this, a [huggingface 
dataset](https://huggingface.co/docs/datasets/access.html) is created.  


## Fine-tuning approaches
This repository will experiment with different domain-/task-adapting fine-tuning approaches.

* Task-adaptive fine-tuning of [BERTweet](https://github.com/VinAIResearch/BERTweet) on unsupervised MLM objective
* [Adapter-based fine-tuning](https://neurips2021-nlp.github.io/papers/9/CameraReady/NeurIPS2021_UDA_with_adapter.pdf)
* Additional tuning towards SentenceTransformer embeddings

## Evaluation
Evaluation of the embeddings is done via 
* MLM perplexity
* Downstream task performance on the CrisisBench benchmark
* Additional CoyPu-specific downstream tasks: e.g., event detection
