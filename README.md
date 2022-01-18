# crisis-lm
WIP: Development of a crisis-tweet LM 

## Data
Load [CrisisBench corpus](https://crisisnlp.qcri.org/crisis_datasets_benchmarks) into data folder.

## Fine-tuning approaches
This repository will experiment with different LM fine-tuning approaches.

* Replication of CrisisBERT via sequence classification learning with DistilBERT
* Unsupervised MLM objective
* [Adapter-based fine-tuning](https://neurips2021-nlp.github.io/papers/9/CameraReady/NeurIPS2021_UDA_with_adapter.pdf)
* 