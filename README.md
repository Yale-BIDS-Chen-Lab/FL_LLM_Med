# distributed_LLM_learning_simulator

This is a simulator of Federated Learning for LLM fine-tuning on a single host. It implements our works.

## Install Environment

This is a Python project. The third party dependencies are listed in [pyproject.toml](https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/pyproject.toml).

Use PIP to set up the environment:

```
python3 -m pip install . --upgrade --force-reinstall --user
```

## Original and our case study datasets

We take some of the open and private datasets as training and testing benchmark:
- **[MIMIC-III](https://physionet.org/content/mimiciii/1.4/)**
- **[MTSamples](https://mtsamples.com/)**
- **UTP**
  
The independent validation set is below:
- **[I2B2](https://www.i2b2.org/NLP/DataSets/)**

The case study on new annotation is used:
- **YNHHS**(Yale New Haven Health System )

## Model Finetune

### Algorithm

Based on Fed-MedLoRA, we further propose **Fed-MedLoRA+**, which dynamically estimates each site's contribution and performs adaptive, data-aware aggregation to mitigate the effects of cross-site data heterogeneity. 

### Setting

**Zero-shot** and **Single site** are used as baseline. **Centralized learning** is upper bound. For our experiments, we calculated the results of **Fed-MedLoRA** and **Fed-MedLoRA+**.

### Models

**[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)**, 
**[LLaMA3-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)**, and 
**[DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)** 
are open models, used in our experiments.

### Training

For model training, we use [train_bert.sh](https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/train_bert.sh) to train **Bio_ClinicalBERT** model, 
and use [train_mix.sh](https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/train_mix.sh) to train other models (**LLaMA3-8B** and **DeepSeek-R1-Distill-Llama-8B**). 
Both of these corresponding configuration files are located in a subfolder of [conf](https://github.com/cyyever/distributed_LLM_learning_simulator/tree/main/conf).

### Evaluations

In this study, we calculate the **strict and lenient F1 scores** of different settings on **[NER](https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/evaluate.sh)**(named entity recognition) and 
**[RE](https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/re_evaluate.sh)**(relation extraction) tasks. 
Inference and evaluation are calculated at the same time.



