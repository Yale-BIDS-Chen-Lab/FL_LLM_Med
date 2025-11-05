# FL_LLM_Med

This repository provides the related codes and models for federated instruction tuning for privacy-preserving clinical large language models (LLMs). 

## Install Environment

This is a Python project. The third party dependencies are listed in [pyproject.toml]([https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/pyproject.toml](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/pyproject.toml)).

Use PIP to set up the environment:

```
python3 -m pip install . --upgrade --force-reinstall --user
```

## Datasets

We use five sites cohorts, which consist of 42,198 entities and 41,570 relations covering four main entity categories and 16 relation types. 
- In-domain training/ testing benchmark:
  - [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) (Medical Information Mart for Intensive Care)
  - [MTSamples](https://mtsamples.com/) (Transcribed Medical Transcription Sample Reports and Examples)
  - [UTP](https://arxiv.org/abs/2411.10020) (UT Physicians notes)
  
- External patient cohort validation
  - [I2B2](https://www.i2b2.org/NLP/DataSets/) (Informatics for integrating biology & the bedside)

- Case study on newly annotated clinical notes
  - [YNHH](https://www.ynhhs.org/) (Yale New Haven Health system) 

## Model Finetune
### Clinical information extraction (IE) tasks
Named entity recognition (NER) and relation extraction (RE). 
### Models
- LLMs: [LLaMA3-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [LLaMA3-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [DeepSeek-R1-Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B), QW
- BERT models: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
### Baselines


### Algorithm

Based on Fed-MedLoRA, we further propose **Fed-MedLoRA+**, which dynamically estimates each site's contribution and performs adaptive, data-aware aggregation to mitigate the effects of cross-site data heterogeneity. 

### Setting

**Zero-shot** and **Single site** are used as baseline. **Centralized learning** is upper bound. For our experiments, we calculated the results of **Fed-MedLoRA** and **Fed-MedLoRA+**.

Zero-shot means we didn't fine-tune the models. Single site is without federated learning support. Centralized training pools data from all sites to train a single model. 

### Models

**[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)**, 
**[LLaMA3-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)**, and 
**[DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)** 
are open models, used in our experiments.

### Training

For model training, we use [train_bert.sh](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/train_bert.sh) to train **Bio_ClinicalBERT** model, 
and use [train_mix.sh](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/train_mix.sh) to train other models (**LLaMA3-8B** and **DeepSeek-R1-Distill-Llama-8B**). 
Both of these corresponding configuration files are located in a subfolder of [conf](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/tree/main/conf). 
Modify the contents of common.yaml to change the configuration parameters. 

#### Parameters in common.yaml

Take train_mix.sh as an example, introduce these parameters in [common.yaml](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/conf/medical_mix/common.yaml).

- **dataset_name**:

- **dataset_sampling**: file_split or random_split. "file_split" means that one dataset is only distributed to one client. "random_split" means the sentences in one dataset distributed to diffenent clients are random.

- **distributed_algorithm**: adaptor_avg or fed_avg. "fed_avg" is for Bio_ClinicalBERT model. "adaptor_avg" is for others.

- **train_files** and **test_files** in "dataset_kwargs": The test_files are the corresponding the train_files, such as RE_MIMIC3_**train**.json and RE_MIMIC3_**test**.json. If the file name begin with "RE_", it means that this file is for **RE**(relation extraction) training. Otherwise, it's for **NER**(named entity recognition) training.

- **no_validation** in "dataset_kwargs": true or false. If it's 'false', we should add a new parameter(validation_files) into "dataset_kwargs", which is used in "Fed-MedLoRA+" algorithm.

### Evaluations

In this study, we calculate the **strict and lenient F1 scores** of different settings on **[NER](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/evaluate.sh)** and 
**[RE](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/re_evaluate.sh)** tasks. 
Inference and evaluation are calculated at the same time.













