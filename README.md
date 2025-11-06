# FL_LLM_Med

This repository provides the related codes and models for federated instruction tuning for privacy-preserving clinical large language models (LLMs). 

## Install environment

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

## Experiment configurations 
We compare our methods Fed-MedLoRA and Fed-MedLoRA+ across the following settings. 
### Clinical information extraction (IE) tasks
Named entity recognition (NER) and relation extraction (RE). 
### Models
- LLMs: [LLaMA3-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [LLaMA3-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [DeepSeek-R1-Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B), [QWen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- BERT models: [Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
### LLM Baselines
- Zero-shot
- Single-site fine-tuning
### Upper bound reference 
- Centralized fine-tuning
### Federated learning baselines 
- [FedSA-LoRA](https://arxiv.org/abs/2410.01463)

## Model fine-tuning
### Hyper-parameter configuration in common.yaml

Take the [common.yaml](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/conf/medical_ner/common.yaml) file of train_ner.sh as an example. 

- **dataset_name**: hugging_face_yale_ner
- **dataset_sampling**: file_split or random_split. Here, 'file_split' denotes that each site owns one dataset (refer to [allocation.py](src/preprocess/allocation.py)). For example, site A has MIMIC-III, site B has MTSamples and site C has UTP. 'random_split' denotes that all datasets are merged together and evenly distributed to different sites (refer to [iob_iid_split.py](src/command/iob_iid_split.py)). 
- **distributed_algorithm**: adaptor_avg or fed_avg where 'adaptor_avg' is the aggregation method for LLMs and 'fed_avg' is for BERT models. 
- **train_files** and **test_files** in 'dataset_kwargs': These two directories provide specific directories for training sets and testing sets.
- **prompt_file**: medical_html.txt
- **evaluation_prompt_file**: medical_html_evaluation.txt
- **no_validation** in 'dataset_kwargs': true or false. If it is true, then the running algorithm is Fed-MedLoRA. If it is false, then we we need to add a new hyper-parameter 'validation_files' into 'dataset_kwargs' to run the Fed-MedLoRA+ algorithm. 

### Fine-tuning LLMs and BERT models  
  - To fine-tune LLMs, e.g., LLaMA3 and DeepSeek-R1-Distill, run train_ner.sh for the NER task, train_re.sh for the RE task, and train_mix.sh for both the NER and RE tasks. 

    For example, the train_ner.sh for fine-tuning LLaMA3-8B with 2 communication rounds, 2 epochs across 3 sites is: 
    ```
    python3 ./simulator.py --config-name medical_ner/Meta-Llama-3-8B.yaml  ++medical_ner.round=2 ++medical_ner.epoch=2 ++medical_ner.worker_number=3
    ```
    If ++medical_ner.worker_number=1 and there is only one training dataset, then it is single-site fine-tuning; otherwise, it is centralized fine-tuning. The fine-tuned LoRA modules will be saved in the session directory. 
    
  - To fine-tune BERT models, run train_bert.sh with NER or RE datasets for NER or RE tasks, respectively. The fine-tuned BERT models will be saved in the session directory. 

### Evaluations

In this study, we calculate the **strict and lenient F1 scores** of different settings on **[NER](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/evaluate.sh)** and 
**[RE](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/re_evaluate.sh)** tasks. 
Inference and evaluation are calculated at the same time.






















