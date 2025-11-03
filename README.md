# FL_LLM_Med

This is a simulator of Federated Learning for LLM fine-tuning on a single host. It implements our works.

## Install Environment

This is a Python project. The third party dependencies are listed in [pyproject.toml]([https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/pyproject.toml](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/pyproject.toml)).

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

Zero-shot means we didn't fine-tune the models. Single site is without federated learning support. Centralized training pools data from all sites to train a single model. 

### Models

**[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)**, 
**[LLaMA3-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)**, and 
**[DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)** 
are open models, used in our experiments.

### Training

For model training, we use [train_bert.sh]([https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/train_bert.sh](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/train_bert.sh) to train **Bio_ClinicalBERT** model, 
and use [train_mix.sh]([https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/train_mix.sh](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/train_mix.sh) to train other models (**LLaMA3-8B** and **DeepSeek-R1-Distill-Llama-8B**). 
Both of these corresponding configuration files are located in a subfolder of [conf]([https://github.com/cyyever/distributed_LLM_learning_simulator/tree/main/conf](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/tree/main/conf). 
Modify the contents of common.yaml to change the configuration parameters. 

#### Parameters in common.yaml

Take train_mix.sh as an example, introduce these parameters in [common.yaml]([https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/conf/medical_mix/common.yaml](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/conf/medical_mix/common.yaml).

- **dataset_name**:

- **dataset_sampling**: file_split or random_split. "file_split" means that one dataset is only distributed to one client. "random_split" means the sentences in one dataset distributed to diffenent clients are random.

- **distributed_algorithm**: adaptor_avg or fed_avg. "fed_avg" is for Bio_ClinicalBERT model. "adaptor_avg" is for others.

- **train_files** and **test_files** in "dataset_kwargs": The test_files are the corresponding the train_files, such as RE_MIMIC3_**train**.json and RE_MIMIC3_**test**.json. If the file name begin with "RE_", it means that this file is for **RE**(relation extraction) training. Otherwise, it's for **NER**(named entity recognition) training.

- **no_validation** in "dataset_kwargs": true or false. If it's 'false', we should add a new parameter(validation_files) into "dataset_kwargs", which is used in "Fed-MedLoRA+" algorithm.

### Evaluations

In this study, we calculate the **strict and lenient F1 scores** of different settings on **[NER]([https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/evaluate.sh](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/evaluate.sh)** and 
**[RE]([https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/re_evaluate.sh](https://github.com/Yale-BIDS-Chen-Lab/FL_LLM_Med/blob/main/re_evaluate.sh)** tasks. 
Inference and evaluation are calculated at the same time.






