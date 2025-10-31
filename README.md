# distributed_LLM_learning_simulator

This is a simulator of Federated Learning for LLM fine-tuning on a single host. It implements our works.

## Install Environment

This is a Python project. The third party dependencies are listed in [pyproject.toml](https://github.com/cyyever/distributed_LLM_learning_simulator/blob/main/pyproject.toml).

Use PIP to set up the environment:

```
python3 -m pip install . --upgrade --force-reinstall --user
```

## Original and our case study datasets

The following datasets are used for training and testing benchmark:
- MIMIC-III
- MTSamples
- UTP
  
The independent validation set is below:
- I2B2

The case study on new annotation is used:
- YNHH


