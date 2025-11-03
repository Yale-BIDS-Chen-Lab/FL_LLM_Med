import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TQDM_DISABLE"] = "1"
os.environ["LEAST_REQUIRED_DEVICE_MEMORY_IN_GB"] = "10"

import cyy_huggingface_toolbox  # noqa: F401
from cyy_naive_lib.log import log_info, redirect_stdout_to_logger
from distributed_learning_simulation import (
    load_config,
    train,
)

project_path = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_path)
import src.method  # noqa: F401

if __name__ == "__main__":
    config_path = os.path.join(project_path, "conf")
    with redirect_stdout_to_logger(logger_names=["transformers"]):
        config = load_config(config_path=config_path, import_libs=False)
        log_info("Use train files %s", config.dc_config.dataset_kwargs["train_files"])
        log_info("Use test files %s", config.dc_config.dataset_kwargs["test_files"])
        if "validation_files" in config.dc_config.dataset_kwargs:
            log_info(
                "Use validation files %s",
                config.dc_config.dataset_kwargs["validation_files"],
            )
        train(config=config, single_task=True)
