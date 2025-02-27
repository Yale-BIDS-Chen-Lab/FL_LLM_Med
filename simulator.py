import os
import sys

from cyy_naive_lib.log import redirect_stdout_to_logger

os.environ["TQDM_DISABLE"] = "1"
os.environ["LEAST_REQUIRED_DEVICE_MEMORY_IN_GB"] = "10"
from distributed_learning_simulation import (
    load_config,
    train,
)

src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)
import method  # noqa: F401

if __name__ == "__main__":
    config_path = os.path.join(src_path, "..", "conf")
    with redirect_stdout_to_logger(logger_names=["transformers"]):
        config = load_config(
            config_path=config_path,
            global_conf_path=os.path.join(config_path, "global.yaml"),
        )
        # config.heavy_server = False
        # if config.trainer_config.hook_config.use_amp:
        #     log_warning("AMP may slowdown training and increase GPU memory")
        train(config=config, single_task=True)
