import os
import sys

os.environ["WANDB_DISABLED"] = "true"
from cyy_naive_lib.log import log_warning
from distributed_learning_simulation import (
    load_config,
    train,
)

config_path = os.path.join(os.path.dirname(__file__), "conf")
src_path = os.path.join(config_path, "..", "src")
sys.path.insert(0, src_path)
import method  # noqa: F401

if __name__ == "__main__":
    config = load_config(
        config_path=config_path,
        global_conf_path=os.path.join(config_path, "global.yaml"),
    )
    if config.use_amp:
        log_warning("AMP may slowdown training and increase GPU memory")
    train(config=config, single_task=True)
