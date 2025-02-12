import os

os.environ["WANDB_DISABLED"] = "true"
import sys

from distributed_learning_simulation import (
    load_config,
    train,
)

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "conf")
src_path = os.path.join(config_path, "..", "src")
sys.path.insert(0, src_path)
import method  # noqa: F401

if __name__ == "__main__":
    config = load_config(
        config_path=config_path,
        global_conf_path=os.path.join(config_path, "global.yaml"),
    )
    train(config=config, single_task=True)
