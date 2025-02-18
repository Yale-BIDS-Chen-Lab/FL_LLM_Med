import os
import sys
from cyy_naive_lib.log import (
    replace_default_logger,
    initialize_proxy_logger,
    replace_logger,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ["TQDM_DISABLE"] = "1"
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
    replace_default_logger()
    replace_logger("transformers")
    initialize_proxy_logger()
    config = load_config(
        config_path=config_path,
        global_conf_path=os.path.join(config_path, "global.yaml"),
    )
    config.heavy_server = False
    if config.trainer_config.hook_config.use_amp:
        log_warning("AMP may slowdown training and increase GPU memory")
    config.preallocate_device = True
    train(config=config, single_task=True)
