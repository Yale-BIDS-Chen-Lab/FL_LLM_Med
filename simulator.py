import os
import sys

import hydra
from distributed_learning_simulation.config import (
    DistributedTrainingConfig,
)
from distributed_learning_simulation.config import load_config as __load_config
from distributed_learning_simulation.training import train

sys.path.insert(0, os.path.abspath("src"))
import method  # noqa: F401

global_config: DistributedTrainingConfig = DistributedTrainingConfig()


@hydra.main(config_path="./conf", version_base=None)
def load_config(conf) -> None:
    global global_config
    global_config = __load_config(
        conf,
        os.path.join(os.path.dirname(__file__), "conf", "global.yaml"),
        import_libs=False,
    )


if __name__ == "__main__":
    load_config()
    # To avoid OOM
    global_config.worker_number_per_process = int(global_config.worker_number / 2)
    print(global_config.dc_config.dataset_kwargs)
    for k, v in global_config.dc_config.dataset_kwargs.items():
        if "files" in k:
            if isinstance(v, str) and v.startswith("data/"):
                global_config.dc_config.dataset_kwargs[k] = str(
                    os.path.join(os.path.abspath("data"), v[len("data/") :])
                )
                print(global_config.dc_config.dataset_kwargs[k])

    train(config=global_config)
