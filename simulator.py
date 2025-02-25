import os
import sys
from contextlib import redirect_stdout

from cyy_naive_lib.log import (
    StreamToLogger,
    replace_default_logger,
    replace_logger,
)

os.environ["TQDM_DISABLE"] = "1"
from distributed_learning_simulation import (
    load_config,
    train,
)

config_path = os.path.join(os.path.dirname(__file__), "conf")
src_path = os.path.join(config_path, "..", "src")
sys.path.insert(0, src_path)
import method  # noqa: F401

if __name__ == "__main__":
    # disable hydra output dir
    for option in [
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]:
        sys.argv.append(option)
    replace_default_logger()
    replace_logger("transformers")

    with redirect_stdout(StreamToLogger()):
        config = load_config(
            config_path=config_path,
            global_conf_path=os.path.join(config_path, "global.yaml"),
        )
        # config.heavy_server = False
        # if config.trainer_config.hook_config.use_amp:
        #     log_warning("AMP may slowdown training and increase GPU memory")
        config.preallocate_device = True
        train(config=config, single_task=True)
