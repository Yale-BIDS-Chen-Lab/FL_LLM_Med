import logging
import os
import sys
from contextlib import redirect_stdout

from cyy_naive_lib.log import (
    initialize_proxy_logger,
    replace_default_logger,
    replace_logger,
)

os.environ["TQDM_DISABLE"] = "1"
from cyy_naive_lib.log import log_warning
from distributed_learning_simulation import (
    load_config,
    train,
)


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger) -> None:
        self.logger = logger

    def write(self, buf) -> None:
        for line in buf.rstrip().splitlines():
            self.logger.info(line.rstrip())

    def flush(self) -> None:
        pass


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
    initialize_proxy_logger()
    config = load_config(
        config_path=config_path,
        global_conf_path=os.path.join(config_path, "global.yaml"),
    )
    config.heavy_server = False
    if config.trainer_config.hook_config.use_amp:
        log_warning("AMP may slowdown training and increase GPU memory")
    config.preallocate_device = True

    with redirect_stdout(StreamToLogger(logging.getLogger())):
        train(config=config, single_task=True)
