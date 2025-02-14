import os

from distributed_learning_simulation import ExecutorProtocol


class DatapipelineMixin(ExecutorProtocol):
    def read_prompt(self) -> str:
        prompt_file = self.config.dc_config.dataset_kwargs["prompt_file"]
        prompt_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "prompt", prompt_file
        )
        with open(prompt_file, encoding="utf8") as f:
            return f.read()
