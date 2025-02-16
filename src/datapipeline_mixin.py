import os

from cyy_torch_toolbox import DataPipeline
from distributed_learning_simulation import ExecutorProtocol


class DatapipelineMixin(ExecutorProtocol):
    def get_text_pipeline(self) -> DataPipeline:
        return DataPipeline()

    def read_prompt(self, for_evaluation: bool = False) -> str:
        prompt_file = self.config.dc_config.dataset_kwargs["prompt_file"]
        if for_evaluation:
            prompt_file = self.config.dc_config.dataset_kwargs["evaluation_prompt_file"]
        prompt_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "prompt", prompt_file
        )
        with open(prompt_file, encoding="utf8") as f:
            return f.read()
