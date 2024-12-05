import os

from cyy_torch_toolbox import Inferencer
from distributed_learning_simulation import (
    AggregationServer,
)


class LLMAggregationServer(AggregationServer):
    def get_tester(self, *args, **kwargs) -> Inferencer:
        inferencer = super().get_tester(*args, **kwargs)
        prompt_file = self.config.dc_config.dataset_kwargs["prompt_file"]
        prompt_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "prompt", prompt_file
        )
        with open(prompt_file, encoding="utf8") as f:
            inferencer.dataset_collection.set_prompt(f.read())
        return inferencer
