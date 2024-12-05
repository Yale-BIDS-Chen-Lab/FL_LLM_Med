import os

from distributed_learning_simulation import (
    AggregationWorker,
)


class LLMAggregationWorker(AggregationWorker):
    def _before_training(self) -> None:
        super()._before_training()
        prompt_file = self.config.dataset_kwargs["prompt_file"]
        prompt_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "prompt", prompt_file
        )
        with open(prompt_file, encoding="utf8") as f:
            self.trainer.dataset_collection.set_prompt(f.read())
