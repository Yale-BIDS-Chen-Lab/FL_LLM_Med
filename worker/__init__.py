from cyy_torch_toolbox import TextDatasetCollection
from distributed_learning_simulation import (
    AggregationWorker,
)

from ..datapipeline_mixin import DatapipelineMixin


class LLMTextWorker(AggregationWorker, DatapipelineMixin):
    def _before_training(self) -> None:
        super()._before_training()
        self.dataset_collection.set_prompt(self.read_prompt())

    @property
    def dataset_collection(self) -> TextDatasetCollection:
        assert isinstance(self.trainer.dataset_collection, TextDatasetCollection)
        return self.trainer.dataset_collection
