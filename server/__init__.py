from cyy_torch_toolbox import Inferencer, TextDatasetCollection
from datapipeline_mixin import DatapipelineMixin
from distributed_learning_simulation import (
    AggregationServer,
)


class LLMTextServer(AggregationServer, DatapipelineMixin):
    def get_tester(self, *args, **kwargs) -> Inferencer:
        inferencer = super().get_tester(*args, **kwargs)
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        inferencer.dataset_collection.set_prompt(self.read_prompt())
        return inferencer
