from cyy_torch_toolbox import Inferencer, TensorDict, TextDatasetCollection, tensor_to
from distributed_learning_simulation import AggregationServer

from datapipeline_mixin import DatapipelineMixin

__all__ = ["FinetuneAdaptorServer"]


class LLMTextServer(AggregationServer, DatapipelineMixin):
    def get_tester(self) -> Inferencer:
        inferencer = super().get_tester()
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        if inferencer.dataset_collection.prompt is None:
            inferencer.dataset_collection.set_prompt(self.read_prompt())
        return inferencer


class FinetuneAdaptorServer(LLMTextServer):
    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        tester.model_evaluator.load_perf_model_state_dict(
            parameter, device=tester.device
        )
