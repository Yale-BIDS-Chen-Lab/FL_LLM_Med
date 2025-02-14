from cyy_torch_toolbox import Inferencer, TensorDict, TextDatasetCollection
from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from distributed_learning_simulation import AggregationServer
from datapipeline_mixin import DatapipelineMixin

__all__ = ["FinetuneAdaptorServer", "LLMTextServer"]


class LLMTextServer(AggregationServer, DatapipelineMixin):
    added_transform = False

    def get_tester(self) -> Inferencer:
        inferencer = super().get_tester()
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        if not self.added_transform:
            for transform in self.get_text_pipeline().transforms:
                inferencer.dataset_collection.append_text_transform(transform)
            self.added_transform = True
        if inferencer.dataset_collection.prompt is None:
            inferencer.dataset_collection.set_prompt(self.read_prompt())
        return inferencer


class FinetuneAdaptorServer(AggregationServer):
    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        model_evaluator = tester.model_evaluator
        assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)
        model_evaluator.load_perf_model_state_dict(parameter, device=tester.device)
