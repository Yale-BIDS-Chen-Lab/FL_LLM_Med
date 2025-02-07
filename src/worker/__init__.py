from cyy_torch_toolbox import TensorDict, TextDatasetCollection
from distributed_learning_simulation import AggregationWorker

from datapipeline_mixin import DatapipelineMixin

__all__ = ["FinetuneAdaptorWorker"]


class LLMTextWorker(AggregationWorker, DatapipelineMixin):
    def _before_training(self) -> None:
        super()._before_training()
        self.dataset_collection.set_prompt(self.read_prompt())

    @property
    def dataset_collection(self) -> TextDatasetCollection:
        assert isinstance(self.trainer.dataset_collection, TextDatasetCollection)
        return self.trainer.dataset_collection


class FinetuneAdaptorWorker(LLMTextWorker):
    def _before_training(self) -> None:
        super()._before_training()
        self._model_loading_fun = self._load_adaptor

    def _get_parameters(self) -> TensorDict:
        return self.trainer.model_evaluator.get_perf_model_state_dict(
            self.trainer.model
        )

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        self.trainer.model_evaluator.load_perf_model_state_dict(adaptor_parameter)
