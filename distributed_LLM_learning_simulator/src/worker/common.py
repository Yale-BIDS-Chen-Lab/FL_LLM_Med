from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import (
    ClassificationDatasetCollection,
    TensorDict,
    TextDatasetCollection,
)
from distributed_learning_simulation import AggregationWorker

from ..datapipeline_mixin import DatapipelineMixin

__all__ = ["FinetuneAdaptorWorker", "LLMTextWorker"]


class LLMTextWorker(AggregationWorker, DatapipelineMixin):
    def _before_training(self) -> None:
        self._send_parameter_diff: bool = False
        super()._before_training()
        self.set_prompt(self.dataset_collection, for_evaluation=False)

    @property
    def dataset_collection(self) -> TextDatasetCollection:
        assert isinstance(
            self.trainer.dataset_collection,
            TextDatasetCollection | ClassificationDatasetCollection,
        )
        return self.trainer.dataset_collection


class FinetuneAdaptorWorker(LLMTextWorker):
    def _before_training(self) -> None:
        super()._before_training()
        self._model_loading_fun = self._load_adaptor

    @property
    def model_evaluator(self) -> HuggingFaceModelEvaluatorForFinetune:
        model_evaluator = self.trainer.model_evaluator
        assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)
        return model_evaluator

    def _get_parameters(self) -> TensorDict:
        return self.model_evaluator.get_perf_model_state_dict(self.trainer.model)

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        self.model_evaluator.load_perf_model_state_dict(
            adaptor_parameter, device=self.trainer.device
        )
