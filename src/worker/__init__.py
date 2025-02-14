from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import TensorDict, TextDatasetCollection
from distributed_learning_simulation import AggregationWorker

from datapipeline_mixin import DatapipelineMixin

__all__ = ["FinetuneAdaptorWorker"]


class LLMTextWorker(AggregationWorker, DatapipelineMixin):
    def _before_training(self) -> None:
        self._send_parameter_diff: bool = False
        super()._before_training()
        for transform in self.get_text_pipeline().transforms:
            self.dataset_collection.append_text_transform(transform)
        self.dataset_collection.set_prompt(self.read_prompt())

    @property
    def dataset_collection(self) -> TextDatasetCollection:
        assert isinstance(self.trainer.dataset_collection, TextDatasetCollection)
        return self.trainer.dataset_collection


class FinetuneAdaptorWorker(LLMTextWorker):
    def _before_training(self) -> None:
        super()._before_training()
        self._model_loading_fun = self._load_adaptor
        with self.context.global_store.default_lock:
            if self.context.thread_local_store.has("tokenizer"):
                self.model_evaluator.set_tokenizer(
                    self.context.thread_local_store.get("tokenizer")
                )
            else:
                self.context.thread_local_store.store(
                    "tokenizer", self.model_evaluator.tokenizer
                )
        if self.hold_log_lock:
            log_info("model is %s", self.trainer.model)

    @property
    def model_evaluator(self) -> HuggingFaceModelEvaluatorForFinetune:
        return self.trainer.model_evaluator

    def pause(self, in_round: bool = False) -> None:
        if not in_round:
            self.model_evaluator.set_tokenizer(None)
        super().pause(in_round=in_round)

    def _get_parameters(self) -> TensorDict:
        return self.model_evaluator.get_perf_model_state_dict(self.trainer.model)

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        self.model_evaluator.load_perf_model_state_dict(
            adaptor_parameter, device=self.trainer.device
        )
