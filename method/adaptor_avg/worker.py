from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import TensorDict
from distributed_learning_simulation import (
    AggregationWorker,
)


class FinetuneAdaptorWorker(AggregationWorker):
    def _get_sent_parameters(self) -> TensorDict:
        if self._model_loading_fun is None:
            self._model_loading_fun = self._load_adaptor
        assert isinstance(
            self.trainer.model_evaluator, HuggingFaceModelEvaluatorForFinetune
        )
        return self.trainer.model_evaluator.get_perf_model_state_dict(
            self.trainer.model
        )

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        assert isinstance(
            self.trainer.model_evaluator, HuggingFaceModelEvaluatorForFinetune
        )
        self.trainer.model_evaluator.load_perf_model_state_dict(adaptor_parameter)

    # def _load_result_from_server(self, result: Message) -> None:
    #     model_path = os.path.join(
    #         self.save_dir, "aggregated_model", f"round_{self.round_index}.pk"
    #     )
    #     parameter: ModelParameter = {}
    #     match result:
    #         case ParameterMessage():
    #             parameter = result.parameter
    #             if self._keep_model_cache or self._send_parameter_diff:
    #                 self._model_cache.cache_parameter(result.parameter, path=model_path)
    #         case DeltaParameterMessage():
    #             assert self._model_cache.has_data
    #             self._model_cache.add_parameter_diff(
    #                 result.delta_parameter, path=model_path
    #             )
    #             parameter = self._model_cache.parameter
    #         case _:
    #             raise NotImplementedError()
    #     load_parameters(
    #         trainer=self.trainer,
    #         parameter=parameter,
    #         reuse_learning_rate=self._reuse_learning_rate,
    #         loading_fun=self._model_loading_fun,
    #     )
    #     if result.end_training:
    #         self._force_stop = True
    #         raise StopExecutingException()
