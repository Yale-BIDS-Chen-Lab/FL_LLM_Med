from cyy_torch_toolbox import TensorDict

from ..method_forward import LLMTextWorker
from .data_pipeline import get_pipeline


class FinetuneAdaptorWorker(LLMTextWorker):
    def _before_training(self) -> None:
        super()._before_training()
        for transform in get_pipeline().transforms:
            self.dataset_collection.append_text_transform(transform)
        self._model_loading_fun = self._load_adaptor

    def _get_parameters(self) -> TensorDict:
        return self.trainer.model_evaluator.get_perf_model_state_dict(
            self.trainer.model
        )

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        self.trainer.model_evaluator.load_perf_model_state_dict(adaptor_parameter)
