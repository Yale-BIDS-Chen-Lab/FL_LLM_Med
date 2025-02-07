from typing import Any
from cyy_torch_toolbox import Inferencer, TensorDict, TextDatasetCollection

from ..method_forward import LLMTextServer
from .data_pipeline import get_pipeline


class FinetuneAdaptorServer(LLMTextServer):
    def get_tester(self, *args: Any, **kwargs: Any) -> Inferencer:
        inferencer = super().get_tester(*args, **kwargs)
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        for transform in get_pipeline().transforms:
            inferencer.dataset_collection.append_text_transform(transform)
        return inferencer

    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        tester.model_evaluator.load_perf_model_state_dict(parameter)
