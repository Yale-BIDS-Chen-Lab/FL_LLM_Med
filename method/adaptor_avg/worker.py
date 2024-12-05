import os
import sys

from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import TensorDict, TransformType

worker_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "worker"
)
sys.path.append(worker_path)

from worker.aggregation_worker import LLMAggregationWorker


class FinetuneAdaptorWorker(LLMAggregationWorker):
    def _before_training(self) -> None:
        super()._before_training()
        self.trainer.dataset_collection.append_transform(
            self.format_input, key=TransformType.InputText
        )

    def _get_sent_parameters(self) -> TensorDict:
        if self._model_loading_fun is None:
            self._model_loading_fun = self._load_adaptor
        assert isinstance(
            self.trainer.model_evaluator, HuggingFaceModelEvaluatorForFinetune
        )
        return self.trainer.model_evaluator.get_perf_model_state_dict(
            self.trainer.model
        )

    @classmethod
    def format_input(cls, example) -> str:
        sample = example["data"]
        return "\n".join(
            ["### Input", sample["input"], "### Output", str(sample["output"])]
        )

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        assert isinstance(
            self.trainer.model_evaluator, HuggingFaceModelEvaluatorForFinetune
        )
        self.trainer.model_evaluator.load_perf_model_state_dict(adaptor_parameter)
