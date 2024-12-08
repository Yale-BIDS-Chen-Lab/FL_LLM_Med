import os
import sys

from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import TensorDict, Transform

worker_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "worker"
)
sys.path.append(worker_path)

from worker.aggregation_worker import LLMAggregationWorker


class FinetuneAdaptorWorker(LLMAggregationWorker):
    def _before_training(self) -> None:
        super()._before_training()
        self.dataset_collection.append_text_transform(Transform(fun=self.format_input))
        self._model_loading_fun = self._load_adaptor

    def _get_sent_parameter_names(self) -> set[str] | None:
        assert isinstance(
            self.trainer.model_evaluator, HuggingFaceModelEvaluatorForFinetune
        )
        return set(
            self.trainer.model_evaluator.get_perf_model_state_dict(
                self.trainer.model
            ).keys()
        )

    @classmethod
    def format_input(cls, sample) -> str:
        return "\n".join(
            ["### Input", sample["input"], "### Output", str(sample["output"])]
        )

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        assert isinstance(
            self.trainer.model_evaluator, HuggingFaceModelEvaluatorForFinetune
        )
        self.trainer.model_evaluator.load_perf_model_state_dict(adaptor_parameter)
