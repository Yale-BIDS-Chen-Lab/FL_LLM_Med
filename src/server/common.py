import os

from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import (
    ClassificationDatasetCollection,
    Inferencer,
    MachineLearningPhase,
    TensorDict,
    TextDatasetCollection,
)
from distributed_learning_simulation import AggregationServer

from ..datapipeline_mixin import DatapipelineMixin


class LLMTextServer(AggregationServer, DatapipelineMixin):
    def get_tester(
        self,
        phase: MachineLearningPhase = MachineLearningPhase.Test,
        for_evaluation: bool = False,
    ) -> Inferencer:
        self.config.model_config.model_kwargs.pop("load_in_4bit", None)
        self.config.model_config.model_kwargs.pop("load_in_8bit", None)
        if "finetune_config" not in self.config.model_config.model_kwargs:
            self.config.model_config.model_kwargs["finetune_config"] = {}
        self.config.model_config.model_kwargs["finetune_config"]["inference_mode"] = (
            True
        )
        inferencer = super().get_tester(phase=phase)
        assert isinstance(
            inferencer.dataset_collection,
            TextDatasetCollection | ClassificationDatasetCollection,
        )
        if inferencer.dataset_collection.prompt is None:
            self.set_prompt(
                dc=inferencer.dataset_collection, for_evaluation=for_evaluation
            )
        return inferencer


class FinetuneAdaptorServer(LLMTextServer):
    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        model_evaluator = tester.model_evaluator
        assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)
        model_evaluator.load_perf_model_state_dict(parameter, device=tester.device)

    def _server_exit(self) -> None:
        assert self.current_aggregated_model.has_data
        tester = self.get_tester(for_evaluation=False)
        # merge Rola layers
        tester.replace_model(lambda old_model: old_model.merge_and_unload())
        model_evaluator = tester.model_evaluator
        assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)
        model_evaluator.save_pretrained(os.path.join(self.save_dir, "finetuned_model"))
