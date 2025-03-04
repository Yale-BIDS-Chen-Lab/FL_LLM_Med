import os

from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import Inferencer, TensorDict, TextDatasetCollection
from distributed_learning_simulation import AggregationServer

from ..datapipeline_mixin import DatapipelineMixin


class LLMTextServer(AggregationServer, DatapipelineMixin):
    added_transform = False

    def get_tester(self, for_evaluation: bool = False) -> Inferencer:
        self.config.model_config.model_kwargs.pop("load_in_4bit", None)
        self.config.model_config.model_kwargs.pop("load_in_8bit", None)
        if "finetune_config" not in self.config.model_config.model_kwargs:
            self.config.model_config.model_kwargs["finetune_config"] = {}
        self.config.model_config.model_kwargs["finetune_config"]["inference_mode"] = (
            True
        )
        inferencer = super().get_tester()
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        if not self.added_transform:
            for transform in self.get_text_pipeline().transforms:
                inferencer.dataset_collection.append_text_transform(transform)
            self.added_transform = True
        if for_evaluation:
            assert inferencer.dataset_collection.prompt is None
        if inferencer.dataset_collection.prompt is None:
            inferencer.dataset_collection.set_prompt(
                self.read_prompt(for_evaluation=for_evaluation)
            )
        return inferencer


class FinetuneAdaptorServer(LLMTextServer):
    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        model_evaluator = tester.model_evaluator
        assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)
        model_evaluator.load_perf_model_state_dict(parameter, device=tester.device)

    def _server_exit(self) -> None:
        assert self.current_aggregated_model.has_data
        tester = self.get_tester()
        # merge Rola layers
        tester.replace_model(lambda old_model: old_model.merge_and_unload())
        model_evaluator = tester.model_evaluator
        assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)
        model_evaluator.save_pretrained(os.path.join(self.save_dir, "finetuned_model"))
