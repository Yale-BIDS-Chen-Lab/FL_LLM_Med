from typing import Any

import torch
from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_naive_lib.log import log_debug, log_warning
from cyy_torch_toolbox import Inferencer, MachineLearningPhase, TensorDict
from peft import PeftModel

from ..sft import SFTTrainerMinxin, load_perf_model_state_dict
from .common import LLMTextServer

__all__ = ["SFTServer"]


class SFTServer(LLMTextServer, SFTTrainerMinxin):
    cached_validator: None | Inferencer = None

    def get_validator(self, for_evaluation: bool = False) -> Inferencer:
        if self.cached_validator is None:
            self.cached_validator = super().get_tester(
                phase=MachineLearningPhase.Validation, for_evaluation=for_evaluation
            )
        return self.cached_validator

    def _get_init_model(self) -> TensorDict:
        init_global_model_path = self.config.algorithm_kwargs.get(
            "global_model_path", None
        )
        tester = self.get_tester()
        evaluator = tester.model_evaluator
        assert isinstance(evaluator, HuggingFaceModelEvaluatorForFinetune)
        if init_global_model_path is None:
            return HuggingFaceModelEvaluatorForFinetune.get_perf_model_state_dict(
                evaluator.model
            )

        finetuned_model = PeftModel.from_pretrained(
            model=evaluator.underlying_model, model_id=init_global_model_path
        )
        self.load_parameter(
            tester=tester,
            parameter=HuggingFaceModelEvaluatorForFinetune.get_perf_model_state_dict(
                finetuned_model
            ),
        )
        return self.sft_get_perf_model_state_dict()

    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        sft_trainer = self.get_sft_trainer(tester)
        log_debug("load parameter to device %s", tester.device)
        load_perf_model_state_dict(sft_trainer.model, parameter, device=tester.device)

    def _get_metric(self, tester: Inferencer) -> Any:
        sft_trainer = self.get_sft_trainer()
        sft_trainer.model.to(device=tester.device)
        with torch.inference_mode():
            metrics = sft_trainer.evaluate(
                eval_dataset=self.get_sft_trainer_dataset(executor=tester)
            )
            log_warning("metric is %s", metrics)
            return metrics

    def _server_exit(self) -> None:
        sft_trainer = self.get_sft_trainer()
        sft_trainer.save_model()
