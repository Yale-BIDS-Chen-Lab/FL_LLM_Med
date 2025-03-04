from typing import Any

import torch
from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_naive_lib.log import log_debug, log_warning
from cyy_torch_toolbox import Inferencer, TensorDict
from distributed_learning_simulation import ModelParameter

from ..sft import SFTTrainerMinxin, load_perf_model_state_dict
from .common import LLMTextServer

__all__ = ["SFTServer"]


class SFTServer(LLMTextServer, SFTTrainerMinxin):
    cached_tester: None | Inferencer = None

    def get_tester(self, for_evaluation: bool = False) -> Inferencer:
        if self.cached_tester is None:
            self.cached_tester = super().get_tester(for_evaluation=for_evaluation)
        return self.cached_tester

    def _get_init_model(self) -> ModelParameter:
        model = self.get_tester().model_evaluator.model
        return HuggingFaceModelEvaluatorForFinetune.get_perf_model_state_dict(model)

    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        assert tester is self.cached_tester
        sft_trainer = self.get_sft_trainer(tester)
        log_debug("load parameter to device %s", tester.device)
        load_perf_model_state_dict(sft_trainer.model, parameter, device=tester.device)

    def _get_metric(self, tester: Inferencer) -> Any:
        assert self.cached_tester is not None
        sft_trainer = self.get_sft_trainer()
        sft_trainer.model.to(device=self.cached_tester.device)
        with torch.inference_mode():
            metrics = sft_trainer.evaluate(
                eval_dataset=self.get_sft_trainer_dataset(executor=tester)
            )
            log_warning("metric is %s", metrics)
            return metrics

    def _server_exit(self) -> None:
        sft_trainer = self.get_sft_trainer()
        sft_trainer.save_model()
