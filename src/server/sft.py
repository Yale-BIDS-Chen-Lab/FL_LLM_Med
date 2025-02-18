import os
import torch
import sys
from typing import Any

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)

from cyy_torch_toolbox import Inferencer, TensorDict
from datasets import Dataset

from sft import SFTTrainerMinxin, load_perf_model_state_dict

from .common import FinetuneAdaptorServer

__all__ = ["SFTServer"]


class SFTServer(FinetuneAdaptorServer, SFTTrainerMinxin):
    # performance_tester: None | Inferencer = None
    #
    # def setup_tester_for_performance(
    #     self,
    #     parameter: ModelParameter | ParameterMessage,
    #     log_performance_metric: bool = True,
    # ) -> Inferencer:
    #     if self.performance_tester is None:
    #         self.performance_tester = super().setup_tester_for_performance(
    #             parameter=parameter, log_performance_metric=log_performance_metric
    #         )
    #     return self.performance_tester
    cached_tester: None | Inferencer = None

    def load_parameter(self, tester: Inferencer, parameter: TensorDict) -> None:
        self.cached_tester = tester
        sft_trainer = self.get_sft_trainer(tester)
        load_perf_model_state_dict(
            sft_trainer.model_wrapped, parameter, device=tester.device
        )

    def _get_metric(self, tester: Inferencer) -> Any:
        sft_trainer = self.get_sft_trainer()
        sft_trainer.predict(test_dataset=self.get_evaluation_dataset())
        sft_trainer.save_model()
        return {}

    def get_evaluation_dataset(self, tester: Inferencer) -> Dataset:
        assert self.cached_tester is not None
        dataset = Dataset.from_list(self.cached_tester.dataloader.dataset)
        tokenizer = self.cached_tester.model_evaluator.tokenizer

        def preprocess_function(examples):
            return tokenizer(examples["input"], truncation=True)

        return dataset.map(preprocess_function, batched=True)
