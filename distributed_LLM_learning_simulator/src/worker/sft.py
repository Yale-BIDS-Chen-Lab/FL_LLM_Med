import os
import sys

import torch

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)

from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import TensorDict

from ..sft import SFTTrainerMinxin, load_perf_model_state_dict
from .common import LLMTextWorker

__all__ = ["SFTTrainerWorker"]


class SFTTrainerWorker(LLMTextWorker, SFTTrainerMinxin):
    _sample_size: None | int = None

    def _before_training(self) -> None:
        device = self.context.get_device()
        self.context.release_device_lock()
        self.trainer.set_device(device)
        self.trainer.mutable_model_config.model_kwargs["device_map"] = {"": device}
        self.trainer.remove_unrelated_datasets()
        self._model_loading_fun = self._load_adaptor
        super()._before_training()

    def _train(self, first_training: bool) -> None:
        if first_training:
            self.trainer._prepare_execution()
        sft_trainer = self.get_sft_trainer(self.trainer)
        sft_trainer.train()
        sent_data = self._get_sent_data()
        self.clear_sft_trainer()
        self._aggregation(sent_data=sent_data)

    def _get_parameters(self) -> TensorDict:
        return self.sft_get_perf_model_state_dict()

    def pause(self, in_round: bool = False) -> None:
        super().pause(in_round=in_round)
        if not in_round:
            self.clear_sft_trainer()
            log_info("used cuda memory: %s", torch.cuda.memory_allocated())

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        load_perf_model_state_dict(
            model=self._sft_trainer.model_wrapped
            if self._sft_trainer is not None
            else self.trainer.running_model_evaluator.model,
            state_dict=adaptor_parameter,
            device=self.trainer.device,
        )

    def get_aggregation_weight(self) -> float:
        if self._sample_size is not None:
            return self._sample_size
        self._sample_size = 0
        assert self._sft_trainer is not None
        for batch in self._sft_trainer.get_train_dataloader():
            self._sample_size += (batch["labels"][..., 1:] != -100).sum().item()
        assert self._sample_size is not None and self._sample_size > 0
        log_info("sample_size is %s", self._sample_size)
        return self._sample_size
