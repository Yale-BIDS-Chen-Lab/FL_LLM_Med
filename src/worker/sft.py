import gc
import os
import sys

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)
from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import TensorDict
from cyy_naive_lib.log import log_info

from sft import SFTTrainerMinxin, load_perf_model_state_dict

from .common import LLMTextWorker

__all__ = ["SFTTrainerWorker"]


class SFTTrainerWorker(LLMTextWorker, SFTTrainerMinxin):
    _sample_size: None | int = None

    def _before_training(self) -> None:
        super()._before_training()
        self._model_loading_fun = self._load_adaptor

    def _train(self, first_training: bool, training_kwargs: dict) -> None:
        assert not training_kwargs
        sft_trainer = self.get_sft_trainer(self.trainer)
        sft_trainer.train()
        # TODO disable
        sft_trainer.save_model()
        self.clear_sft_trainer()
        self._aggregation(sent_data=self._get_sent_data())

    def _get_parameters(self) -> TensorDict:
        return HuggingFaceModelEvaluatorForFinetune.get_perf_model_state_dict(
            self.get_sft_trainer(self.trainer).model_wrapped
        )

    def pause(self, in_round: bool = False) -> None:
        super().pause(in_round=in_round)
        if not in_round:
            self._sft_trainer = None
            gc.collect()

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        load_perf_model_state_dict(
            model=self.get_sft_trainer(self.trainer).model_wrapped,
            state_dict=adaptor_parameter,
            device=self.trainer.device,
        )

    def get_aggregation_weight(self) -> float:
        if self._sample_size is not None:
            return self._sample_size
        self._sample_size = 0
        for batch in self.get_sft_trainer().get_train_dataloader():
            self._sample_size += (batch["labels"][..., 1:] != -100).sum().item()
        assert self._sample_size is not None and self._sample_size > 0
        log_info("sample_size is %s", self._sample_size)
        return self._sample_size
