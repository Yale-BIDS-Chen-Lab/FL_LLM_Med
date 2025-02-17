from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import TensorDict
from datasets import Dataset

from ..sft import SFTTrainerMinxin, load_perf_model_state_dict
from .common import FinetuneAdaptorWorker

__all__ = ["SFTTrainerWorker"]


class SFTTrainerWorker(FinetuneAdaptorWorker, SFTTrainerMinxin):
    def _train(self, first_training: bool, training_kwargs: dict) -> None:
        assert not training_kwargs
        sft_trainer = self.get_sft_trainer()
        sft_trainer.train()
        sft_trainer.save_model()
        self._aggregation(sent_data=self._get_sent_data())

    def _get_parameters(self) -> TensorDict:
        return HuggingFaceModelEvaluatorForFinetune.get_perf_model_state_dict(
            self.get_sft_trainer().model_wrapped
        )

    def get_training_dataset(self):
        return Dataset.from_list(self.trainer.dataloader.dataset)

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        load_perf_model_state_dict(
            model=self.get_sft_trainer().model_wrapped,
            state_dict=adaptor_parameter,
            device=self.trainer.device,
        )
