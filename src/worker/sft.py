import os

from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import TensorDict
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from .common import FinetuneAdaptorWorker

__all__ = ["SFTTrainerWorker"]


class SFTTrainerWorker(FinetuneAdaptorWorker):
    sft_trainer: None | SFTTrainer = None

    def __formatting_func(self, sample) -> str:
        return sample["input"]

    def create_sft_trainer(self) -> None:
        if self.trainer.device.type.lower() == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.trainer.device.index)
        os.environ["NO_TOKENIZER_TRANSFORMS"] = "1"
        if self.sft_trainer is not None:
            return
        learning_rate = self.trainer.hyper_parameter.learning_rate
        assert isinstance(learning_rate, float)

        training_args = SFTConfig(
            per_device_train_batch_size=self.trainer.hyper_parameter.batch_size,
            # max_grad_norm=0.3,
            num_train_epochs=self.trainer.hyper_parameter.epoch,
            learning_rate=learning_rate,
            gradient_checkpointing=False,
            bf16=True,
            save_total_limit=0,
            output_dir=os.path.join(self.save_dir, "SFTTrainer"),
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            max_seq_length=self.config.dc_config.dataset_kwargs.get(
                "input_max_len", 1024
            ),
        )

        self.model_evaluator.to_device(device=self.trainer.device)
        model = self.trainer.model
        assert hasattr(model, "hf_device_map")
        model.hf_device_map = {"": self.trainer.device}
        self.sft_trainer = SFTTrainer(
            model,
            train_dataset=Dataset.from_list(self.trainer.dataloader.dataset),
            formatting_func=self.__formatting_func,
            args=training_args,
        )

    def _train(self, first_training: bool, training_kwargs: dict) -> None:
        assert not training_kwargs
        self.create_sft_trainer()
        assert self.sft_trainer is not None
        self.sft_trainer.train()
        self._aggregation(sent_data=self._get_sent_data())

    def _get_parameters(self) -> TensorDict:
        assert self.sft_trainer is not None
        return self.model_evaluator.get_perf_model_state_dict(self.sft_trainer.model)
