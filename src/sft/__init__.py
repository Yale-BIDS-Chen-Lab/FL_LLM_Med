import os
from typing import Protocol

import torch
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import Config, Executor, TensorDict, Trainer, tensor_to
from datasets import Dataset
from distributed_learning_simulation import ExecutorProtocol
from peft.utils.save_and_load import set_peft_model_state_dict
from trl import SFTConfig, SFTTrainer

__all__ = ["SFTTrainerMinxin", "get_SFTConfig", "load_perf_model_state_dict"]


def load_perf_model_state_dict(
    model, state_dict: TensorDict, device: torch.device
) -> None:
    state_dict = tensor_to(state_dict, device=device)
    _, unexpected_keys = set_peft_model_state_dict(
        model=model,
        peft_model_state_dict=state_dict,
        ignore_mismatched_sizes=False,
    )
    assert not unexpected_keys


def get_SFTConfig(config: Config, executor: Executor, output_dir: str) -> SFTConfig:
    learning_rate = 2.0e-5
    if isinstance(executor, Trainer):
        learning_rate = executor.hyper_parameter.learning_rate
    assert isinstance(learning_rate, float)
    return SFTConfig(
        per_device_train_batch_size=executor.hyper_parameter.batch_size,
        num_train_epochs=executor.hyper_parameter.epoch,
        learning_rate=learning_rate,
        bf16=True,
        save_total_limit=0,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=config.dc_config.dataset_kwargs.get("input_max_len", 1024),
    )


class SFTTrainerMinxin(ExecutorProtocol, Protocol):
    __sft_trainer: None | SFTTrainer = None

    @property
    def trainer(self) -> Trainer: ...

    def get_training_dataset(self):
        return Dataset.from_list([])

    def get_evaluation_dataset(self):
        return Dataset.from_list([])

    def __formatting_func(self, sample) -> str:
        return sample["input"]

    def get_sft_trainer(self) -> SFTTrainer:
        os.environ["NO_TOKENIZER_TRANSFORMS"] = "1"
        if self.__sft_trainer is not None:
            return self.__sft_trainer
        log_info("perform create_sft_trainer")
        learning_rate = self.trainer.hyper_parameter.learning_rate
        assert isinstance(learning_rate, float)
        device = torch.device("cuda:0")
        # self.trainer.set_device_fun(self.context.get_device)
        # device = self.trainer.device
        log_info("use device %s", device)
        if device.type.lower() == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)
            device = torch.device("cuda:0")

        self.trainer.mutable_model_config.model_kwargs["device_map"] = {"": device}
        output_dir = os.path.join(self.save_dir, "SFTTrainer")
        training_args = get_SFTConfig(
            config=self.config, executor=self.trainer, output_dir=output_dir
        )

        # self.model_evaluator.to_device(device=device)
        model = self.trainer.model
        self.__sft_trainer = SFTTrainer(
            model,
            train_dataset=self.get_training_dataset(),
            formatting_func=self.__formatting_func,
            args=training_args,
        )
        return self.__sft_trainer
