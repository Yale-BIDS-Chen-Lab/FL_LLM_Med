import os
import torch
from typing import Protocol

from transformers.training_args import AcceleratorConfig
from cyy_naive_lib.log import log_info, log_debug
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
    # for n in sorted(state_dict.keys()):
    #     log_info("%s is %s", n, state_dict[n])
    _, unexpected_keys = set_peft_model_state_dict(
        model=model,
        peft_model_state_dict=state_dict,
        ignore_mismatched_sizes=False,
    )
    assert not unexpected_keys


def get_SFTConfig(config: Config, executor: Executor, output_dir: str) -> SFTConfig:
    # torch.backends.cudnn.benchmark = True
    learning_rate = 2.0e-5
    if isinstance(executor, Trainer):
        learning_rate = executor.hyper_parameter.learning_rate
    assert isinstance(learning_rate, float)
    accelerate_config = AcceleratorConfig()
    accelerate_config.non_blocking = True
    return SFTConfig(
        accelerator_config=accelerate_config,
        per_device_train_batch_size=executor.hyper_parameter.batch_size,
        num_train_epochs=executor.hyper_parameter.epoch,
        learning_rate=learning_rate,
        logging_steps=0.1,
        bf16=True,
        # tf32=True,
        output_dir=output_dir,
        lr_scheduler_type="cosine",
        gradient_checkpointing=config.model_config.model_kwargs.get(
            "use_gradient_checkpointing", False
        ),
        save_steps=0.3,
        save_total_limit=1,
        save_safetensors=False,
        report_to="none",
        warmup_ratio=0.05,
        logging_nan_inf_filter=False,
        max_seq_length=config.dc_config.dataset_kwargs.get("input_max_len", 1024),
        # max_length=config.dc_config.dataset_kwargs.get("input_max_len", 1024),
    )


class SFTTrainerMinxin(ExecutorProtocol, Protocol):
    _sft_trainer: None | SFTTrainer = None

    def get_training_dataset(self):
        return None

    def get_evaluation_dataset(self):
        return None

    def _formatting_func(self, sample) -> str:
        return sample["input"]

    def get_sft_trainer(self, executor: Executor | None = None) -> SFTTrainer:
        os.environ["NO_TOKENIZER_TRANSFORMS"] = "1"
        if self._sft_trainer is not None:
            return self._sft_trainer
        assert executor is not None
        device = self.context.get_device(set_visible_device=True)
        self.context.release_device_lock()
        log_debug("use device %s", device)
        executor.set_device(device)

        output_dir = os.path.join(self.save_dir, "SFTTrainer")
        training_args = get_SFTConfig(
            config=self.config, executor=executor, output_dir=output_dir
        )
        if self.hold_log_lock:
            log_info("SFTConfig is %s", training_args)

        executor.mutable_model_config.model_kwargs["device_map"] = {"": device}

        model = executor.model
        training_dataset = Dataset.from_list(executor.dataloader.dataset)
        self._sft_trainer = SFTTrainer(
            model,
            train_dataset=training_dataset,
            formatting_func=self._formatting_func,
            args=training_args,
        )
        return self._sft_trainer
