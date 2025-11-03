import gc
import os
import uuid
from typing import Protocol

import torch
from cyy_huggingface_toolbox import (
    HuggingFaceModelEvaluator,
    HuggingFaceModelEvaluatorForFinetune,
)
from cyy_naive_lib.log import log_debug, log_error, log_info
from cyy_torch_toolbox import Config, Executor, TensorDict, Trainer, tensor_to
from datasets import Dataset
from distributed_learning_simulation import ExecutorProtocol
from peft.utils.save_and_load import set_peft_model_state_dict
from transformers.trainer_pt_utils import AcceleratorConfig
from trl import SFTConfig, SFTTrainer

__all__ = ["SFTTrainerMinxin", "get_SFTConfig", "load_perf_model_state_dict"]


def load_perf_model_state_dict(
    model, state_dict: TensorDict, device: torch.device
) -> None:
    state_dict = tensor_to(state_dict, device=device)
    total_state_size = 0
    for k in state_dict.values():
        total_state_size += k.nbytes
    log_info("total_state_size is %s", total_state_size)
    _, unexpected_keys = set_peft_model_state_dict(
        model=model,
        peft_model_state_dict=state_dict,
        ignore_mismatched_sizes=False,
    )
    assert not unexpected_keys


def get_SFTConfig(config: Config, executor: Executor, output_dir: str) -> SFTConfig:
    # torch.backends.cudnn.benchmark = True
    learning_rate = 2.0e-4
    if isinstance(executor, Trainer):
        assert isinstance(executor.hyper_parameter.learning_rate, float)
        learning_rate = executor.hyper_parameter.learning_rate
    accelerate_config = AcceleratorConfig()
    accelerate_config.non_blocking = True
    return SFTConfig(
        accelerator_config=accelerate_config.to_dict(),
        per_device_train_batch_size=executor.hyper_parameter.batch_size,
        num_train_epochs=executor.hyper_parameter.epoch,
        learning_rate=learning_rate,
        logging_steps=10,
        bf16=True,
        completion_only_loss=False,
        output_dir=output_dir,
        lr_scheduler_type="cosine",
        gradient_checkpointing=config.model_config.model_kwargs.get(
            "use_gradient_checkpointing", False
        ),
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        warmup_ratio=0.05,
        logging_nan_inf_filter=False,
        eval_accumulation_steps=1,
        bf16_full_eval=True,
        prediction_loss_only=True,
        max_length=config.dc_config.dataset_kwargs.get("input_max_len", 2048),
    )


class SFTTrainerMinxin(ExecutorProtocol, Protocol):
    _sft_trainer: None | SFTTrainer = None

    @property
    def sft_trainer(self) -> SFTTrainer:
        assert self._sft_trainer is not None
        return self._sft_trainer

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

        model = executor.running_model_evaluator.underlying_model
        train_dataset = self.get_sft_trainer_dataset(executor)
        sft_trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            args=training_args,
            peft_config=executor.running_model_evaluator.peft_config,
        )
        assert sft_trainer.model is sft_trainer.model_wrapped
        self._sft_trainer = sft_trainer
        return self._sft_trainer

    def get_sft_trainer_dataset(self, executor: Executor) -> Dataset:
        log_info("dataset size is %s", len(executor.dataloader.dataset))
        dataset = Dataset.from_list(executor.dataloader.dataset)
        assert isinstance(executor.model_evaluator, HuggingFaceModelEvaluator)
        tokenizer = executor.model_evaluator.tokenizer

        def preprocess_function(examples):
            res = tokenizer(
                examples["input"],
                truncation=True,
                padding=True,
                max_length=self.config.dc_config.dataset_kwargs["input_max_len"],
            ).to(device=executor.device, non_blocking=True)
            return res

        return dataset.map(
            preprocess_function, batched=True, new_fingerprint=str(uuid.uuid4())
        )

    def sft_get_perf_model_state_dict(self) -> TensorDict:
        assert self._sft_trainer is not None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        data = HuggingFaceModelEvaluatorForFinetune.get_perf_model_state_dict(
            self._sft_trainer.model_wrapped
        )
        return tensor_to(data, device=torch.device("cpu"))

    def clear_sft_trainer(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            log_info("before clear used cuda memory: %s", torch.cuda.memory_allocated())
        if self._sft_trainer is not None:
            self._sft_trainer.model.to("cpu")
        self._sft_trainer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            log_info("after clear used cuda memory: %s", torch.cuda.memory_allocated())
