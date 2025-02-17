import os

import torch
from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import TensorDict, TextDatasetCollection
from datasets import Dataset
from distributed_learning_simulation import AggregationWorker
from trl import SFTConfig, SFTTrainer

from datapipeline_mixin import DatapipelineMixin

__all__ = ["FinetuneAdaptorWorker", "SFTTrainerWorker"]


class LLMTextWorker(AggregationWorker, DatapipelineMixin):
    def _before_training(self) -> None:
        self._send_parameter_diff: bool = False
        super()._before_training()
        for transform in self.get_text_pipeline().transforms:
            self.dataset_collection.append_text_transform(transform)
        self.dataset_collection.set_prompt(self.read_prompt())

    @property
    def dataset_collection(self) -> TextDatasetCollection:
        assert isinstance(self.trainer.dataset_collection, TextDatasetCollection)
        return self.trainer.dataset_collection


class FinetuneAdaptorWorker(LLMTextWorker):
    def _before_training(self) -> None:
        super()._before_training()
        self._model_loading_fun = self._load_adaptor
        with self.context.global_store.default_lock:
            if self.context.thread_local_store.has("tokenizer"):
                self.model_evaluator.set_tokenizer(
                    self.context.thread_local_store.get("tokenizer")
                )
            else:
                self.context.thread_local_store.store(
                    "tokenizer", self.model_evaluator.tokenizer
                )
        if self.hold_log_lock:
            log_info("model is %s", self.trainer.model)

    @property
    def model_evaluator(self) -> HuggingFaceModelEvaluatorForFinetune:
        return self.trainer.model_evaluator

    def pause(self, in_round: bool = False) -> None:
        if not in_round:
            self.model_evaluator.set_tokenizer(None)
        super().pause(in_round=in_round)

    def _get_parameters(self) -> TensorDict:
        return self.model_evaluator.get_perf_model_state_dict(self.trainer.model)

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        self.model_evaluator.load_perf_model_state_dict(
            adaptor_parameter, device=self.trainer.device
        )


class SFTTrainerWorker(FinetuneAdaptorWorker):
    sft_trainer: None | SFTTrainer = None

    def __formatting_func(self, sample) -> str:
        return sample["input"]

    def create_sft_trainer(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.trainer.device.index)
        os.environ["NO_TOKENIZER_TRANSFORMS"] = "1"
        if self.sft_trainer is not None:
            return
        learning_rate = self.trainer.hyper_parameter.learning_rate
        assert isinstance(learning_rate, float)

        training_args = SFTConfig(
            per_device_train_batch_size=self.trainer.hyper_parameter.batch_size,
            max_grad_norm=0.3,
            num_train_epochs=self.trainer.hyper_parameter.epoch,
            learning_rate=learning_rate,
            gradient_checkpointing=False,
            bf16=True,
            save_total_limit=0,
            output_dir=self.save_dir,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            max_seq_length=1000,
        )

        self.trainer.model_evaluator.to_device(device=self.trainer.device)
        model = self.trainer.model
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
