from __future__ import annotations
import torch


from trl import SFTTrainer

from ..method_forward import (
    SFTTrainerWorker,
)


class FedSALoRATrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.lora_lambda1 = kwargs.pop("lora_lambda1", 0.01)
        self.lora_lambda2 = kwargs.pop("lora_lambda2", 0.01)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if return_outputs:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, **kwargs
            )
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)

        reg_loss = torch.tensor(0.0, device=loss.device)
        l1 = self.lora_lambda1
        l2 = self.lora_lambda2

        if l1 > 0 or l2 > 0:
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    # Y is (r, d_in), column-wise norm corresponds to input features
                    reg_loss += l2 * torch.norm(param, p=2, dim=0).sum()
                elif "lora_B" in name:
                    # B is (d_out, r), row-wise norm corresponds to output features
                    reg_loss += l1 * torch.norm(param, p=2, dim=1).sum()

        loss += reg_loss
        return (loss, outputs) if return_outputs else loss


class FedSALoRAWorker(SFTTrainerWorker):
    old_state: None | TensorDict = None
    _lora_lambda1: float = 0.01
    _lora_lambda2: float = 0.01

    def _before_training(self) -> None:
        # Keep group-lasso coefficients as trainer-only settings to avoid
        # polluting model kwargs/state handling.
        candidates = []
        try:
            candidates.append(self.config.model_config.model_kwargs)
        except Exception:
            pass
        try:
            candidates.append(self.trainer.mutable_model_config.model_kwargs)
        except Exception:
            pass
        for kwargs in candidates:
            if not isinstance(kwargs, dict):
                continue
            if "lora_lambda1" in kwargs:
                self._lora_lambda1 = float(kwargs.pop("lora_lambda1"))
            if "lora_lambda2" in kwargs:
                self._lora_lambda2 = float(kwargs.pop("lora_lambda2"))
        try:
            algo_kwargs = self.config.algorithm_kwargs
            if isinstance(algo_kwargs, dict):
                if "lora_lambda1" in algo_kwargs:
                    self._lora_lambda1 = float(algo_kwargs["lora_lambda1"])
                if "lora_lambda2" in algo_kwargs:
                    self._lora_lambda2 = float(algo_kwargs["lora_lambda2"])
        except Exception:
            pass
        super()._before_training()

    def get_sft_trainer_cls(self) -> type[SFTTrainer]:
        return FedSALoRATrainer

    def get_sft_trainer_kwargs(self, executor: Executor) -> dict:
        return {
            "lora_lambda1": self._lora_lambda1,
            "lora_lambda2": self._lora_lambda2,
        }

    def _get_parameters(self) -> TensorDict:
        if not self._stopped():
            self.old_state = tensor_clone(super()._get_parameters())
            assert self.old_state is not None
            state = {k: v for k, v in self.old_state.items() if "lora_A" in k}
            assert state
            return state
        log_warning("Send full adaptor")
        assert self.old_state is not None
        return self.old_state

    def _after_training(self) -> None:
        self._in_after_training = True
        message = self._get_sent_data()
        message.end_training = True
        log_warning("Send final adaptor")
        self._aggregation(sent_data=message)
        sft_trainer = self.get_sft_trainer(self.trainer)
        sft_trainer.save_model()
        super()._after_training()

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        if self.old_state is None:
            super()._load_adaptor(adaptor_parameter)
            return

        adaptor_parameter = {
            k: v for k, v in adaptor_parameter.items() if "lora_A" in k
        }
        log_debug("Received layers:%s", list(adaptor_parameter.keys()))
        assert adaptor_parameter
        for k, v in adaptor_parameter.items():
            assert k in self.old_state
            self.old_state[k] = v
        super()._load_adaptor(self.old_state)
