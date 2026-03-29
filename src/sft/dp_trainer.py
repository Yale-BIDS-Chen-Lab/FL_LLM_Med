from typing import Any

import torch
from cyy_naive_lib.log import log_info
from distributed_learning_simulation.dp import (
    compute_dp_sigma,
    dp_add_noise_to_gradients,
    dp_clip_gradients,
)
from transformers import TrainerCallback
from trl.trainer.sft_trainer import SFTTrainer


class _DPGradientCallback(TrainerCallback):
    """Applies DP-SGD gradient clipping and noise once per optimizer step."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma
        self._log_interval = 100
        self._step = 0

    def _compute_grad_norm(self, gradients: list[torch.Tensor]) -> float:
        return torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in gradients])
        ).item()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs["model"]
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        gradients = [p.grad for p in params_with_grad]

        should_log = self._step % self._log_interval == 0
        pre_norm = 0.0
        if should_log:
            pre_norm = self._compute_grad_norm(gradients)

        dp_clip_gradients(gradients, C=1.0)

        post_clip_norm = 0.0
        if should_log:
            post_clip_norm = self._compute_grad_norm(gradients)

        dp_add_noise_to_gradients(
            gradients, C=1.0, sigma=self.sigma, batch_size=batch_size
        )

        if should_log:
            post_noise_norm = self._compute_grad_norm(gradients)
            num_params = sum(p.numel() for p in params_with_grad)
            log_info(
                "DP step %d: batch_size=%d, pre_clip=%.4f, "
                "post_clip=%.4f, post_noise=%.4f, sigma=%.4f, C=1.0, num_params=%d",
                self._step,
                batch_size,
                pre_norm,
                post_clip_norm,
                post_noise_norm,
                self.sigma,
                num_params,
            )
        self._step += 1


class DPSFTTrainer(SFTTrainer):
    """SFTTrainer with batch-level differential privacy gradient noise."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dp_epsilon: float = kwargs.pop("dp_epsilon")
        dp_delta: float = kwargs.pop("dp_delta")
        dp_sigma: float = compute_dp_sigma(dp_epsilon, dp_delta)
        super().__init__(*args, **kwargs)
        self.args.max_grad_norm = 0
        self.add_callback(_DPGradientCallback(sigma=dp_sigma))
        log_info("DP gradient callback added (sigma=%.4f)", dp_sigma)
