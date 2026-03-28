from typing import Any

from cyy_naive_lib.log import log_info
from distributed_learning_simulation.dp import (
    compute_dp_sigma,
    dp_clip_and_noise_gradients,
)
from transformers import TrainerCallback
from trl.trainer.sft_trainer import SFTTrainer


class _DPGradientCallback(TrainerCallback):
    """Applies DP gradient clipping and noise once per optimizer step."""

    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        dp_clip_and_noise_gradients(
            parameters=list(kwargs["model"].parameters()),
            C=1.0,
            sigma=self.sigma,
        )


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
