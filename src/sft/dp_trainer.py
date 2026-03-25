from typing import Any

import torch
from distributed_learning_simulation.dp import (
    compute_dp_sigma,
    dp_clip_and_noise_gradients,
)
from trl.trainer.sft_trainer import SFTTrainer


class DPSFTTrainer(SFTTrainer):
    """SFTTrainer with batch-level differential privacy gradient noise."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        dp_epsilon: float = kwargs.pop("dp_epsilon")
        dp_delta: float = kwargs.pop("dp_delta")
        self.dp_sigma: float = kwargs.pop(
            "dp_sigma", compute_dp_sigma(dp_epsilon, dp_delta)
        )
        super().__init__(*args, **kwargs)

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        loss = super().training_step(
            model=model,
            inputs=inputs,
            num_items_in_batch=num_items_in_batch,
        )
        dp_clip_and_noise_gradients(
            parameters=list(model.parameters()),
            sigma=self.dp_sigma,
        )
        return loss
