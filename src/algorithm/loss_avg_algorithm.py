from collections.abc import Callable
from typing import Any

import torch
from distributed_learning_simulation.algorithm import FedAVGAlgorithm
from distributed_learning_simulation.message import ParameterMessage


class AggregationByLossAlgorithm(FedAVGAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fun: Callable | None = None

    def _get_weight(
        self, worker_data: ParameterMessage, name: str, parameter: Any
    ) -> Any:
        worker_data.aggregation_weight = torch.tensor(
            self.loss_fun(worker_data=worker_data), dtype=torch.float64
        ).exp()
        return worker_data.aggregation_weight
