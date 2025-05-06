from collections.abc import Callable
from typing import Any

from distributed_learning_simulation.algorithm import FedAVGAlgorithm
from distributed_learning_simulation.message import ParameterMessage


class AggregationByLossAlgorithm(FedAVGAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fun: Callable | None = None

    def _get_weight(
        self, worker_data: ParameterMessage, name: str, parameter: Any
    ) -> Any:
        worker_data.aggregation_weight = self.loss_fun(worker_data=worker_data)
        return worker_data.aggregation_weight
