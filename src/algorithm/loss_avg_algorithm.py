from collections.abc import Callable
from typing import Any

from distributed_learning_simulation.algorithm import FedAVGAlgorithm
from distributed_learning_simulation.message import Message, ParameterMessage


class AggregationByLossAlgorithm(FedAVGAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fun: Callable | None = None

    def process_worker_data(
        self,
        worker_id: int,
        worker_data: Message | None,
    ) -> bool:
        res = super().process_worker_data(worker_id=worker_id, worker_data=worker_data)
        self._all_worker_data[worker_id].aggregation_weight = None
        return res

    def _get_weight(
        self, worker_data: ParameterMessage, name: str, parameter: Any
    ) -> Any:
        if worker_data.aggregation_weight is None:
            worker_data.aggregation_weight = self.loss_fun(worker_data=worker_data)
        return worker_data.aggregation_weight
