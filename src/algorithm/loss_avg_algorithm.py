from collections.abc import Callable

import torch
from cyy_torch_toolbox import ModelParameter
from distributed_learning_simulation.algorithm import FedAVGAlgorithm
from distributed_learning_simulation.message import ParameterMessage


class AggregationByLossAlgorithm(FedAVGAlgorithm):
    def __init__(self) -> None:
        super().__init__()
        self._worker_loss: dict[int, torch.Tensor] = {}
        self.__parameter: dict[str, list[tuple[int, torch.Tensor]]] = {}
        self._total_loss: torch.Tensor | None = None
        self.loss_fun: Callable | None = None

    def _accumulate_parameter(
        self,
        worker_id: int,
        worker_data: ParameterMessage,
        name: str,
        parameter: torch.Tensor,
    ) -> None:
        assert self.loss_fun is not None
        if worker_id not in self._worker_loss:
            self._worker_loss[worker_id] = torch.tensor(
                self.loss_fun(worker_data=worker_data), dtype=torch.float64
            ).exp()
        data = (worker_id, parameter)
        if name not in self.__parameter:
            self.__parameter[name] = [data]
        else:
            self.__parameter[name].append(data)

    def _aggregate_parameter(self) -> ModelParameter:
        if self._total_loss is None:
            assert len(self._worker_loss) > 1
            all_losses = list(self._worker_loss.values())
            self._total_loss = sum(all_losses[1:], start=all_losses[0])
        result: ModelParameter = {}
        for name, d in self.__parameter.items():
            parameter_sum = None
            for worker_id, parameter in d:
                factor = (
                    parameter.to(dtype=torch.float64) * self._worker_loss[worker_id]
                )
                if parameter_sum is None:
                    parameter_sum = factor
                else:
                    parameter_sum += factor
            assert parameter_sum
            result[name] = parameter_sum / self._total_loss
        return result
