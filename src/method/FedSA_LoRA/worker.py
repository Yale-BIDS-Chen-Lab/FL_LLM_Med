from ..method_forward import (
    SFTTrainerWorker,
)
from cyy_torch_toolbox import TensorDict


class FedSALoRAWorker(SFTTrainerWorker):
    def _get_parameters(self) -> TensorDict:
        state = super()._get_parameters()
        return state
