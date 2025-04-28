from cyy_torch_toolbox import TensorDict

from ..method_forward import (
    SFTTrainerWorker,
)


class FedSALoRAWorker(SFTTrainerWorker):
    def _get_parameters(self) -> TensorDict:
        state = super()._get_parameters()
        state = {k: v for k, v in state.items() if "lora_A" in k}
        assert state
        return state
