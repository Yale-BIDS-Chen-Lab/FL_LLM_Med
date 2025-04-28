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

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        state = super()._get_parameters()
        for k, v in adaptor_parameter.items():
            assert k in state
            state[k] = v
        super()._load_adaptor(state)
