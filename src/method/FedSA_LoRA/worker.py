from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import TensorDict, tensor_clone

from ..method_forward import (
    SFTTrainerWorker,
)


class FedSALoRAWorker(SFTTrainerWorker):
    old_state: None | TensorDict = None

    def _get_parameters(self) -> TensorDict:
        self.old_state = tensor_clone(super()._get_parameters())
        assert self.old_state is not None
        state = {k: v for k, v in self.old_state.items() if "lora_A" in k}
        log_info("Upload layers:%s", list(state.keys()))
        assert state
        return state

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        if self._sft_trainer is None:
            super()._load_adaptor(adaptor_parameter)
            return

        assert self.old_state is not None
        adaptor_parameter = {
            k: v for k, v in adaptor_parameter.items() if "lora_A" in k
        }
        assert adaptor_parameter
        for k, v in adaptor_parameter.items():
            assert k in self.old_state
            self.old_state[k] = v
        super()._load_adaptor(self.old_state)
