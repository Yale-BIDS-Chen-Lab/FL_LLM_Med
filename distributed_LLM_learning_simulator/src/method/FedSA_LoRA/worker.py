from cyy_naive_lib.log import log_debug, log_warning
from cyy_torch_toolbox import TensorDict, tensor_clone

from ..method_forward import (
    SFTTrainerWorker,
)


class FedSALoRAWorker(SFTTrainerWorker):
    old_state: None | TensorDict = None

    def _get_parameters(self) -> TensorDict:
        if not self._stopped():
            self.old_state = tensor_clone(super()._get_parameters())
            assert self.old_state is not None
            state = {k: v for k, v in self.old_state.items() if "lora_A" in k}
            assert state
            return state
        log_warning("Send full adaptor")
        assert self.old_state is not None
        return self.old_state

    def _after_training(self) -> None:
        self._in_after_training = True
        message = self._get_sent_data()
        message.end_training = True
        log_warning("Send final adaptor")
        self._aggregation(sent_data=message)
        sft_trainer = self.get_sft_trainer(self.trainer)
        sft_trainer.save_model()
        super()._after_training()

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        if self.old_state is None:
            super()._load_adaptor(adaptor_parameter)
            return

        adaptor_parameter = {
            k: v for k, v in adaptor_parameter.items() if "lora_A" in k
        }
        log_debug("Received layers:%s", list(adaptor_parameter.keys()))
        assert adaptor_parameter
        for k, v in adaptor_parameter.items():
            assert k in self.old_state
            self.old_state[k] = v
        super()._load_adaptor(self.old_state)
