from cyy_naive_lib.log import log_debug, log_warning
from cyy_preprocessing_pipeline import tensor_clone
from cyy_torch_toolbox import TensorDict

from ..method_forward import SFTTrainerWorker


class FedSALoRAWorker(SFTTrainerWorker):
    old_state: None | TensorDict = None

    def _get_parameters(self) -> TensorDict:
        self.old_state = tensor_clone(super()._get_parameters())
        if not self._stopped():
            state = {k: v for k, v in self.old_state.items() if "lora_A" in k}
            assert state
            return state
        log_warning("Send full adaptor")
        assert self.old_state is not None
        return self.old_state

    def _after_training(self) -> None:
        super()._after_training()
        message = self._get_sent_data()
        message.end_training = True
        self._aggregation(sent_data=message)
        self.get_sft_trainer(self.trainer).save_model()

    def _load_adaptor(self, adaptor_parameter: TensorDict) -> None:
        if self._stopped() or self.old_state is None:
            super()._load_adaptor(adaptor_parameter)
            return

        lora_a_params = {k: v for k, v in adaptor_parameter.items() if "lora_A" in k}
        log_debug("Received layers:%s", list(lora_a_params.keys()))
        assert lora_a_params
        self.old_state.update(lora_a_params)
        super()._load_adaptor(self.old_state)
