from cyy_naive_lib.log import log_info
from ..method_forward import FinetuneAdaptorWorker, get_iob_pipeline


class NERWorker(FinetuneAdaptorWorker):
    def _before_training(self) -> None:
        super()._before_training()
        for transform in get_iob_pipeline().transforms:
            self.dataset_collection.append_text_transform(transform)
        if self.hold_log_lock:
            log_info("model is %s", self.trainer.model)
