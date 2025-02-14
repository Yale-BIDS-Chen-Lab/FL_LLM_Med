


class NERWorker(FinetuneAdaptorWorker):
    def _before_training(self) -> None:
        super()._before_training()
