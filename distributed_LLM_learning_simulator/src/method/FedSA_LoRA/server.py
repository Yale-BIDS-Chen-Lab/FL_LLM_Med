from cyy_naive_lib.log import log_warning

from ..method_forward import (
    SFTServer,
)


class FedSALoRAServer(SFTServer):
    def _stopped(self) -> bool:
        return self._stop

    def _server_exit(self) -> None:
        log_warning("Store final session")
        super()._server_exit()
