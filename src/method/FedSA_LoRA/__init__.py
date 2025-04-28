from distributed_learning_simulation import (
    AlgorithmRepository,
    FedAVGAlgorithm,
)
from .worker import FedSALoRAWorker

from ..method_forward import (
    SFTServer,
    SFTTrainerWorker,
)

AlgorithmRepository.register_algorithm(
    algorithm_name="FedSA-LoRA",
    client_cls=FedSALoRAWorker,
    server_cls=SFTServer,
    algorithm_cls=FedAVGAlgorithm,
)
