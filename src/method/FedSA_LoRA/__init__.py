from distributed_learning_simulation import (
    AlgorithmRepository,
    FedAVGAlgorithm,
)

from .server import FedSALoRAServer
from .worker import FedSALoRAWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="FedSA-LoRA",
    client_cls=FedSALoRAWorker,
    server_cls=FedSALoRAServer,
    algorithm_cls=FedAVGAlgorithm,
)
