from distributed_learning_simulation import (
    AlgorithmRepository,
    FedAVGAlgorithm,
)

from .server import NERServer
from .worker import NERWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="adaptor_avg",
    client_cls=NERWorker,
    server_cls=NERServer,
    algorithm_cls=FedAVGAlgorithm,
)
