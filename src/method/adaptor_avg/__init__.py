from distributed_learning_simulation import (
    CentralizedAlgorithmFactory,
    FedAVGAlgorithm,
)

from .server import FinetuneAdaptorServer
from .worker import FinetuneAdaptorWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="adaptor_avg",
    client_cls=FinetuneAdaptorWorker,
    server_cls=FinetuneAdaptorServer,
    algorithm_cls=FedAVGAlgorithm,
)
