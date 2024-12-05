import os
import sys

from distributed_learning_simulation import (
    CentralizedAlgorithmFactory,
    FedAVGAlgorithm,
)

from .worker import FinetuneAdaptorWorker

server_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "worker"
)
sys.path.append(server_path)
from server.aggregation_server import LLMAggregationServer

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="adaptor_avg",
    client_cls=FinetuneAdaptorWorker,
    server_cls=LLMAggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
