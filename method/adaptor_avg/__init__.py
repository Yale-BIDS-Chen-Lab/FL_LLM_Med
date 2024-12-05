import os
import sys

from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import TensorDict
from distributed_learning_simulation import (
    AggregationServer,
    CentralizedAlgorithmFactory,
    FedAVGAlgorithm,
)

worker_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "worker"
)
server_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "worker"
)
sys.path.append(worker_path)
sys.path.append(server_path)
from server.aggregation_server import LLMAggregationServer

from .worker import FinetuneAdaptorWorker

CentralizedAlgorithmFactory.register_algorithm(
    algorithm_name="adaptor_avg",
    client_cls=FinetuneAdaptorWorker,
    server_cls=LLMAggregationServer,
    algorithm_cls=FedAVGAlgorithm,
)
