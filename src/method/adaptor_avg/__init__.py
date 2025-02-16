from distributed_learning_simulation import (
    AlgorithmRepository,
    FedAVGAlgorithm,
)

from ..method_forward import FinetuneAdaptorServer, SFTTrainerWorker

AlgorithmRepository.register_algorithm(
    algorithm_name="adaptor_avg",
    client_cls=SFTTrainerWorker,
    server_cls=FinetuneAdaptorServer,
    algorithm_cls=FedAVGAlgorithm,
)
