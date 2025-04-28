from distributed_learning_simulation import (
    AlgorithmRepository,
    FedAVGAlgorithm,
)

from ..method_forward import (
    FinetuneAdaptorServer,
    FinetuneAdaptorWorker,
    SFTServer,
    SFTTrainerWorker,
)

AlgorithmRepository.register_algorithm(
    algorithm_name="adaptor_avg",
    client_cls=SFTTrainerWorker,
    server_cls=SFTServer,
    algorithm_cls=FedAVGAlgorithm,
)
