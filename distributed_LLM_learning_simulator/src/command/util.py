import copy
import os

from cyy_torch_toolbox import Inferencer, load_local_files
from distributed_learning_simulation import (
    Session,
    get_server,
)


def get_tester(
    session: Session, data_file: str, for_language_modeling: bool = False
) -> tuple[Inferencer, set]:
    assert os.path.isfile(data_file), data_file
    config = copy.deepcopy(session.config)
    if for_language_modeling:
        if "train_files" in config.dc_config.dataset_kwargs:
            config.dc_config.dataset_kwargs["train_files"] = [data_file]
        if "test_files" in config.dc_config.dataset_kwargs:
            config.dc_config.dataset_kwargs["test_files"] = [data_file]
    else:
        if "train_files" in config.dc_config.dataset_kwargs:
            for f in config.dc_config.dataset_kwargs["train_files"]:
                assert os.path.isfile(f), f
        if "test_files" in config.dc_config.dataset_kwargs:
            for f in config.dc_config.dataset_kwargs["test_files"]:
                assert os.path.isfile(f), f
    print(config.dc_config.dataset_kwargs)
    print(config.model_config.model_kwargs)
    print(config.trainer_config.hook_config)
    if for_language_modeling:
        config.hyper_parameter_config.batch_size = 1024

    server = get_server(config=config)
    if for_language_modeling:
        tester = server.get_tester(for_evaluation=True)
    else:
        tester: Inferencer = server.get_tester()

    tester.model_evaluator.tokenizer.padding_side = "left"
    old_labels = set(
        copy.deepcopy(tester.dataset_collection.get_labels(use_cache=False))
    )
    if not for_language_modeling:
        print("labels is ", len(old_labels))
        tester.mutable_dataset_collection.transform_all_datasets(
            transformer=lambda _: load_local_files([data_file]),
        )
        if hasattr(tester.model_evaluator.model, "labels"):
            tester.model_evaluator.model.labels = old_labels

    return tester, old_labels
