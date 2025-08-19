import contextlib
import copy
import os

import dill
from cyy_torch_toolbox import Inferencer, load_local_files
from distributed_learning_simulation import (
    Session,
    get_server,
)


def get_tester(session: Session, data_file: str) -> tuple[Inferencer, set]:
    assert os.path.isfile(data_file), data_file
    config = copy.deepcopy(session.config)
    if "train_files" in config.dc_config.dataset_kwargs:
        for f in config.dc_config.dataset_kwargs["train_files"]:
            assert os.path.isfile(f), f
    if "test_files" in config.dc_config.dataset_kwargs:
        for f in config.dc_config.dataset_kwargs["test_files"]:
            assert os.path.isfile(f), f
    print(config.dc_config.dataset_kwargs)
    print(config.model_config.model_kwargs)
    print(config.trainer_config.hook_config)
    if config.model_config.model_name.startswith("hugging_face_causal_lm_"):
        config.hyper_parameter_config.batch_size = 1024

    server = get_server(config=config)
    tester: Inferencer = server.get_tester()
    with contextlib.suppress(BaseException):
        tester = server.get_tester(for_evaluation=True)

    old_labels = set(
        copy.deepcopy(tester.dataset_collection.get_labels(use_cache=False))
    )
    print("labels is ", len(old_labels))
    tester.mutable_dataset_collection.transform_all_datasets(
        transformer=lambda _: load_local_files([data_file]),
    )
    tester.model_evaluator.tokenizer.padding_side = "left"
    if hasattr(tester.model_evaluator.model, "labels"):
        tester.model_evaluator.model.labels = old_labels

    return tester, old_labels


def get_model(
    tester: Inferencer,
    session: Session,
    zero_shot: bool,
    worker_index: int | None = None,
) -> None:
    if not zero_shot:
        assert worker_index is None
        with open(session.last_model_path, "rb") as f:
            parameters = dill.load(f)
            tester.model_util.load_parameters(parameters)
