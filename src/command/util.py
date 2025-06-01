import copy
import dill
import os

from cyy_torch_toolbox import Inferencer, load_local_files
from distributed_learning_simulation import (
    Session,
    get_server,
)
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM
from vllm import LLM


def get_tester(session_dir: str, data_file: str) -> Inferencer:
    assert os.path.isdir(session_dir), session_dir
    assert os.path.isfile(data_file), data_file
    session = Session(session_dir=session_dir)
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
    config.hyper_parameter_config.batch_size = 1024

    server = get_server(config=config)
    tester: Inferencer = server.get_tester(for_evaluation=True)
    tester.mutable_dataset_collection.transform_all_datasets(
        transformer=lambda _: load_local_files([data_file]),
    )
    tester.model_evaluator.tokenizer.padding_side = "left"
    return tester


def get_model(
    tester: Inferencer,
    session_dir: str,
    zero_shot: bool,
    worker_index: int | None = None,
) -> None:
    assert os.path.isdir(session_dir), session_dir
    session = Session(session_dir=session_dir)

    if not zero_shot:
        assert worker_index is None
        with open(session.last_model_path, "rb") as f:
            parameters = dill.load(f)
            tester.model_util.load_parameters(parameters)


def get_vllm_model(
    session_dir: str, zero_shot: bool, worker_index: int | None = None
) -> str:
    assert os.path.isdir(session_dir), session_dir
    session = Session(session_dir=session_dir)

    model_name = session.config.model_config.model_name.removeprefix(
        "hugging_face_causal_lm_"
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if not zero_shot:
        if worker_index is not None:
            assert worker_index < session.config.worker_number
            save_dir = os.path.join(
                session.session_dir, f"worker_{worker_index}", "SFTTrainer"
            )
        else:
            save_dir = os.path.join(session.server_dir, "SFTTrainer")
        finetuned_model = PeftModel.from_pretrained(model=model, model_id=save_dir)
        model = finetuned_model.merge_and_unload()
    model.save_pretrained("./finetuned_model")
    return "./finetuned_model"


def get_vllm(model_name: str, tester: Inferencer) -> LLM:
    llm = LLM(
        model=model_name,
        generation_config="auto",
        tokenizer=model_name,
        dtype="bfloat16",
        max_model_len=2048,
    )
    llm.set_tokenizer(tester.model_evaluator.tokenizer)
    return llm
