import copy
import os
import sys
from collections.abc import Generator

src_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, src_path)

from cyy_naive_lib.fs.tempdir import TempDir
from cyy_torch_toolbox import Inferencer, load_local_files
from distributed_learning_simulation import (
    Session,
    get_server,
)
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM
from vllm import LLM, RequestOutput, SamplingParams

os.environ["NO_TOKENIZER_TRANSFORMS"] = "true"

import src.method  # noqa: F401
from src.server import LLMTextServer


def get_vllm_output(
    session_dir: str | None = None,
    data_file: str | None = None,
) -> Generator[tuple[dict, RequestOutput]]:
    if session_dir is not None:
        assert os.path.isdir(session_dir), session_dir
    if data_file is not None:
        assert os.path.isfile(data_file), data_file
    session = Session(session_dir=session_dir)
    config = copy.deepcopy(session.config)
    if "train_files" in config.dc_config.dataset_kwargs:
        for f in config.dc_config.dataset_kwargs["train_files"]:
            assert os.path.isfile(f), f
    if "test_files" in config.dc_config.dataset_kwargs:
        for f in config.dc_config.dataset_kwargs["test_files"]:
            assert os.path.isfile(f), f
    config.hyper_parameter_config.batch_size = 1024

    config.apply_global_config()
    server = get_server(config=config)
    assert isinstance(server, LLMTextServer)
    tester: Inferencer = server.get_tester(for_evaluation=True)
    if data_file is not None:
        tester.mutable_dataset_collection.transform_all_datasets(
            transformer=lambda _: load_local_files([data_file]),
        )

    with TempDir():
        save_dir = os.path.join(session.server_dir, "SFTTrainer")
        model_name = session.config.model_config.model_name.removeprefix(
            "hugging_face_causal_lm_"
        )
        model = AutoModelForCausalLM.from_pretrained(model_name)
        finetuned_model = PeftModel.from_pretrained(model=model, model_id=save_dir)
        merge_model = finetuned_model.merge_and_unload()
        merge_model.save_pretrained("./finetuned_model")

        # Create an LLM with built-in default generation config.
        # The generation config is set to None by default to keep
        # the behavior consistent with the previous version.
        # If you want to use the default generation config from the model,
        # you should set the generation_config to "auto".

        llm = LLM(
            model="./finetuned_model",
            generation_config="auto",
            tokenizer=model_name,
            dtype="bfloat16",
            max_model_len=2048,
        )
        tester.model_evaluator.tokenizer.padding_side = "left"
        llm.set_tokenizer(tester.model_evaluator.tokenizer)

        # Load the default sampling parameters from the model.
        sampling_params = SamplingParams(n=1, max_tokens=2048, temperature=0)

        for batch in tester.dataloader:
            # Generate texts from the prompts. The output is a list of RequestOutput objects
            # that contain the prompt, generated text, and other information.
            batch_size = batch["batch_size"]
            batch_list: list[dict] = [
                {}
                for _ in range(batch_size)
            ]
            for k, v in batch.items():
                if isinstance(v, list):
                    for idx, a in enumerate(v):
                        batch_list[idx][k] = a
            yield from zip(
                batch_list,
                llm.generate(batch["inputs"], sampling_params),
                strict=False,
            )
