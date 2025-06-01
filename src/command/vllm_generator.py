import os
import sys
from collections.abc import Generator

src_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, src_path)

from cyy_naive_lib.fs.tempdir import TempDir
from vllm import LLM, RequestOutput, SamplingParams

os.environ["NO_TOKENIZER_TRANSFORMS"] = "true"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import src.method  # noqa: F401

from .util import get_tester, get_vllm_model


def get_vllm_output(
    session_dir: str, data_file: str, zero_shot: bool, worker_index: int | None = None
) -> Generator[tuple[dict, RequestOutput]]:
    tester = get_tester(session_dir=session_dir, data_file=data_file)

    with TempDir():
        # Create an LLM with built-in default generation config.
        # The generation config is set to None by default to keep
        # the behavior consistent with the previous version.
        # If you want to use the default generation config from the model,
        # you should set the generation_config to "auto".
        model_name = get_vllm_model(
            session_dir=session_dir, zero_shot=zero_shot, worker_index=worker_index
        )

        llm = LLM(
            model=model_name,
            generation_config="auto",
            tokenizer=model_name,
            dtype="bfloat16",
            max_model_len=2048,
        )
        llm.set_tokenizer(tester.model_evaluator.tokenizer)

        # Load the default sampling parameters from the model.
        sampling_params = SamplingParams(n=1, max_tokens=2048, temperature=0)

        for batch in tester.dataloader:
            # Generate texts from the prompts. The output is a list of RequestOutput objects
            # that contain the prompt, generated text, and other information.
            batch_size = batch["batch_size"]
            batch_list: list[dict] = [{} for _ in range(batch_size)]
            for k, v in batch.items():
                if isinstance(v, list):
                    for idx, a in enumerate(v):
                        batch_list[idx][k] = a
            yield from zip(
                batch_list,
                llm.generate(batch["inputs"], sampling_params),
                strict=False,
            )
