import os
import sys
from collections.abc import Generator

src_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, src_path)

from cyy_naive_lib.fs.tempdir import TempDir
from cyy_torch_toolbox import Inferencer
from vllm import RequestOutput, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import src.method  # noqa: F401
from distributed_learning_simulation import (
    Session,
)
from util import get_vllm_model
from vllm import LLM


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


def get_vllm_output(
    tester: Inferencer,
    session: Session,
    zero_shot: bool,
    worker_index: int | None = None,
) -> Generator[tuple[dict, RequestOutput]]:
    with TempDir():
        model_name = get_vllm_model(
            session=session, zero_shot=zero_shot, worker_index=worker_index
        )
        llm = get_vllm(tester=tester, model_name=model_name)

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
