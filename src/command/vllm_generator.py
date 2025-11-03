import os
import sys
from collections.abc import Generator

src_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, src_path)

from cyy_huggingface_toolbox.inference import get_llm_engine as get_llm_engine_back
from cyy_torch_toolbox import Inferencer
from vllm import RequestOutput, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from distributed_learning_simulation import (
    Session,
)
from vllm import LLM

import src.method  # noqa: F401


def get_llm_engine(session: Session, finetuned_model_dir: str | None = None) -> LLM:
    model_name = session.config.model_config.model_name.removeprefix(
        "hugging_face_causal_lm_"
    )
    llm: LLM = get_llm_engine_back(
        pretrained_model_name_or_path=model_name,
        finetuned_model_dir=finetuned_model_dir,
        max_model_len=2048,
    )
    return llm


def get_vllm_output(
    tester: Inferencer, engine: LLM
) -> Generator[tuple[dict, RequestOutput]]:
    # Load the default sampling parameters from the model.
    sampling_params = SamplingParams(n=1, max_tokens=2048, temperature=0, min_tokens=10)

    for batch in tester.dataloader:
        batch_size = batch["batch_size"]
        batch_list: list[dict] = [{} for _ in range(batch_size)]
        for k, v in batch.items():
            if isinstance(v, list):
                for idx, a in enumerate(v):
                    batch_list[idx][k] = a
        yield from zip(
            batch_list,
            engine.generate(batch["inputs"], sampling_params),
            strict=False,
        )
