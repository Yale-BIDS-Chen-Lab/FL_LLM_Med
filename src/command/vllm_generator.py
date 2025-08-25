import os
import sys
from collections.abc import Generator

src_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, src_path)

from cyy_huggingface_toolbox.inference import get_llm_engine
from cyy_torch_toolbox import Inferencer
from vllm import RequestOutput, SamplingParams

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import src.method  # noqa: F401
from distributed_learning_simulation import (
    Session,
)
from vllm import LLM


def get_vllm_output(
    tester: Inferencer,
    session: Session,
    zero_shot: bool,
    worker_index: int | None = None,
) -> Generator[tuple[dict, RequestOutput]]:
    model_name = session.config.model_config.model_name.removeprefix(
        "hugging_face_causal_lm_"
    )
    save_dir = None
    if not zero_shot:
        if worker_index is not None:
            assert worker_index < session.config.worker_number
            save_dir = os.path.join(
                session.session_dir, f"worker_{worker_index}", "SFTTrainer"
            )
        else:
            save_dir = os.path.join(session.server_dir, "SFTTrainer")
    llm: LLM = get_llm_engine(model_name, save_dir, max_model_len=2048)
    # tester.model_evaluator.tokenizer.padding_side = "left"
    # llm.set_tokenizer(tester.model_evaluator.tokenizer)

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
