import os
import sys
from collections.abc import Generator

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
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

os.environ["WANDB_DISABLED"] = "true"
os.environ["NO_TOKENIZER_TRANSFORMS"] = "true"

import method  # noqa: F401
from server import LLMTextServer


def get_vllm_output(
    session_dir: str | None = None,
    data_file: str | None = None,
) -> Generator[tuple[dict, RequestOutput]]:
    session = Session(session_dir=session_dir)
    server = get_server(config=session.config)
    assert isinstance(server, LLMTextServer)
    tester: Inferencer = server.get_tester(for_evaluation=True)
    if data_file is not None:
        print("use test data", data_file)
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
            model="./finetuned_model", generation_config="auto", tokenizer=model_name
        )

        # Load the default sampling parameters from the model.
        sampling_params = SamplingParams(
            n=1, max_tokens=512, stop="<EOS>", temperature=0
        )

        for batch in tester.dataloader:
            # Generate texts from the prompts. The output is a list of RequestOutput objects
            # that contain the prompt, generated text, and other information.
            batch_size = batch["batch_size"]
            batch_list: list[dict] = [
                {"tokenizer": tester.model_evaluator.tokenizer.tokenizer}
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
        return
