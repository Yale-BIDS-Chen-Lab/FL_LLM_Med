import os
import sys
from collections.abc import Generator

from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_naive_lib.fs.tempdir import TempDir
from cyy_torch_toolbox import Inferencer
from distributed_learning_simulation import (
    Session,
    get_server,
)
from peft import PeftModel
from transformers import AutoModelForCausalLM
from vllm import LLM, RequestOutput, SamplingParams

os.environ["WANDB_DISABLED"] = "true"
os.environ["NO_TOKENIZER_TRANSFORMS"] = "true"

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, src_path)
import method  # noqa: F401
from server import FinetuneAdaptorServer


def get_vllm_output() -> Generator[tuple[dict, RequestOutput]]:
    session = Session()
    session.config.model_config.model_kwargs.pop("load_in_4bit", None)
    session.config.model_config.model_kwargs.pop("load_in_8bit", None)
    session.config.model_config.model_kwargs.pop("finetune_config", None)
    server = get_server(config=session.config)
    assert isinstance(server, FinetuneAdaptorServer)
    tester: Inferencer = server.get_tester(for_evaluation=True)
    model_evaluator = tester.model_evaluator
    assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)

    with TempDir():
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(session.session_dir, "SFTTrainer")
        )
        finetuned_model = PeftModel.from_pretrained(
            model=model,
            model_id=session.config.model_config.model_name.removeprefix(
                "hugging_face_causal_lm_"
            ),
        )
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
        )

        # Load the default sampling parameters from the model.
        sampling_params = SamplingParams(
            n=1, max_tokens=512, stop="<EOS>", temperature=0
        )

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
        return
