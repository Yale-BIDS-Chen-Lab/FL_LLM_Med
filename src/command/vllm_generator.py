import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["NO_TOKENIZER_TRANSFORMS"] = "true"
import sys

from cyy_torch_toolbox import Inferencer
from distributed_learning_simulation import get_server, load_config
from vllm import LLM, RequestOutput

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "conf")
src_path = os.path.join(config_path, "..", "src")
sys.path.insert(0, src_path)
import method  # noqa: F401

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "conf")


def get_vllm_output() -> list[tuple[dict, RequestOutput]]:
    config = load_config(
        config_path=config_path,
        global_conf_path=os.path.join(config_path, "global.yaml"),
    )
    server = get_server(config=config)
    tester: Inferencer = server.get_tester()

    # Create an LLM with built-in default generation config.
    # The generation config is set to None by default to keep
    # the behavior consistent with the previous version.
    # If you want to use the default generation config from the model,
    # you should set the generation_config to "auto".
    model_dir = os.getenv("MODEL_DIR", None)
    assert model_dir is not None

    llm = LLM(model=model_dir, generation_config="auto")

    # Load the default sampling parameters from the model.
    sampling_params = llm.get_default_sampling_params()
    # # Modify the sampling parameters if needed.
    # sampling_params.temperature = 0.5

    # assert isinstance(tester.dataset_collection, TextDatasetCollection)
    # print(tester.dataset_collection.get_text_pipeline())
    # tester.dataset_collection.get_text_pipeline
    result: list[tuple[dict, RequestOutput]] = []
    for batch in tester.dataloader:
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        batch_size = batch["batch_size"]
        batch_list: list[dict] = [{} for _ in range(batch_size)]
        for k, v in batch.items():
            if isinstance(v, list):
                if len(v) != batch_size:
                    continue
                for idx, a in enumerate(v):
                    batch_list[idx][k] = a

        result += list(
            zip(
                batch_list, llm.generate(batch["inputs"], sampling_params), strict=False
            )
        )
    return result
