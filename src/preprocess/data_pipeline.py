from typing import Any

from cyy_torch_toolbox.data_pipeline import DataPipeline, Transform


def format_input(sample: dict) -> dict:
    sample["input"] = "\n".join(
        [
            "### Input",
            " ".join(sample["tokens"]),
            "### Output",
            str(sample["annotated_phrases"]),
        ]
    )
    return sample


def get_iob_pipeline() -> DataPipeline:
    pipeline = DataPipeline()
    pipeline.append(Transform(fun=format_input, cacheable=True))
    return pipeline
