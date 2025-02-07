from typing import Any

from cyy_torch_toolbox.data_pipeline import DataPipeline, Transform


def format_input(sample: Any) -> str:
    return "\n".join(
        ["### Input", sample["text"], "### Output", str(sample["annotated_phrases"])]
    )


def get_iob_pipeline() -> DataPipeline:
    pipeline = DataPipeline()
    pipeline.append(Transform(fun=format_input, cacheable=True))
    return pipeline
