from cyy_torch_toolbox.data_pipeline import DataPipeline, Transform


def format_input(sample) -> str:
    return "\n".join(
        ["### Input", sample["input"], "### Output", str(sample["output"])]
    )


def get_pipeline() -> DataPipeline:
    pipeline = DataPipeline()
    pipeline.append(Transform(fun=format_input, cacheable=True))
    return pipeline
