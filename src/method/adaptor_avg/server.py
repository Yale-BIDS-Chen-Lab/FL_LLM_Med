from typing import Any

from cyy_torch_toolbox import Inferencer, TextDatasetCollection

from ..method_forward import FinetuneAdaptorServer, get_iob_pipeline


class NERServer(FinetuneAdaptorServer):
    def get_tester(self, *args: Any, **kwargs: Any) -> Inferencer:
        inferencer = super().get_tester(*args, **kwargs)
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        for transform in get_iob_pipeline().transforms:
            inferencer.dataset_collection.append_text_transform(transform)
        return inferencer
