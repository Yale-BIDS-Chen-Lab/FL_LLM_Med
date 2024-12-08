import os
import sys

from cyy_torch_toolbox import Inferencer, TextDatasetCollection

server_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "server"
)
sys.path.append(server_path)
from server import LLMTextServer

from .data_pipeline import get_pipeline


class FinetuneAdaptorServer(LLMTextServer):
    def get_tester(self, **kwargs) -> Inferencer:
        inferencer = super().get_tester(**kwargs)
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        for transform in get_pipeline().transforms:
            inferencer.dataset_collection.append_text_transform(transform)
        return inferencer
