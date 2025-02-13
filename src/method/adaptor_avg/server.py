import os
from typing import Any

from cyy_torch_toolbox import Inferencer, TextDatasetCollection

from ..method_forward import FinetuneAdaptorServer, get_iob_pipeline


class NERServer(FinetuneAdaptorServer):
    added_transform = False

    def get_tester(self, *args: Any, **kwargs: Any) -> Inferencer:
        inferencer = super().get_tester(*args, **kwargs)
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        if not self.added_transform:
            for transform in get_iob_pipeline().transforms:
                inferencer.dataset_collection.append_text_transform(transform)
            self.added_transform = True
        return inferencer

    def _server_exit(self) -> None:
        tester = self.get_tester()
        # merge Rola layers
        tester.replace_model(lambda old_model: old_model.merge_and_unload())
        tester.model_evaluator.save_pretrained(
            os.path.join(self.save_dir, "finetuned_model")
        )
