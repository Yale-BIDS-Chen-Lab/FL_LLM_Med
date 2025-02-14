import os
from cyy_huggingface_toolbox import HuggingFaceModelEvaluatorForFinetune
from cyy_torch_toolbox import Inferencer, TextDatasetCollection

from ..method_forward import FinetuneAdaptorServer, get_iob_pipeline, LLMTextServer


class NERServer(FinetuneAdaptorServer, LLMTextServer):
    added_transform = False

    def get_tester(self) -> Inferencer:
        inferencer = super().get_tester()
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        if not self.added_transform:
            for transform in get_iob_pipeline().transforms:
                inferencer.dataset_collection.append_text_transform(transform)
            self.added_transform = True
        return inferencer

    def _server_exit(self) -> None:
        assert self.__model_cache.has_data
        self.__model_cache.save()
        # tester = self.get_tester()
        # # merge Rola layers
        # tester.replace_model(lambda old_model: old_model.merge_and_unload())
        # model_evaluator = tester.model_evaluator
        # assert isinstance(model_evaluator, HuggingFaceModelEvaluatorForFinetune)
        # model_evaluator.save_pretrained(os.path.join(self.save_dir, "finetuned_model"))
