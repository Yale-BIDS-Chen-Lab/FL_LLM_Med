from typing import Any
from ner_metrics import classification_report
import json

from cyy_torch_toolbox import Inferencer, TextDatasetCollection

from ..method_forward import FinetuneAdaptorServer, get_iob_pipeline


def parse_prediction(prediction: str) -> list[tuple[str, str, int]]:
    res: list[tuple[str, str, int]] = []
    prediction = (
        prediction.replace("['", '["')
        .replace("']", '"]')
        .replace("', '", '", "')
        .replace("\", '", '", "')
        .replace("[[", "[")
        .replace("]]", "]")
    )
    while prediction:
        idx = prediction.find('["')
        if not idx >= 0:
            break
        prediction = prediction[idx:]
        end_idx = prediction.find('"]')
        if not end_idx >= 0:
            break
        item = prediction[: end_idx + 2]
        # print("item is", item)
        try:
            predicated_pair = json.loads(item)
            if len(predicated_pair) != 3:
                # print("invalid item", predicated_pair)
                continue
            res.append(predicated_pair)
        except Exception:
            break
        prediction = prediction[end_idx + 2 :]

    new_res = []
    for a, b, c in res:
        if not isinstance(a, str) or not isinstance(b, str) or not isinstance(c, int):
            continue
        new_res.append((a, b, c))
    return new_res


def get_NER_metric(tester: Inferencer):
    generated_texts = tester.get_sample_output(generated_kwargs={})
    print(generated_texts)
    prediction = []
    ground_tags = []
    for sample_index, generated_text in generated_texts.items():
        sample = tester.dataset_util.get_sample(sample_index)
        for pair in parse_prediction(generated_text):
            prediction.append(f"{pair[0]} B-{pair[1]}")
        for pair in sample["annotated_phrases"]:
            ground_tags.append(f"{pair[0]} B-{pair[1]}")
    print(ground_tags)
    print(prediction)
    lenient = classification_report(
        tags_true=ground_tags, tags_pred=prediction, mode="lenient"
    )
    strict = classification_report(
        tags_true=ground_tags, tags_pred=prediction, mode="strict"
    )
    scores = []
    for entity in strict:
        strict_scores = strict[entity]
        lenient_scores = lenient[entity]
        scores.append(
            {
                "entity": entity,
                "strict_precision": f"{strict_scores['precision']}",
                "strict_recall": f"{strict_scores['recall']}",
                "strict_f1-score": f"{strict_scores['f1-score']}",
                "lenient_precision": f"{lenient_scores['precision']}",
                "lenient_recall": f"{lenient_scores['recall']}",
                "lenient_f1-score": f"{lenient_scores['f1-score']}",
            }
        )
    print(scores)


class NERServer(FinetuneAdaptorServer):
    def get_tester(self, *args: Any, **kwargs: Any) -> Inferencer:
        inferencer = super().get_tester(*args, **kwargs)
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        for transform in get_iob_pipeline().transforms:
            inferencer.dataset_collection.append_text_transform(transform)
        return inferencer

    def _get_metric(self, tester: Inferencer) -> Any:
        metric = super()._get_metric(tester=tester)
        assert isinstance(metric, dict)
        get_NER_metric(tester)
        return metric
