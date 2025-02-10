import json
from typing import Any

from cyy_naive_lib.log import log_info
from cyy_torch_toolbox import Inferencer, TextDatasetCollection
from nervaluate import Evaluator

from ..method_forward import FinetuneAdaptorServer, get_iob_pipeline


def parse_prediction(prediction: str) -> list[tuple[str, str]]:
    res: list[tuple[str, str]] = []
    prediction = (
        prediction.replace("['", '["')
        .replace("']", '"]')
        .replace("', '", '", "')
        .replace("','", '","')
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
        prediction = prediction[end_idx + 2 :]
        try:
            predicated_pair = json.loads(item)
            if len(predicated_pair) != 2:
                continue
            res.append(predicated_pair)
        except Exception:
            break

    new_res = []
    for a, b in res:
        if not isinstance(a, str) or not isinstance(b, str):
            continue
        new_res.append((a, b))
    return new_res


def get_NER_metric(tester: Inferencer):
    # merge Rola layers
    tester.replace_model(lambda old_model: old_model.merge_and_unload())

    max_new_tokens = 0
    for _, sample in tester.dataset_util.get_raw_samples():
        sample_token = 0
        for phrase in sample["data"]["annotated_phrases"]:
            sample_token += len(phrase[0].split(" ")) * 2 + 10
        max_new_tokens = max(max_new_tokens, sample_token)

    generated_texts = tester.get_sample_output(
        max_new_tokens=max_new_tokens, do_sample=False
    )
    prediction = []
    ground_tags = []
    tag_set = set()
    for sample_index, generated_text in generated_texts.items():
        sample = tester.dataset_util.get_sample(sample_index)
        tags = sample["tags"]
        for tag in tags:
            if tag != "O":
                tag_set.add(tag[2:])
        tokens = sample["tokens"]
        prediction_tags = ["O"] * len(tags)
        for phrase, tag in parse_prediction(generated_text):
            if not phrase:
                continue
            sub_tokens = phrase.split(" ")
            for i in range(len(tokens)):
                if (
                    tokens[i] == sub_tokens[0]
                    and tokens[i : i + len(sub_tokens)] == sub_tokens
                ):
                    prediction_tags[i] = f"B-{tag}"
                    for j in range(i + 1, len(sub_tokens)):
                        prediction_tags[j] = f"I-{tag}"

            # print(phrase, tag)
            # print(sample["annotated_phrases"])
            # idx = sample["annotated_phrases"].index([phrase, tag])
            # if idx >= 0:
            #     token_location = sample["annotated_phrase_locations"][idx]
            #     token_num = len(phrase.split(" "))
            #     for i in range(token_num):
            #         assert tags[token_location + i] != "O"
            #         prediction_tags[token_location + i] = (
            #             tags[token_location + i][:2] + tag
            #         )

        prediction.append(prediction_tags)
        ground_tags.append(tags)
    results, results_per_tag, result_indices, result_indices_by_tag = Evaluator(
        ground_tags, prediction, tags=list(tag_set), loader="list"
    ).evaluate()
    return results


class NERServer(FinetuneAdaptorServer):
    def get_tester(self, *args: Any, **kwargs: Any) -> Inferencer:
        inferencer = super().get_tester(*args, **kwargs)
        assert isinstance(inferencer.dataset_collection, TextDatasetCollection)
        for transform in get_iob_pipeline().transforms:
            inferencer.dataset_collection.append_text_transform(transform)
        return inferencer

    # def _get_metric(self, tester: Inferencer) -> Any:
    #     metric = super()._get_metric(tester=tester)
    #     assert isinstance(metric, dict)
    #     self._round_index += 1
    #     if self._stopped():
    #         results = get_NER_metric(tester)
    #         log_info("round: %s, NER test result %s", self.round_index, results)
    #     self._round_index -= 1
    #     return metric
