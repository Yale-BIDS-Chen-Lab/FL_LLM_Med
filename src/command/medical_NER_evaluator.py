import json
import os

os.environ["WANDB_DISABLED"] = "true"


from nervaluate import Evaluator
from vllm_generator import get_vllm_output


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


def get_NER_metric() -> tuple:
    prediction = []
    ground_tags = []
    tag_set = set()
    for sample, generated_text in get_vllm_output():
        tags = sample["tags"]
        for tag in tags:
            if tag != "O":
                tag_set.add(tag[2:])
        tokens = sample["tokens"]
        prediction_tags = ["O"] * len(tags)
        for phrase, tag in parse_prediction(generated_text.outputs[0].text):
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
    return Evaluator(
        ground_tags, prediction, tags=list(tag_set), loader="list"
    ).evaluate()


if __name__ == "__main__":
    results, results_per_tag, result_indices, result_indices_by_tag = get_NER_metric()
    print("NER test results ", results)
    print("NER test results_per_tag ", results_per_tag)
