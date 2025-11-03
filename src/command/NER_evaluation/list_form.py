import json
from collections.abc import Generator

from .common import find_tag


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
        if not a:
            continue
        new_res.append((a, b))
    return new_res


def get_NER_metric(output_generator: Generator) -> tuple:
    prediction = []
    ground_tags = []
    tag_set = set()
    for sample, generated_text in output_generator:
        tags = sample["tags"]
        for tag in tags:
            if tag != "O":
                tag_set.add(tag[2:])
        tokens = sample["tokens"]
        predicated_tokens = []
        predicated_tags = []
        predicated_candidate_tags = []
        out_text = generated_text.outputs[0].text
        for phrase, tag in parse_prediction(out_text):
            phrase_tokens = phrase.split(" ")
            predicated_tokens += phrase_tokens
            predicated_candidate_tags += [tag] * len(phrase_tokens)
        predicated_tokens_lower = [a.lower() for a in predicated_tokens]
        for token in tokens:
            predicated_tags.append(
                find_tag(
                    token,
                    predicated_tokens,
                    predicated_tokens_lower,
                    predicated_candidate_tags,
                )
            )
        if len(set(tags)) > 1:
            print(tags)
            print(predicated_tags)
            print("print tokens", sample["tokens"])
            print("print input", generated_text.prompt)
            print("print output", generated_text.outputs[0].text)
            print(out_text)

        prediction.append(predicated_tags)
        ground_tags.append(tags)
    return ground_tags, prediction, tag_set
