import argparse
import json
import logging

import nervaluate
from cyy_naive_lib.log import set_level
from medical_NER_evaluation.common import find_tag
from medical_NER_evaluation.html_form import html2bio
from ner_metrics import classification_report
from vllm_generator import get_vllm_output


def flatten_extend(l: list) -> list:
    flat_list = []
    for e in l:
        flat_list.extend(e)
    return flat_list


if __name__ == "__main__":
    set_level(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="Analyze NER result",
    )
    parser.add_argument("--session_dir", help="session dir", type=str, required=True)
    parser.add_argument("--test_file", help="test file", type=str, default=None)
    args = parser.parse_args()

    prediction = []
    ground_tags = []
    entities = ["problem", "treatment", "test", "drug"]

    for sample, generated_text in list(
        get_vllm_output(session_dir=args.session_dir, data_file=args.test_file)
    ):
        tags = sample["tags"]
        out_text = generated_text.outputs[0].text
        tokenizer = sample["tokenizer"]
        tokens = sample["tokens"]
        predicated_tags = []
        predicated_candidate_tags = []
        predicated_tokens, predicated_candidate_tags = html2bio(
            html=out_text, entities=entities, tokenizer=tokenizer
        )
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
        # if len(set(tags)) > 1:
        #     print(tags)
        #     print(predicated_tags)
        #     print("print tokens", sample["tokens"])
        #     print("print input", generated_text.prompt)
        #     print("print output", generated_text.outputs[0].text)
        #     print(out_text)
        #     fdsfds

        prediction.append(predicated_tags)
        ground_tags.append(tags)

    results = nervaluate.Evaluator(
        ground_tags, prediction, tags=list(entities), loader="list"
    ).evaluate()
    print("new metric results ", results[0])
    print("new metric results_per_tag ", results[1])

    for mode in ("lenient", "strict"):
        lenient = classification_report(
            tags_true=flatten_extend(ground_tags),
            tags_pred=flatten_extend(prediction),
            mode=mode,
        )  # for lenient match
        print(mode, " metric ", json.dumps(lenient))
