import argparse
import copy
import json
import logging

import nervaluate
from cyy_naive_lib.log import set_level
from medical_NER_evaluation.common import match_tokens
from medical_NER_evaluation.html_form import html2bio
from ner_metrics import classification_report
from vllm_generator import get_vllm_output


def flatten_extend(l: list[list]) -> list:
    flat_list = []
    for e in l:
        flat_list.extend(e)
    return flat_list


def replace_tag(l: list[list[str]]) -> list[list[str]]:
    res = []
    for a in l:
        new_tags = copy.deepcopy(a)
        for idx, b in enumerate(new_tags):
            for canonical_tag in canonical_tags:
                if canonical_tag in b:
                    new_tags[idx] = b.replace(canonical_tag, "unified_class")
                    break
        res.append(new_tags)
    return res


if __name__ == "__main__":
    set_level(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="Analyze NER result",
    )
    parser.add_argument("--session_dir", help="session dir", type=str, required=True)
    parser.add_argument("--test_file", help="test file", type=str, default=None)
    parser.add_argument("--output_file", help="outputfile", type=str, default=None)
    args = parser.parse_args()

    prediction: list[list[str]] = []
    ground_tags: list[list[str]] = []
    output_f = None
    if args.output_file is not None:
        output_f = open(args.output_file, "w", encoding="utf8")
    vllm_output = list(
        get_vllm_output(session_dir=args.session_dir, data_file=args.test_file)
    )
    canonical_tags: set[str] = set()
    for sample, _ in vllm_output:
        tags = sample["tags"]
        canonical_tags.update(tags)
    canonical_tags.remove("O")
    canonical_tags = {
        tag.removeprefix("I-").removeprefix("B-") for tag in canonical_tags
    }

    for sample, generated_text in vllm_output:
        tags = sample["tags"]
        out_text = generated_text.outputs[0].text
        tokenizer = sample["tokenizer"]
        tokens = sample["tokens"]
        predicated_tags: list[str] = []
        predicated_candidate_tags: list[str] = []
        if output_f is not None:
            joined_tokens = " ".join(tokens)
            output_f.write("<<<<<<<<<<<<<<\n")
            output_f.write(f"{joined_tokens}\n")
            output_f.write("==============\n")
            joined_tags = " ".join(tags)
            output_f.write(f"{joined_tags}\n")
            output_f.write(">>>>>>>>>>>>>>\n")
            output_f.write(f"{out_text}\n")
        predicated_tokens = html2bio(
            html=out_text, canonical_tags=canonical_tags, tokenizer=tokenizer
        )
        predicated_tags = match_tokens(tokens, predicated_tokens)

        prediction.append(predicated_tags)
        ground_tags.append(tags)
    if output_f is not None:
        output_f.close()

    results = nervaluate.Evaluator(
        ground_tags, prediction, tags=list(canonical_tags), loader="list"
    ).evaluate()
    print("new metric results ", results[0])
    print("new metric results_per_tag ", results[1])

    for mode in ("lenient", "strict"):
        lenient = classification_report(
            tags_true=flatten_extend(ground_tags),
            tags_pred=flatten_extend(prediction),
            mode=mode,
        )  # for lenient match
        print(mode, " metric ", json.dumps(lenient, sort_keys=True))

    for mode in ("lenient", "strict"):
        lenient = classification_report(
            tags_true=flatten_extend(replace_tag(ground_tags)),
            tags_pred=flatten_extend(replace_tag(prediction)),
            mode=mode,
        )  # for lenient match
        print(mode, " metric ", json.dumps(lenient, sort_keys=True))
