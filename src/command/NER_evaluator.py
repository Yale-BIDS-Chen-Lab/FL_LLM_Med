import argparse
import json
import logging

import nervaluate
from cyy_naive_lib.algorithm.sequence_op import flatten_list
from cyy_naive_lib.log import set_level
from NER_evaluation.common import match_tokens, replace_tag
from NER_evaluation.html_form import html2bio
from ner_metrics import classification_report
from vllm_generator import get_vllm_output

if __name__ == "__main__":
    set_level(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="Analyze NER result",
    )
    parser.add_argument("--session_dir", help="session dir", type=str, required=True)
    parser.add_argument("--test_file", help="test file", type=str, default=None)
    parser.add_argument(
        "--debug_file", help="contains debug info", type=str, default=None
    )
    args = parser.parse_args()

    prediction: list[list[str]] = []
    ground_tags: list[list[str]] = []
    debug_f = None
    if args.debug_file is not None:
        debug_f = open(args.debug_file, "w", encoding="utf8")
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
        predicated_tokens = html2bio(html=out_text, canonical_tags=canonical_tags)
        predicated_tags = match_tokens(tokens, predicated_tokens)
        if len(set(tags)) > 1 and set(predicated_tags) == {"O"} and debug_f is not None:
            joined_tokens = " ".join(tokens)
            debug_f.write("<<<<<<<<<<<<<<\n")
            debug_f.write(f"{joined_tokens}\n")
            debug_f.write("==============\n")
            joined_tags = " ".join(tags)
            debug_f.write(f"{joined_tags}\n")
            debug_f.write(">>>>>>>>>>>>>>\n")
            debug_f.write(f"{out_text}\n")

        prediction.append(predicated_tags)
        ground_tags.append(tags)
    if debug_f is not None:
        debug_f.close()

    results = nervaluate.Evaluator(
        ground_tags, prediction, tags=list(canonical_tags), loader="list"
    ).evaluate()
    print("new metric results ", results[0])
    print("new metric results_per_tag ", results[1])

    for mode in ("lenient", "strict"):
        result = classification_report(
            tags_true=flatten_list(ground_tags),
            tags_pred=flatten_list(prediction),
            mode=mode,
        )
        print(mode, " metric ", json.dumps(result, sort_keys=True))

    for mode in ("lenient", "strict"):
        result = classification_report(
            tags_true=flatten_list(replace_tag(ground_tags, canonical_tags)),
            tags_pred=flatten_list(replace_tag(prediction, canonical_tags)),
            mode=mode,
        )
        print(mode, " metric ", json.dumps(result, sort_keys=True))
