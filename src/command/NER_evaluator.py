import argparse
import copy
import json
import os
import sys

import cyy_huggingface_toolbox  # noqa: F401
import nervaluate
from cyy_naive_lib.algorithm.sequence_op import flatten_list
from distributed_learning_simulation import (
    Session,
)
from NER_evaluation.common import match_tokens, replace_tag
from NER_evaluation.html_form import html2bio
from ner_metrics import classification_report
from util import get_model, get_tester

project_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, project_path)
import src.method  # noqa: F401

if __name__ == "__main__":
    # set_level(logging.DEBUG)
    parser = argparse.ArgumentParser(
        prog="Analyze NER result",
    )
    parser.add_argument("--session_dir", help="session dir", type=str, required=True)
    parser.add_argument("--test_file", help="test file", type=str, default=None)
    parser.add_argument(
        "--debug_file", help="contains debug info", type=str, default=None
    )
    parser.add_argument(
        "--skipped_tags", help="tags to skip evaluation", type=str, default=None
    )
    parser.add_argument(
        "--zero_shot", help="use pretrained model", type=bool, default=False
    )
    parser.add_argument(
        "--worker_index", help="evaluate worker", type=int, default=None
    )
    args = parser.parse_args()

    prediction: list[list[str]] = []
    ground_tags: list[list[str]] = []
    debug_f = None
    if args.debug_file is not None:
        debug_f = open(args.debug_file, "w", encoding="utf8")
    assert not (args.zero_shot and args.worker_index is not None)
    assert os.path.isdir(args.session_dir), args.session_dir
    canonical_tags: set[str] = set()
    skipped_tags: set[str] = set()
    if args.skipped_tags is not None:
        skipped_tags = set(args.skipped_tags.split(" "))

    session = Session(session_dir=args.session_dir)
    tester, labels = get_tester(session=session, data_file=args.test_file)
    labels = copy.deepcopy(labels)
    canonical_tags = copy.deepcopy(labels)
    labels = sorted(labels)
    canonical_tags.remove("O")
    canonical_tags = {
        tag.removeprefix("I-").removeprefix("B-") for tag in canonical_tags
    }
    if skipped_tags:
        canonical_tags = canonical_tags - skipped_tags
    assert canonical_tags
    print("canonical_tags are", canonical_tags)
    print("skipped_tags are", skipped_tags)
    if session.config.model_config.model_name.startswith("hugging_face_causal_lm_"):
        from vllm_generator import get_vllm_output

        vllm_output = list(
            get_vllm_output(
                tester=tester,
                session=session,
                zero_shot=args.zero_shot,
                worker_index=args.worker_index,
            )
        )
        for sample, generated_text in vllm_output:
            out_text = generated_text.outputs[0].text
            tags = sample["tags"]
            tokens = sample["tokens"]
            predicated_tokens = html2bio(html=out_text, canonical_tags=canonical_tags)
            predicated_tags = match_tokens(tokens, predicated_tokens)
            if (
                len(set(tags)) > 1
                and set(predicated_tags) == {"O"}
                and debug_f is not None
            ):
                debug_f.write("input <<<<<<<<<<<<<<\n")
                joined_tokens = " ".join(tokens)
                if "inputs" in sample:
                    joined_tokens = sample["inputs"]
                debug_f.write(f"{joined_tokens}\n")
                debug_f.write("ground_out ==============\n")
                joined_tags = " ".join(tags)
                if "output" in sample:
                    joined_tags = sample["output"]
                debug_f.write(f"{joined_tags}\n")
                debug_f.write("predicated_out >>>>>>>>>>>>>>\n")
                debug_f.write(f"{out_text}\n")
                predicated_out_text: list[str] = []
                for t in predicated_tokens:
                    if isinstance(t, str):
                        predicated_out_text.append(t)
                    else:
                        predicated_out_text += t[0]
                out_text = " ".join(predicated_out_text)
                debug_f.write(f"{out_text}\n")

            prediction.append(predicated_tags)
            ground_tags.append(tags)
        if debug_f is not None:
            debug_f.close()
    else:
        get_model(
            tester=tester,
            session=session,
            zero_shot=args.zero_shot,
            worker_index=args.worker_index,
        )

        def process_batch(batch_res):
            targets = batch_res["targets"].reshape(batch_res["batch_size"], -1)
            for tags, logits in zip(targets, batch_res["logits"], strict=False):
                assert tags.shape[0] == logits.shape[0]
                mask = tags != -100
                tags = tags[mask].tolist()
                logits = logits[mask].argmax(dim=-1).tolist()
                tags = [labels[tag] for tag in tags]
                predicated_tags = [labels[tag] for tag in logits]

                ground_tags.append(tags)
                prediction.append(predicated_tags)

        tester.process_sample_output(process_batch)

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
