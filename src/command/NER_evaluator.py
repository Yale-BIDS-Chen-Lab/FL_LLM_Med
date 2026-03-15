import argparse
import copy
import functools
import json
import logging
import os
import sys

import numpy as np

os.environ["NO_TOKENIZER_TRANSFORMS"] = "1"
import cyy_huggingface_toolbox  # noqa: F401
from cyy_naive_lib import save_json
from cyy_naive_lib.log import log_warning, set_level
from cyy_preprocessing_pipeline.parsing import approximately_match_tokens, html2bio, json2bio
from cyy_preprocessing_pipeline.parsing.bio.types import CanonicalTags, make_bio_span
from cyy_torch_toolbox import MachineLearningPhase
from distributed_learning_simulation import Session
from NER_evaluation.metric import get_metrics
from NER_evaluation.token_classification import process_batch
from util import get_tester

project_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, project_path)
import src.method  # noqa: F401


def robust_json2bio(json_text: str, canonical_tags: CanonicalTags) -> list | None:
    """Parse JSON entities from LLM output, using json_repair for malformed JSON.
    Returns None if the output is completely unfixable."""
    from json_repair import repair_json

    # Try direct parsing first
    try:
        return json2bio(json_text=json_text, canonical_tags=canonical_tags)
    except Exception:
        pass

    # Use json_repair to fix malformed JSON
    try:
        repaired = repair_json(json_text, return_objects=True)
    except (RecursionError, Exception):
        return None
    if isinstance(repaired, list):
        tokens = []
        for ent in repaired:
            if not isinstance(ent, dict):
                continue
            tag = canonical_tags.match(ent.get("entity", ""))
            text = ent.get("text", "")
            if not tag or not text:
                continue
            words = text.strip().split()
            if words:
                tokens.append(make_bio_span(words, tag))
        return tokens

    return None


def get_finetune_dir(
    zero_shot: bool, session: Session, worker_index: int | None = None
) -> str | None:
    finetuned_model_dir = None
    if not zero_shot:
        if worker_index is not None:
            assert worker_index < session.config.worker_number
            finetuned_model_dir = os.path.join(
                session.worker_dir(worker_index=worker_index), "SFTTrainer"
            )
        else:
            finetuned_model_dir = os.path.join(session.server_dir, "SFTTrainer")
    return finetuned_model_dir


if __name__ == "__main__":
    set_level(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="Analyze NER result",
    )
    parser.add_argument("--session_dir", help="session dir", type=str, required=True)
    parser.add_argument(
        "--pretrained_model_dir", help="pretrained model dir", type=str, default=None
    )
    parser.add_argument("--test_file", help="test file", type=str, default=None)
    parser.add_argument(
        "--skipped_tags", help="tags to skip evaluation", type=str, default=None
    )
    parser.add_argument(
        "--zero_shot", help="use pretrained model", action="store_true"
    )
    parser.add_argument(
        "--parse_gt_html", help="parsing gt html into tags", action="store_true"
    )
    parser.add_argument(
        "--worker_index", help="evaluate worker", type=int, default=None
    )
    parser.add_argument(
        "--sample_size", help="randomly sample from test_file", type=int, default=None
    )
    parser.add_argument(
        "--sample_times", help="times to sample from test_file", type=int, default=None
    )
    parser.add_argument(
        "--output_dir", help="dir to save results", type=str, default=None
    )
    parser.add_argument(
        "--sample_output_dir", help="dir to save sample results", type=str, default=None
    )
    parser.add_argument(
        "--results_file",
        help="file to append clean metric results (separate from vllm noise)",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    assert not (args.zero_shot and args.worker_index is not None)
    assert os.path.isdir(args.session_dir), args.session_dir
    canonical_tags: set[str] = set()
    skipped_tags: set[str] = set()
    if args.skipped_tags is not None:
        skipped_tags = set(args.skipped_tags.split(" "))

    session = Session(session_dir=args.session_dir)
    prompt_file = session.config.dc_config.dataset_kwargs.get("prompt_file", "")
    output_format = "json" if "json" in prompt_file else "html"
    print(f"Auto-detected output_format={output_format} from prompt_file={prompt_file}")
    use_llm = session.config.model_config.model_name.startswith(
        "hugging_face_causal_lm_"
    )
    if args.pretrained_model_dir is not None:
        assert use_llm, session.config.model_config.model_name
        assert os.path.exists(args.pretrained_model_dir), args.pretrained_model_dir
        log_warning(
            "The fine-tuned LLM is %s, the specified LLM is %s, ensure that they are the same",
            session.config.model_config.model_name.removeprefix(
                "hugging_face_causal_lm_"
            ),
            args.pretrained_model_dir,
        )
        session.config.model_config.model_name = (
            f"hugging_face_causal_lm_{args.pretrained_model_dir}"
        )
    if args.sample_times is not None:
        assert args.sample_size is not None

    tester, labels = get_tester(
        session=session, data_file=args.test_file, for_language_modeling=use_llm
    )
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
    test_file_name = os.path.basename(args.test_file) if args.test_file else "unknown"
    print(f"\n{'='*60}")
    print(f"Test file: {test_file_name}")
    print(f"{'='*60}")
    print("canonical_tags are", canonical_tags)
    bio_canonical_tags = CanonicalTags(canonical_tags)
    print("skipped_tags are", skipped_tags)
    sample_times = 1 if args.sample_times is None else args.sample_times
    vllm_engine = None
    for sample_idx in range(sample_times):
        prediction = []
        ground_tags = []
        unfixable_samples: list[dict] = []
        if args.sample_size is not None:
            test_len = len(
                tester.dataset_collection.get_dataset_util(
                    phase=MachineLearningPhase.Test
                )
            )
            assert args.sample_size < test_len
            rng = np.random.default_rng()
            indices = rng.choice(
                list(range(test_len)), size=args.sample_size, replace=False
            ).tolist()
            tester.mutable_dataset_collection.set_subset(
                phase=MachineLearningPhase.Test, indices=set(indices)
            )

        if use_llm:
            from vllm_generator import get_llm_engine, get_vllm_output

            if vllm_engine is None:
                finetuned_model_dir = get_finetune_dir(
                    zero_shot=args.zero_shot,
                    session=session,
                    worker_index=args.worker_index,
                )
                vllm_engine = get_llm_engine(
                    session=session, finetuned_model_dir=finetuned_model_dir
                )

            vllm_output = list(get_vllm_output(tester=tester, engine=vllm_engine))
            for idx, (sample, generated_text) in enumerate(vllm_output):
                out_text = generated_text.outputs[0].text
                tags = sample["tags"]
                assert tags
                tokens = sample["tokens"]
                if args.parse_gt_html:
                    assert "html" in sample
                    parsed_gt_tokens = html2bio(
                        html=sample["html"], canonical_tags=bio_canonical_tags
                    )
                    tags = []
                    tokens = []
                    for t in parsed_gt_tokens:
                        if isinstance(t, str):
                            tokens.append(t)
                            tags.append("O")
                        else:
                            tokens += t[0]
                            tags += t[1]

                if output_format == "json":
                    predicated_tokens = robust_json2bio(
                        json_text=out_text, canonical_tags=bio_canonical_tags
                    )
                    if predicated_tokens is None:
                        unfixable_samples.append(
                            {"index": idx, "output": out_text}
                        )
                        continue
                else:
                    predicated_tokens = html2bio(
                        html=out_text, canonical_tags=bio_canonical_tags
                    )
                tmp_predicated_tags = approximately_match_tokens(
                    tokens, predicated_tokens
                )
                predicated_tags = [
                    t if t is not None else "O" for t in tmp_predicated_tags
                ]
                assert len(predicated_tags) == len(tags)
                prediction.append(predicated_tags)
                ground_tags.append(tags)

                if args.sample_output_dir is not None:
                    assert args.sample_times is None
                    metrics = get_metrics(
                        ground_tags=[tags],
                        prediction=[predicated_tags],
                        canonical_tags=canonical_tags,
                    )
                    lenient_f1 = metrics["lenient"]["unified_class"]["f1-score"]
                    strict_f1 = metrics["strict"]["unified_class"]["f1-score"]

                    os.makedirs(args.sample_output_dir, exist_ok=True)
                    with open(
                        os.path.join(
                            args.sample_output_dir,
                            f"{idx}_{lenient_f1}_{strict_f1}.txt",
                        ),
                        "w",
                        encoding="utf8",
                    ) as debug_f:
                        debug_f.write("input <<<<<<<<<<<<<<\n")
                        joined_tokens = " ".join(tokens)
                        if "inputs" in sample:
                            joined_tokens = sample["inputs"]
                        debug_f.write(f"{joined_tokens}\n")
                        debug_f.write(f"ground_out ({output_format}) ==============\n")
                        if "html" in sample:
                            debug_f.write(f"{sample['html']}\n")
                        else:
                            debug_f.write(f"{sample['output']}\n")
                        debug_f.write("predicated_out >>>>>>>>>>>>>>\n")
                        debug_f.write(f"{out_text}\n")
        else:
            if not args.zero_shot:
                assert args.worker_index is None
                tester.model_util.load_parameters(session.get_last_model_parameters())
            tester.process_sample_output(
                functools.partial(
                    process_batch, ground_tags, prediction, skipped_tags, labels
                )
            )
        total_samples = len(prediction) + len(unfixable_samples)

        metrics = get_metrics(
            ground_tags=ground_tags,
            prediction=prediction,
            canonical_tags=canonical_tags,
        )

        # Build clean result lines
        sample_label = f"{test_file_name}" if sample_times == 1 else f"{test_file_name} sample={sample_idx}"
        result_lines = []
        result_lines.append(f"\n{'='*60}")
        result_lines.append(f"[{sample_label}] output_format={output_format}")
        result_lines.append(f"[{sample_label}] Parse stats: {len(prediction)}/{total_samples} parsed, {len(unfixable_samples)} unfixable")
        if unfixable_samples:
            result_lines.append(f"[{sample_label}] Unfixable indices: {[s['index'] for s in unfixable_samples]}")
        result_lines.append(f"[{sample_label}] Metrics:")
        for metric_type in ("strict", "lenient"):
            if metric_type in metrics:
                for tag, values in metrics[metric_type].items():
                    if isinstance(values, dict) and "f1-score" in values:
                        result_lines.append(
                            f"  [{sample_label}] {metric_type:8s} {tag:20s} "
                            f"P={values.get('precision', 0):.4f} "
                            f"R={values.get('recall', 0):.4f} "
                            f"F1={values['f1-score']:.4f}"
                        )
        result_lines.append(f"{'='*60}")
        result_text = "\n".join(result_lines)

        # Print to stdout
        print(result_text)

        # Append to results file if specified
        if args.results_file:
            with open(args.results_file, "a", encoding="utf8") as rf:
                rf.write(result_text + "\n")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            metrics["test_file"] = test_file_name
            metrics["output_format"] = output_format
            metrics["unfixable_count"] = len(unfixable_samples)
            metrics["total_count"] = total_samples
            metrics["unfixable_samples"] = unfixable_samples
            save_json(
                metrics, os.path.join(args.output_dir, f"metrics_{test_file_name}_{sample_idx}.json")
            )
