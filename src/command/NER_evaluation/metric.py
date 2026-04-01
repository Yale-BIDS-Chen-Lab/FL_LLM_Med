import json

import matplotlib.pyplot as plt
import nervaluate
from cyy_naive_lib.algorithm.sequence_op import flatten_list
from NER_evaluation.common import replace_tag
from ner_metrics import classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve


def _build_y(
    ground_tags: list[list[str]],
    prediction: list[list[str]],
    sample_confidences: list[float],
    match_fn,
) -> tuple[list[int], list[float]]:
    y_true = []
    y_score = []
    for gt_tags, pred_tags, conf in zip(
        ground_tags, prediction, sample_confidences, strict=True
    ):
        for gt, pred in zip(gt_tags, pred_tags, strict=True):
            y_true.append(1 if match_fn(gt) else 0)
            y_score.append(conf if match_fn(pred) else 0.0)
    return y_true, y_score


def compute_auprc(
    ground_tags: list[list[str]],
    prediction: list[list[str]],
    canonical_tags: set[str],
    sample_confidences: list[float],
) -> dict[str, float]:
    """Compute token-level AUPRC using per-sample confidence scores.

    Entity-predicted tokens get the sample's confidence as score; O tokens get 0.
    Returns dict with per-tag and unified_class AUPRC values.
    """
    results: dict[str, float] = {}

    # Unified class: any entity vs O
    y_true, y_score = _build_y(
        ground_tags, prediction, sample_confidences, lambda t: t != "O"
    )
    if any(y_true):
        results["unified_class"] = float(average_precision_score(y_true, y_score))

    # Per tag (uses original tags, e.g. B-treatment, I-treatment)
    for tag in sorted(canonical_tags):
        y_true, y_score = _build_y(
            ground_tags, prediction, sample_confidences, lambda t, tag=tag: t.endswith(f"-{tag}")
        )
        if any(y_true):
            results[tag] = float(average_precision_score(y_true, y_score))

    return results


def plot_prc(
    ground_tags: list[list[str]],
    prediction: list[list[str]],
    canonical_tags: set[str],
    sample_confidences: list[float],
    output_path: str,
) -> None:
    """Plot token-level precision-recall curves and save to file."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Unified class
    y_true, y_score = _build_y(
        ground_tags, prediction, sample_confidences, lambda t: t != "O"
    )
    if any(y_true):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, label=f"unified_class (AP={ap:.4f})", linewidth=2)

    # Per tag
    for tag in sorted(canonical_tags):
        y_true, y_score = _build_y(
            ground_tags, prediction, sample_confidences, lambda t, tag=tag: t.endswith(f"-{tag}")
        )
        if any(y_true):
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, label=f"{tag} (AP={ap:.4f})", linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Token-level Precision-Recall Curve")
    ax.legend(loc="best")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_metrics(ground_tags, prediction, canonical_tags) -> dict:
    results = nervaluate.Evaluator(
        ground_tags, prediction, tags=list(canonical_tags), loader="list"
    )
    print("new metric results ", results.summary_report())

    for mode in ("lenient", "strict"):
        result = classification_report(
            tags_true=flatten_list(ground_tags),
            tags_pred=flatten_list(prediction),
            mode=mode,
        )
        print(mode, " metric ", json.dumps(result, sort_keys=True))

    result = {}
    for mode in ("lenient", "strict"):
        report = classification_report(
            tags_true=flatten_list(replace_tag(ground_tags, canonical_tags)),
            tags_pred=flatten_list(replace_tag(prediction, canonical_tags)),
            mode=mode,
        )
        result[mode] = report
    print(" metric ", json.dumps(result, sort_keys=True))
    return result
