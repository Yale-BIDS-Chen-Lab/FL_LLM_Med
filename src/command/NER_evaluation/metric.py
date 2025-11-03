import json

import nervaluate
from cyy_naive_lib.algorithm.sequence_op import flatten_list
from NER_evaluation.common import replace_tag
from ner_metrics import classification_report


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
