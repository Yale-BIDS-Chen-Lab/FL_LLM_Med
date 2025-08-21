import json

import nervaluate
from cyy_naive_lib.algorithm.sequence_op import flatten_list
from ner_metrics import classification_report

from NER_evaluation.common import replace_tag


def print_metrics(ground_tags, prediction, canonical_tags):
    # results = nervaluate.Evaluator(
    #     ground_tags, prediction, tags=list(canonical_tags), loader="list"
    # ).evaluate()
    # print("new metric results ", results)

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
