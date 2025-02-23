import argparse

from ner_metrics import classification_report
from vllm_generator import get_vllm_output

from .medical_NER_evaluation.common import find_tag
from .medical_NER_evaluation.html_form import html2bio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Analyze NER result",
    )
    parser.add_argument("--session_dir", help="session dir", type=str, required=True)
    parser.add_argument("--test_file", help="test file", type=str, default=None)
    args = parser.parse_args()

    prediction = []
    ground_tags = []
    entities = ["problem", "treatment", "test", "drug"]

    for sample, generated_text in get_vllm_output(
        session_dir=args.session_dir, data_file=args.test_file
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

    lenient = classification_report(
        tags_true=ground_tags, tags_pred=prediction, mode="lenient"
    )  # for lenient match
    print(lenient)
    strict = classification_report(
        tags_true=ground_tags, tags_pred=prediction, mode="strict"
    )
    print(strict)

    # print(tag_set)
    # return Evaluator(
    #     ground_tags, prediction, tags=list(tag_set), loader="list"
    # ).evaluate()
    # print("NER test results ", results)
    # print("NER test results_per_tag ", results_per_tag)
