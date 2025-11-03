import argparse
import json
import os
import sys

from datasets import load_dataset

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)

from NER_evaluation.html_form import html2bio


def format_prompt(prompt: str) -> str:
    assert prompt is not None
    lines = prompt.splitlines()
    i = 0
    for line in lines:
        if line.startswith("Use <span class="):
            break
        i += 1
    assert i != 0
    prompt = "\n".join(lines[:i] + sorted(lines[i:]))
    return prompt


def refine_prompt(prompt: str) -> str:
    return (
        prompt.replace(
            "denote a calendar date, time, or duration related to a test.",
            "denote a calendar date, time, or duration related to a test/treatment/problem/drug.",
        )
        .replace(
            "denote a calendar date, time, or duration related to a treatment.",
            "denote a calendar date, time, or duration related to a test/treatment/problem/drug.",
        )
        .replace(
            "denote a calendar date, time, or duration related to a problem.",
            "denote a calendar date, time, or duration related to a test/treatment/problem/drug.",
        )
        .replace(
            "denote a calendar date, time, or duration related to a drug.",
            "denote a calendar date, time, or duration related to a test/treatment/problem/drug.",
        )
        .strip()
    )


def add_class(prompt: str, line: str) -> str | None:
    assert line
    if not line.startswith("Use <span class="):
        return None
    tag_idx = line.find(">")
    if tag_idx < 0:
        return None
    tag = line[:tag_idx]
    if tag in prompt:
        return None
    return f"{prompt}\n{line}"


def strip_text(content: str) -> str:
    lines = content.splitlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="CSV parser",
    )
    parser.add_argument("--csv_files", help="CSV files", type=str, required=True)
    parser.add_argument(
        "--output_file",
        help="output filename",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    prompt = None
    dataset = load_dataset(
        "csv", data_files=args.csv_files.split(" "), split="all"
    ).to_dict()

    pairs = []
    prompt_set = set()
    for i in range(len(dataset["unprocessed"])):
        text = f"{dataset['unprocessed'][i]} {dataset['processed'][i]}".strip()
        text = text.replace("<EOS>", "")
        text = strip_text(text)
        idx = text.find("### Input")
        assert idx >= 0
        new_prompt = text[:idx]
        text = text.removeprefix(new_prompt)
        new_prompt = refine_prompt(new_prompt)
        if prompt is None:
            prompt = new_prompt
            prompt_set = set(prompt.splitlines())
        elif prompt != new_prompt:
            new_prompt_set = set(new_prompt.splitlines())
            if prompt_set.issubset(new_prompt_set):
                prompt = new_prompt
                prompt_set = set(prompt.splitlines())
                assert "" not in prompt_set
            else:
                while not new_prompt_set.issubset(prompt_set):
                    for line in new_prompt_set:
                        if line not in prompt_set:
                            tmp = add_class(prompt, line)
                            if tmp is None:
                                raise RuntimeError(line)
                            prompt = tmp
                            prompt_set = set(prompt.splitlines())
                            break

        text = strip_text(text)
        lines = text.splitlines()
        assert len(lines) == 2
        prefix = "### Input Text:"
        assert lines[0].startswith(prefix)
        input_text = lines[0].removeprefix(prefix).strip()
        # if len(tags) != input_text.split(" "):
        #     print("====================")
        #     print(input_text)
        #     print("====================")
        #     for tag in html2bio(input_text):
        #         if isinstance(tag, str):
        #             print(tag)
        #         else:
        #             print(tag[0])
        #     print("====================")

        prefix = "### Output Text:"
        assert lines[1].startswith(prefix)
        output_text = lines[1].removeprefix(prefix).strip()
        # print("=====")
        # print(input_text)
        # print(output_text)
        # print("=====")
        tags: list[str] = []
        tokens: list[str] = []
        for token in html2bio(output_text):
            if isinstance(token, str):
                tokens.append(token)
                tags.append("O")
            else:
                tokens += token[0]
                tags += token[1]
        assert tokens
        assert tags
        pairs.append(
            {"input": input_text, "output": output_text, "tags": tags, "tokens": tokens}
        )

    assert prompt is not None
    prompt = format_prompt(prompt)
    # print(prompt)

    with open(
        args.output_file,
        "w",
        encoding="utf8",
    ) as f:
        json.dump(pairs, f)
