import argparse
import json
import os
import sys

from datasets import load_dataset

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)


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
        prog="IBO IID split",
    )
    parser.add_argument("--csv_file", help="raw data dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        help="output dir for json files",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    prompt = None
    dataset = load_dataset("csv", data_files=[args.csv_file], split="all").to_dict()

    pairs = []
    prompt_set = set()
    print("sample size", len(dataset["unprocessed"]))
    for i in range(len(dataset["unprocessed"])):
        text = f"{dataset['unprocessed'][i]} {dataset['processed'][i]}".strip()
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
        input_text = lines[0][len(prefix) :].strip()
        prefix = "### Output Text:"
        assert lines[1].startswith(prefix)
        output_text = lines[1][len(prefix) :].replace("<EOS>", "").strip()
        pairs.append({"input": input_text, "output": output_text})
    print(prompt)
    with open(
        os.path.join(
            args.output_dir,
            os.path.basename(args.csv_file.replace(".csv", ".json")),
        ),
        "w",
        encoding="utf8",
    ) as f:
        json.dump({"prompt": prompt, "data": pairs}, f)
