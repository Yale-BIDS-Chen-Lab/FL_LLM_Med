from typing import Any
from cyy_naive_lib.fs.path import list_files
import os


class Parser:
    def parse(self, lines: list[str]) -> Any:
        raise NotImplementedError()


class I2B2(Parser):
    def parse(self, lines: list[str]) -> list[tuple[list[str], str]]:
        phrase: list[tuple[list[str], str]] = []
        last_type = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            token, token_type = line.split("\t")
            if token_type == "O":
                phrase.append(([token], token_type))
                last_type = ""

            if token_type.startswith("B-"):
                last_type = token_type[2:]
                phrase.append(([token], token_type))

            if token_type.startswith("I-"):
                assert last_type == token_type[2:]
                last_type = token_type[2:]
                phrase[-1][0].append(token)
        return phrase


def parse_file(file: str) -> Any:
    parsers = {".i2b2": I2B2()}
    for suffix, parser in parsers.items():
        if file.endswith(suffix):
            with open(file, encoding="utf8") as f:
                return parser.parse(f.readlines())
    raise NotImplementedError()


def parse_local_data(data_dir: str) -> dict:
    assert os.path.isdir(data_dir)
    res = {}
    for file in list_files(data_dir):
        res[file] = parse_file(file)
    return res
