import os
from typing import Any

from cyy_naive_lib.fs.path import list_files, list_files_by_suffixes


class Parser:
    def parse(self, lines: list[str]) -> Any:
        raise NotImplementedError()


type IOBData = list[tuple[list[str], str | None]]


class IOB(Parser):
    def parse(self, lines: list[str]) -> IOBData:
        phrase: IOBData = []
        last_type: str | None = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            idx = line.rfind("\t")
            token_type = line[idx + 1 :]
            token = line[:idx]
            if token_type == "O":
                last_type = None
                phrase.append(([token], last_type))
            elif token_type.startswith("B-"):
                last_type = token_type[2:]
                phrase.append(([token], last_type))
            elif token_type.startswith("I-"):
                this_type = token_type[2:]
                if last_type == this_type:
                    phrase[-1][0].append(token)
                else:
                    last_type = this_type
                    phrase.append(([token], last_type))
            else:
                raise RuntimeError(f"invalid line:{line}")
        return phrase


class IOBRecord:
    def __init__(self, data: IOBData) -> None:
        self.__data = data

    @property
    def text(self) -> str:
        return " ".join(" ".join(t[0]) for t in self.__data)


def parse_file(file: str) -> Any:
    parsers = {".bio": IOB(), ".iob": IOB()}
    for suffix, parser in parsers.items():
        if file.endswith(suffix):
            with open(file, encoding="utf8") as f:
                return parser.parse(f.readlines())
    raise NotImplementedError()


def parse_dir(data_dir: str, suffix: str | None = None) -> dict:
    assert os.path.isdir(data_dir)
    res = {}
    files = (
        list_files(data_dir)
        if suffix is None
        else list_files_by_suffixes(data_dir, suffixes=[suffix])
    )
    for file in files:
        res[file] = parse_file(file)
    return res
