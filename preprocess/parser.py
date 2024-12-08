from typing import Any
from cyy_naive_lib.fs.path import list_files, list_files_by_suffixes
import os


class Parser:
    def parse(self, lines: list[str]) -> Any:
        raise NotImplementedError()


class IOB(Parser):
    def parse(self, lines: list[str]) -> list[tuple[list[str], str | None]]:
        phrase: list[tuple[list[str], str | None]] = []
        last_type = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            idx = line.rfind("\t")
            token_type = line[idx + 1 :]
            token = line[:idx]
            if token_type == "O":
                phrase.append(([token], None))
                last_type = ""
            elif token_type.startswith("B-"):
                last_type = token_type[2:]
                phrase.append(([token], last_type))
            elif token_type.startswith("I-"):
                this_type = token_type[2:]
                if last_type == this_type:
                    phrase[-1][0].append(token)
                    last_type = ""
                else:
                    phrase.append(([token], this_type))
            else:
                raise RuntimeError(f"invalid line:{line}")
        return phrase


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
