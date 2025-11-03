import os
from typing import Any

from cyy_naive_lib.fs.path import list_files, list_files_by_suffixes
from cyy_preprocessing_pipeline.dataset import IOBParser, IOBRecord, JSONParser


def parse_file(file: str) -> Any:
    parsers = {".bio": IOBParser(), ".iob": IOBParser(), ".json": JSONParser()}
    for suffix, parser in parsers.items():
        if file.endswith(suffix):
            with open(file, encoding="utf8") as f:
                return parser.parse(f.readlines())
    raise NotImplementedError()


def parse_dir(data_dir: str, suffix: str | None = None) -> dict:
    assert os.path.isdir(data_dir)
    res = {}
    suffixes = []
    if suffix is not None:
        suffixes.append(suffix)
        if suffix == "bio":
            suffixes.append("iob")
        if suffix == "iob":
            suffixes.append("bio")
    files = (
        list_files(data_dir)
        if not suffixes
        else list_files_by_suffixes(data_dir, suffixes=suffixes)
    )
    for file in files:
        res[file] = parse_file(file)
    return res


__all__ = ["IOBRecord", "parse_dir"]
