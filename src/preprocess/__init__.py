import os
from typing import Any

from cyy_naive_lib.fs.path import list_files, list_files_by_suffixes

from .data_pipeline import get_iob_pipeline
from .iob import IOB, IOBRecord


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


__all__ = ["parse_dir", "IOBRecord", "get_iob_pipeline"]
