from pathlib import Path
from typing import Any

from cyy_preprocessing_pipeline.dataset import (
    IOBParser,
    IOBRecord,
    JSONParser,
    JSONRecord,
)


def parse_file(file: str) -> Any:
    parsers = {".bio": IOBParser(), ".iob": IOBParser(), ".json": JSONParser()}
    for suffix, parser in parsers.items():
        if file.endswith(suffix):
            with open(file, encoding="utf8") as f:
                return parser.parse(f.readlines())
    raise NotImplementedError()


def parse_dir(data_dir: str, suffix: str | None = None) -> dict:
    data_path = Path(data_dir)
    assert data_path.is_dir()
    res = {}
    suffixes = []
    if suffix is not None:
        suffixes.append(suffix)
        if suffix == "bio":
            suffixes.append("iob")
        if suffix == "iob":
            suffixes.append("bio")
    if suffixes:
        files = [
            str(p) for s in suffixes for p in data_path.rglob(f"*.{s}") if p.is_file()
        ]
    else:
        files = [str(p) for p in data_path.rglob("*") if p.is_file()]
    for file in files:
        res[file] = parse_file(file)
    return res


__all__ = ["IOBRecord", "JSONRecord", "parse_dir"]
