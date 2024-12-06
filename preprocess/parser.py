from typing import Any


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


def parse_file(file: str) -> list:
    parsers = {".i2b2": I2B2()}
    for suffix, parser in parsers.items():
        if file.endswith(suffix):
            with open(file, encoding="utf8") as f:
                return parser.parse(f.readlines())
    raise NotImplementedError()
