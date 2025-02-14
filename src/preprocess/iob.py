import os

from .parser import Parser


class IOBRecord:
    def __init__(self) -> None:
        self.__tokens: list[str] = []
        self.__token_tags: list[str] = []
        self.last_tag: str | None = None
        self.phrase: list[tuple[list[str], str, int]] = []

    def add_line(self, token: str, token_tag: str) -> None:
        self.__tokens.append(token)
        self.__token_tags.append(token_tag)
        if token_tag == "O":
            self.last_tag = None
        elif token_tag.startswith("B-"):
            self.last_tag = token_tag[2:]
            self.phrase.append(([token], self.last_tag, len(self.__tokens) - 1))
        elif token_tag.startswith("I-"):
            this_tag = token_tag[2:]
            if self.last_tag == this_tag:
                self.phrase[-1][0].append(token)
            else:
                self.last_tag = this_tag
                self.phrase.append(([token], self.last_tag, len(self.__token_tags) - 1))
        else:
            raise RuntimeError(f"invalid line:{token} {token_tag}")

    def to_json(self) -> dict:
        return {
            "tokens": self.tokens,
            "annotated_phrases": self.annotated_phrases,
            "annotated_phrase_locations": self.annotated_phrase_locations,
            "tags": self.__token_tags,
        }

    @property
    def tokens(self) -> list[str]:
        return self.__tokens

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def annotated_phrase_locations(self) -> list[int]:
        return [p[2] for p in self.phrase]

    @property
    def annotated_phrases(self) -> list[tuple[str, str]]:
        return [(" ".join(p[0]), p[1]) for p in self.phrase]


class IOB(Parser):
    def parse(self, lines: list[str]) -> list[IOBRecord]:
        results: list[IOBRecord] = []
        record = IOBRecord()
        for _line in lines:
            line = _line.strip()
            skip_empty_line = os.getenv("SKIP_EMPTY_LINE")
            assert skip_empty_line is not None
            if not line and int(skip_empty_line):
                if record.tokens:
                    results.append(record)
                    record = IOBRecord()
                continue
            idx = line.rfind("\t")
            token_tag = line[idx + 1 :]
            token = line[:idx]
            record.add_line(token, token_tag)
        if record.tokens:
            results.append(record)
        return results
