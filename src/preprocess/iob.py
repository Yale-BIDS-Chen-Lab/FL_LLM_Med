from .parser import Parser

type IOBData = list[tuple[list[str], str | None]]


class IOBRecord:
    def __init__(self, data: IOBData) -> None:
        self.__data = data

    def to_json(self) -> dict:
        return {"tokens": self.tokens, "annotated_phrases": self.annotated_phrases}

    @property
    def tokens(self) -> list[str]:
        return sum((t[0] for t in self.__data), start=[])

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def annotated_phrases(self) -> list[tuple[str, str]]:
        res: list[tuple[str, str]] = []
        for item in self.__data:
            if item[1]:
                res.append((" ".join(item[0]), item[1]))
        return res


class IOB(Parser):
    def parse(self, lines: list[str]) -> list[IOBRecord]:
        results: list[IOBRecord] = []
        phrase: IOBData = []
        last_type: str | None = None
        for line in lines:
            line = line.strip()
            if not line:
                if phrase:
                    results.append(IOBRecord(phrase))
                    phrase = []
                last_type = None
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
        return results
