from .parser import Parser

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

    @property
    def annotated_phrases(self) -> list[tuple[str, str]]:
        res: list[tuple[str, str]] = []
        for item in self.__data:
            if item[1] is not None:
                res.append((" ".join(item[0]), item[1]))
        return res
