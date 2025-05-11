import json

from .parser import Parser


class IOBJSONRecord:
    def __init__(self, json_content: dict) -> None:
        self.__json_content = json_content

    def to_json(self) -> dict:
        return self.__json_content

    @property
    def token_tags(self) -> list[str]:
        return self.to_json()["tags"]


class IOBRecord:
    def __init__(self) -> None:
        self.__tokens: list[str | tuple[list[str], str]] = []
        self.__token_tags: list[str] = []
        self.__last_tag: str = "O"

    def add_line(self, token: str, token_tag: str) -> None:
        assert " " not in token
        self.__tokens.append(token)
        self.__token_tags.append(token_tag)
        if token_tag == "O":
            self.__last_tag = token_tag
        elif token_tag.startswith("B-"):
            old_last_tag = self.__last_tag
            self.__last_tag = token_tag
            self.__tokens.pop()
            self.__tokens.append(([token], self.__last_tag[2:]))
        elif token_tag.startswith("I-"):
            old_last_tag = self.__last_tag
            self.__last_tag = token_tag
            if self.__last_tag[2:] == old_last_tag[2:]:
                self.__tokens.pop()
                assert isinstance(self.__tokens[-1], tuple)
                assert self.__tokens[-1][1] == self.__last_tag[2:]
                self.__tokens[-1][0].append(token)
            else:
                self.__tokens.pop()
                # In principle, this should be I tag, but it's changed to B-tag for easier evaluation
                self.__token_tags.pop()
                self.__token_tags.append("B-" + token_tag[2:])
                self.__tokens.append(([token], self.__last_tag[2:]))
        else:
            raise RuntimeError(f"invalid line:{token} {token_tag}")

    @property
    def token_tags(self) -> list[str]:
        return self.__token_tags

    def to_json(self) -> dict:
        tokens = self.tokens
        assert len(tokens) == len(self.token_tags)

        return {
            "tokens": tokens,
            "tags": self.token_tags,
            "html": self.html,
        }

    @property
    def annotated_tokens(self) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for t in self.__tokens:
            if isinstance(t, str):
                result.append((t, "O"))
            else:
                tag = t[1]
                for token in t[0]:
                    result.append((token, tag))
        return result

    @property
    def tokens(self) -> list[str]:
        result: list[str] = []
        for t in self.__tokens:
            if isinstance(t, str):
                result.append(t)
            else:
                result += t[0]
        return result

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def annotated_phrases(self) -> list[tuple[str, str]]:
        return [(" ".join(p[0]), p[1]) for p in self.tokens if not isinstance(p, str)]

    @property
    def html(self) -> str:
        result: list[str] = []
        for t in self.__tokens:
            if isinstance(t, str):
                result.append(t)
            else:
                phrase = " ".join(t[0])
                result.append(f"<span class='{t[1]}'>{phrase}</span>")
        return " ".join(result)


class IOB(Parser):
    def parse(self, lines: list[str]) -> list[IOBRecord]:
        results: list[IOBRecord] = []
        record = IOBRecord()
        for _line in lines:
            line = _line.strip()
            if not line:
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


class IOBJSON(Parser):
    def parse(self, lines: list[str]) -> list[IOBJSONRecord]:
        results: list[IOBJSONRecord] = []
        for item in json.loads("\n".join(lines)):
            results.append(IOBJSONRecord(item))
        return results
