from collections.abc import Iterable

import bs4


def tokenize(txt: str) -> list[str]:
    for mark in ":;,.":
        if mark in txt:
            txt = txt.replace(mark, f" {mark} ")
    lines = txt.strip().split(" ")
    return [line for line in lines if line]


def html2bio(
    html: str,
    canonical_tags: Iterable[str] | None = None,
) -> list[tuple[list[str], list[str]] | str]:
    tokens: list[tuple[list[str], list[str]] | str] = []
    if not html:
        return []
    canonical_tag_lower: list[str] | None = None
    if canonical_tags is not None:
        canonical_tags = list(canonical_tags)
        assert canonical_tags
        canonical_tag_lower = [e.lower() for e in canonical_tags]

    # Parse HTML using BeautifulSoup
    soup = bs4.BeautifulSoup(html, "html.parser")
    last_tag_text = None
    for child in soup:
        match child:
            case bs4.element.NavigableString():
                if child.get_text() == last_tag_text:
                    last_tag_text = None
                    continue
                tokens += tokenize(child.get_text())
            case bs4.element.Tag():
                last_tag_text = child.get_text()
                words = tokenize(last_tag_text)
                if not words:
                    continue
                if child.name.lower() != "span":
                    tokens += words
                    continue
                classes: str | Iterable[str] = child.attrs.get("class", [])
                classes = [classes] if isinstance(classes, str) else list(classes)
                if not classes:
                    tokens += words
                    continue
                tag: str | None = None
                if canonical_tags is not None:
                    assert canonical_tag_lower is not None
                    for c in classes:
                        if c.lower() in canonical_tag_lower:
                            tag = canonical_tags[canonical_tag_lower.index(c.lower())]
                            break
                else:
                    assert isinstance(classes, list)
                    assert len(classes) == 1
                    tag = classes[0]
                if tag is None:
                    tokens += words
                    continue
                tags = []
                tags.append(f"B-{tag}")
                for _ in range(len(words) - 1):
                    tags.append(f"I-{tag}")
                tokens.append((words, tags))
            case _:
                raise NotImplementedError(child)
    return tokens
