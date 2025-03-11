from collections.abc import Iterable

import bs4


def tokenize(txt: str) -> list[str]:
    return txt.strip().split(" ")


def html2bio(
    html: str,
    canonical_tags: Iterable[str],
) -> list[tuple[list[str], list[str]] | str]:
    tokens: list[tuple[list[str], list[str]] | str] = []
    assert html
    canonical_tags = list(canonical_tags)
    assert canonical_tags
    canonical_tag_lower = [e.lower() for e in canonical_tags]

    # Parse HTML using BeautifulSoup
    soup = bs4.BeautifulSoup(html, "html.parser")
    for child in soup:
        match child:
            case bs4.element.NavigableString():
                tokens += tokenize(child.get_text())
            case bs4.element.Tag():
                words = tokenize(child.get_text())
                if not words:
                    continue
                if child.name.lower() != "span":
                    tokens += words
                    continue
                classes = child.attrs.get("class", [])
                if isinstance(classes, str):
                    classes = [classes]
                if not classes:
                    tokens += words
                    continue
                tag: str | None = None
                for c in classes:
                    if c.lower() in canonical_tag_lower:
                        tag = canonical_tags[canonical_tag_lower.index(c.lower())]
                        break
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
