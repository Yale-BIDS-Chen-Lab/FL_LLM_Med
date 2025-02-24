import transformers
from bs4 import BeautifulSoup as bs
from bs4 import Tag


def html2bio(
    html: str, entities: list[str], tokenizer: transformers.PreTrainedTokenizerFast
) -> tuple[list[list[str]], list[str]]:
    tokens: list[list[str]] = []
    tags: list[str] = []
    assert html
    assert entities
    entities_lower = [e.lower() for e in entities]

    # Parse HTML using BeautifulSoup
    soup = bs(html, "html.parser")

    for child in soup.find_all("span"):
        assert isinstance(child, Tag)
        classes = child.attrs.get("class", [])
        if not classes:
            continue
        tag: str | None = None
        for c in classes:
            if c.lower() in entities_lower:
                tag = entities[entities_lower.index(c.lower())]
                break
        if tag is None:
            continue
        words = tokenizer.tokenize(child.get_text())
        if not words:
            continue
        tokens.append(words)
        tags.append(f"B-{tag}")
        for _ in range(len(words) - 1):
            tags.append(f"I-{tag}")
    return tokens, tags
