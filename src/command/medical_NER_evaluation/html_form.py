
import transformers
from bs4 import BeautifulSoup as bs
from bs4 import Tag
from bs4.element import NavigableString


def html2bio(
    html: str, entities: list[str], tokenizer: transformers.PreTrainedTokenizerFast
) -> tuple:
    tokens = []
    tags = []
    assert html
    assert entities

    # Parse HTML using BeautifulSoup
    soup = bs(html, "html.parser")

    # Extract text under 'p' tags and convert to BIO format
    for child in soup.children:
        if isinstance(child, NavigableString):
            child_tokens = tokenizer.tokenize(child.get_text())
            child_tags = ["O"] * len(child_tokens)
            tokens += child_tokens
            tags += child_tags

    for child in soup.find_all("span"):
        assert isinstance(child, Tag)
        if "class" not in child.attrs:
            continue
        # assert len(child.attrs["class"]) == 1
        entity = child.attrs["class"][0]
        words = tokenizer.tokenize(child.get_text())
        tokens += words
        if len(words) != 0:
            if entity != "O" and entity in entities:
                tags.append(f"B-{entity}")
                for _ in range(len(words) - 1):
                    tags.append(f"I-{entity}")
            else:
                child_tags = ["O"] * len(words)
                tags += child_tags
    assert len(tokens) == len(tags)
    return tokens, tags
