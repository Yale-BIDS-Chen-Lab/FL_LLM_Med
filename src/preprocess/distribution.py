from collections import Counter

from .iob import IOBRecord


def token_distribution(all_records: list[IOBRecord]) -> dict[str, set[str]]:
    token_and_tags: dict[str, set[str]] = {}
    for record in all_records:
        for token, tag in record.annotated_tokens:
            if token not in token_and_tags:
                token_and_tags[token] = set()
            token_and_tags[token].add(tag.removeprefix("I-").removeprefix("B-"))

    print(token_and_tags)
    return token_and_tags


def tag_distribution(all_records: list[IOBRecord]) -> Counter:
    counter: Counter = Counter()
    for record in all_records:
        counter.update(
            [tag.removeprefix("I-").removeprefix("B-") for tag in record.token_tags]
        )
    return counter
