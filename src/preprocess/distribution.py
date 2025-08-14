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


def phrase_distribution(all_records: list[IOBRecord]) -> dict[str, set[str]]:
    phrase_and_tags: dict[str, set[str]] = {}
    for record in all_records:
        for phrase, tag in record.annotated_phrases:
            if phrase not in phrase_and_tags:
                phrase_and_tags[phrase] = set()
            phrase_and_tags[phrase].add(tag.removeprefix("I-").removeprefix("B-"))

    print(phrase_and_tags)
    return phrase_and_tags
