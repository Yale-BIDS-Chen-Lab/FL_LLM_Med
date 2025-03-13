from .iob import IOBRecord


def token_distribution(all_records: list[IOBRecord]) -> dict[str, set[str]]:
    token_and_tags: dict[str, set[str]] = {}
    for record in all_records:
        for token, tag in record.annotated_tokens:
            if tag != "O":
                if token not in token_and_tags:
                    token_and_tags[token] = set()
                token_and_tags[token].add(tag.removeprefix("I-").removeprefix("B-"))

    print(token_and_tags)
    return token_and_tags
    # tag_and_tokens: dict[str, set[str]] = {}
    # for token, tags in token_and_tags.items():
    #     for tag in tags:
    #         if tag not in tag_and_tokens:
    #             tag_and_tokens[tag] = set()
    #         tag_and_tokens[tag].add(token)
