import copy


def match_token(a: str, b: str, strict: bool = False) -> bool:
    if strict:
        return a == b
    a_set = set(a.lower())
    b_set = set(b.lower())
    return len(a_set.intersection(b_set)) / len(a_set.union(b_set)) > 0.9


def match_tokens(
    tokens: list[str], pred_tokens: list[str | tuple[list[str], list[str]]]
) -> list[str]:
    # case sensitive
    if not tokens:
        return []
    token_len = len(tokens)
    tags = []
    while tokens:
        token = tokens[0]
        if not pred_tokens:
            tokens = tokens[1:]
            tags.append("O")
            continue
        has_match = False
        for idx, element in enumerate(pred_tokens):
            if isinstance(element, str):
                pred_token = element
                if match_token(token, pred_token):
                    has_match = True
                    tokens = tokens[1:]
                    tags.append("O")
                    pred_tokens = pred_tokens[idx + 1 :]
                    break
            else:
                phase = element[0]
                if len(tokens) >= len(phase) and all(
                    match_token(tokens[i], phase[i]) for i in range(len(phase))
                ):
                    has_match = True
                    tags += element[1]
                    tokens = tokens[len(phase) :]
                    pred_tokens = pred_tokens[1:]
                    break
        if not has_match:
            tags.append("O")
            tokens = tokens[1:]
    assert len(tags) == token_len
    return tags


def replace_tag(l: list[list[str]], canonical_tags: set[str]) -> list[list[str]]:
    res = []
    assert "unified_class" not in canonical_tags
    for a in l:
        new_tags = copy.deepcopy(a)
        for idx, b in enumerate(new_tags):
            for canonical_tag in canonical_tags:
                if canonical_tag in b:
                    new_tags[idx] = b.replace(canonical_tag, "unified_class")
                    break
        res.append(new_tags)
    return res
