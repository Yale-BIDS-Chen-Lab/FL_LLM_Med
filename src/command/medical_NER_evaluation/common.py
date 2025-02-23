def find_tag(token, pre_tokens, pre_tokens_lower, pre_tags) -> str:
    # case sensitive
    try:
        idx = pre_tokens.index(token)
        return pre_tags[idx]
    except ValueError:
        pass
    # case insensitive
    try:
        idx = pre_tokens_lower.index(token.lower())
        return pre_tags[idx]
    except ValueError:
        pass
    # partial match
    for idx, t in enumerate(pre_tokens_lower):
        if t in token.lower() or token.lower() in t:
            return pre_tags[idx]
    return "O"
