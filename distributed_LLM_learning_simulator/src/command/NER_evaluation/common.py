import copy


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
