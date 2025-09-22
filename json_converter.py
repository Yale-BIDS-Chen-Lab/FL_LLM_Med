import json
from cyy_naive_lib.fs.path import list_files_by_suffixes

from cyy_naive_lib import load_json, save_json


def convert_json_to_ner(input_json):
    res = load_json(input_json)
    content: str = res["content"]
    ner_annotations = {}
    for token_info in res["indexes"].values():
        assert isinstance(token_info, dict)
        if "Entity" in token_info:
            entity = token_info["Entity"]
            assert isinstance(entity, list)
            for e in entity:
                begin = e["begin"]
                end = e["end"]
                semantic = e["semantic"].lower()
                if semantic in ("problem", "treatment", "test", "drug"):
                    assert begin not in ner_annotations
                    ner_annotations[begin] = {"ner": (begin, end, semantic)}
    sentences = []
    sentences_begin = []
    sentences_end = []
    sentence_tokens = []
    sentence_tags = []
    last_tag: str | None = None
    last_tag_end: int | None = None
    for key, token_info in res["indexes"].items():
        assert isinstance(token_info, dict)
        if "Sentence" in token_info:
            sentence = token_info["Sentence"]
            assert isinstance(sentence, list)
            assert len(sentence) == 1
            e = sentence[0]
            begin = e["begin"]
            end = e["end"]
            sentence = content[begin:end]
            assert sentence
            sentences.append(sentence)
            sentences_begin.append(begin)
            sentences_end.append(end)
            sentence_tokens.append([])
            sentence_tags.append([])
        if "Token" in token_info:
            assert len(token_info["Token"]) == 1
            token_begin = token_info["Token"][0]["begin"]
            token_end = token_info["Token"][0]["end"]
            assert token_begin >= sentences_begin[-1] and token_end <= sentences_end[-1]
            sentence_tokens[-1].append(content[token_begin:token_end])
            if last_tag_end is not None and token_end <= last_tag_end + 1:
                sentence_tags[-1].append(f"I-{last_tag}")
            else:
                sentence_tags[-1].append("O")
        if "Entity" in token_info:
            entity = token_info["Entity"]
            assert isinstance(entity, list)
            # assert len(entity)==1,key
            for e in entity:
                begin = e["begin"]
                end = e["end"]
                semantic = e["semantic"].lower()
                if semantic in ("problem", "treatment", "test", "drug"):
                    assert "Token" in token_info, key
                    # assert begin == token_begin, key
                    if begin != token_begin:
                        print("skip annotation", key)
                        continue
                    ner_annotations[begin] = {"ner": (begin, end, semantic)}
                    last_tag_end = end
                    last_tag = semantic
                    sentence_tags[-1][-1] = f"B-{last_tag}"

    assert sentences
    assert len(sentences) == len(sentence_tokens)
    for t in sentence_tokens:
        assert t
    sentence_idx = len(sentences) - 1
    last_sentence_begin = -1
    last_sentence_end = -1
    for k in sorted(list(ner_annotations.keys()), reverse=True):
        annotation = ner_annotations[k]

        if "ner" in annotation:
            begin, end, semantic = annotation["ner"]
            while True:
                last_sentence_begin = sentences_begin[sentence_idx]
                last_sentence_end = sentences_end[sentence_idx]
                if begin < last_sentence_begin:
                    sentence_idx -= 1
                    assert sentence_idx >= 0
                else:
                    break
            assert begin >= last_sentence_begin and end <= last_sentence_end + 1
            begin -= last_sentence_begin
            end -= last_sentence_begin
            sentence = sentences[sentence_idx]
            sentence = (
                sentence[:begin]
                + f'<span class="{semantic}">'
                + sentence[begin:end]
                + "</span>"
                + sentence[end:]
            )
            sentences[sentence_idx] = sentence
    res = []
    for sentence, tags, tokens in zip(
        sentences, sentence_tags, sentence_tokens, strict=False
    ):
        assert isinstance(tags, list)
        tags_set = set(tags)
        if len(tags_set) == 1 and "O" in tags_set:
            continue
        res.append({"tokens": tokens, "tags": tags, "html": sentence})
    return res


total_result = []
for json_file in list_files_by_suffixes(dir_to_search="xxx", suffixes=".json"):
    total_result += convert_json_to_ner(json_file)
assert total_result
save_json(total_result, "all_NER.json")
