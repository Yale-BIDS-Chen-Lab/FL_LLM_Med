import json
import os
from cyy_naive_lib import load_json


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
    for token_info in res["indexes"].values():
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

    assert sentences
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
    for line in sentences:
        print(line)
        pass
    with open(
        input_json.removesuffix(".json") + "_NER.json", "tw", encoding="utf8"
    ) as f:
        json.dump(content, f)


convert_json_to_ner("/home/cyy/10708968_887748326.json")
