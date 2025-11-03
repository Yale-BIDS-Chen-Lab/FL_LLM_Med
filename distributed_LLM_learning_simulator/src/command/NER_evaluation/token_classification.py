def process_batch(
    ground_tags, prediction, skipped_tags, labels: list[str], batch_res
) -> None:
    targets = batch_res["targets"].reshape(batch_res["batch_size"], -1)
    for sample_targets, sample_prediction, word_ids in zip(
        targets.tolist(),
        batch_res["logits"].argmax(dim=-1).tolist(),
        batch_res["word_ids"].tolist(),
        strict=False,
    ):
        assert len(word_ids) == len(sample_targets)
        assert len(word_ids) == len(sample_prediction)

        previous_word_id: None | int = None
        predicated_tags = []
        tags = []
        for idx, word_id in enumerate(word_ids):
            if word_id == -1:
                word_id = None
            if (
                word_id is not None and word_id != previous_word_id
            ):  # Only label the first token of a given word.
                tags.append(labels[sample_targets[idx]])
                predicated_tags.append(labels[sample_prediction[idx]])
            previous_word_id = word_id
        for skipped_tag in skipped_tags:
            tags = ["O" if skipped_tag in tag else tag for tag in tags]
            predicated_tags = [
                "O" if skipped_tag in tag else tag for tag in predicated_tags
            ]
        ground_tags.append(tags)
        prediction.append(predicated_tags)
