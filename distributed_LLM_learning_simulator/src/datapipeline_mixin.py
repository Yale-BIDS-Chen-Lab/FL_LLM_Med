import os
from dataclasses import dataclass
from typing import Any

from cyy_preprocessing_pipeline import Transform
from cyy_torch_toolbox import TextDatasetCollection
from distributed_learning_simulation import ExecutorProtocol


@dataclass(kw_only=True)
class MedicalREPromptReduction(Transform):
    def __init__(self, name: str = "reduce_prompt") -> None:
        super().__init__(name=name, fun=self.reduce_prompt)

    @classmethod
    def reduce_prompt(cls, data: Any) -> Any:
        assert isinstance(data, dict)
        assert "input" in data
        tags: set[str] = set()
        input_text: str = data["input"]
        lines: list[str] = input_text.splitlines()
        remain_lines = []
        for line in lines:
            prefix = 'Use <span class="'
            if line.startswith(prefix):
                tmp_line = line.removeprefix(prefix)
                idx = tmp_line.index('"')
                assert idx > 0
                tags.add(tmp_line[:idx])
            else:
                remain_lines.append(line)
        assert tags
        unused_tags = set()
        for tag in tags:
            has_tag = any(tag in line for line in remain_lines)
            if not has_tag:
                unused_tags.add(tag)
        # for tag in ("treatment", "drug", "problem", "test"):
        #     if tag in unused_tags:
        #         unused_tags.remove(tag)
        lines = input_text.splitlines()
        input_lines = []
        for line in lines:
            keep_line = True
            for tag in unused_tags:
                prefix = f'Use <span class="{tag}'
                if line.startswith(prefix):
                    keep_line = False
                    break
            if keep_line:
                input_lines.append(line)
        assert input_lines
        # guide_idx = -1
        # for idx, line in enumerate(input_lines):
        #     if "Modifier Entity Markup Guide" in line.strip():
        #         guide_idx = idx
        #         break
        # if guide_idx >= 0 and not any(
        #     line.startswith("Use <span class=") for line in input_lines[guide_idx:]
        # ):
        #     input_lines.pop(guide_idx)
        data["input"] = "\n".join(input_lines)
        return data


class DatapipelineMixin(ExecutorProtocol):
    def get_prompt_file(self, for_evaluation: bool) -> str:
        key = "evaluation_prompt_file" if for_evaluation else "prompt_file"
        prompt_file = self.config.dc_config.dataset_kwargs.get(key)
        assert isinstance(prompt_file, str)
        prompt_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "prompt", prompt_file
        )
        assert os.path.isfile(prompt_file), prompt_file
        return prompt_file

    def set_prompt(self, dc: TextDatasetCollection, for_evaluation: bool) -> None:
        prompt_file = self.get_prompt_file(for_evaluation=for_evaluation)
        with open(prompt_file, encoding="utf8") as f:
            prompt = f.read()
            dc.set_prompt(prompt)
        if not for_evaluation and self.config.dc_config.dataset_kwargs.get(
            "tailor_prompt_for_training", False
        ):
            dc.append_post_prompt_text_transform(transform=MedicalREPromptReduction())
