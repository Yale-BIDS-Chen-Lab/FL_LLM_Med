import os
from dataclasses import dataclass
from typing import Any

from cyy_torch_toolbox import TextDatasetCollection, Transform
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
        lines = data["input"].splitlines()
        remain_lines = []
        for line in lines:
            prefix = 'Use <span class="'
            if line.startswith(prefix):
                tmp_line = line.removeprefix(prefix)
                idx = tmp_line.indexOf('"')
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
        lines = data["input"].splitlines()
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
        data["input"] = "\n".join(input_lines)
        return data


class DatapipelineMixin(ExecutorProtocol):
    def set_prompt(self, dc: TextDatasetCollection, for_evaluation: bool) -> None:
        prompt_file = self.config.dc_config.dataset_kwargs["prompt_file"]
        if for_evaluation:
            prompt_file = self.config.dc_config.dataset_kwargs["evaluation_prompt_file"]
        prompt_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "prompt", prompt_file
        )
        with open(prompt_file, encoding="utf8") as f:
            prompt = f.read()
            dc.set_prompt(prompt)
        if self.config.dc_config.dataset_kwargs.get(
            "tailor_prompt_for_training", False
        ):
            dc.append_post_prompt_text_transform(transform=MedicalREPromptReduction())
