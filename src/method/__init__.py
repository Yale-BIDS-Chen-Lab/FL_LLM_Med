import importlib
import os

import cyy_huggingface_toolbox  # noqa: F401

for entry in os.scandir(os.path.dirname(os.path.abspath(__file__))):
    if not entry.is_dir():
        continue
    if entry.name.startswith(".") or entry.name.startswith("__"):
        continue
    importlib.import_module(f".{entry.name}", "src.method")
