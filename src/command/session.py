import os


def get_config(session_dir: str | None = None):
    model_dir = os.getenv("SESSION_DIR", None)
    assert model_dir is not None
