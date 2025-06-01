import os
import sys

src_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, src_path)


from .util import get_tester, get_model


def get_output(
    session_dir: str, data_file: str, zero_shot: bool, worker_index: int | None = None
) -> dict:
    tester = get_tester(session_dir=session_dir, data_file=data_file)

    get_model(
        tester=tester,
        session_dir=session_dir,
        zero_shot=zero_shot,
        worker_index=worker_index,
    )
    return tester.get_sample_output()
