import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Add missing column to json files",
    )
    parser.add_argument("--file", help="raw data file", type=str, required=True)
    args = parser.parse_args()
    assert os.path.isfile(args.file)
    res = None
    print(args.file)
    with open(args.file, encoding="utf8") as f:
        res = json.load(f)
        assert isinstance(res, list)
        for a in res:
            assert isinstance(a, dict)
            for k in ("input", "output", "html"):
                if k not in a:
                    a[k] = ""
    with open(args.file, "w", encoding="utf8") as f:
        json.dump(res, f)
