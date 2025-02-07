import argparse
import json

import os
import sys

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)
from preprocess import parse_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Convert IBO Data",
    )
    parser.add_argument("--data_dir", help="raw data dir", type=str, required=True)
    parser.add_argument(
        "--output_file", help="output json file", type=str, required=True
    )
    args = parser.parse_args()
    result = parse_dir(args.data_dir, "bio")
    result |= parse_dir(args.data_dir, "iob")
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(result, f)
