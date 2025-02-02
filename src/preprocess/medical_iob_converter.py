import argparse
import json
from parser import parse_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Convert Data",
    )
    parser.add_argument("--data_dir", help="raw data dir", type=str, required=True)
    parser.add_argument(
        "--output_file", help="output json file", type=str, required=True
    )
    args = parser.parse_args()
    result = parse_dir(args.data_dir, "bio")
    result |= parse_dir(args.data_dir, "iob")
    with open(args.output_file, encoding="utf8") as f:
        json.dump(result, f)
