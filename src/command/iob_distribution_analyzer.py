import argparse
import os
import sys

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)

from preprocess import parse_dir
from preprocess.distribution import token_distribution

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Analyze IBO Distribution",
    )
    parser.add_argument("--data_dir", help="raw data dir", type=str, required=True)
    parser.add_argument("--sub_dirs", help="raw data sub-dir", type=str, default=None)
    args = parser.parse_args()
    data_dir = args.data_dir
    if args.sub_dirs is not None:
        data_dirs = [os.path.join(data_dir, s) for s in args.sub_dirs.split(":")]
    else:
        data_dirs = [data_dir]

    for data_dir in data_dirs:
        assert os.path.isdir(data_dir)
        print("check ", data_dir)
        result = parse_dir(data_dir, "bio")
        result |= parse_dir(args.data_dir, "iob")
        all_records = []
        for records in result.values():
            all_records += records
        assert all_records
        token_distribution(all_records=all_records)
