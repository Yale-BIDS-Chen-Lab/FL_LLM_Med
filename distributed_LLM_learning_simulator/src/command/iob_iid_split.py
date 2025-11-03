import argparse
import collections
import json
import os
import sys

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)

from preprocess import IOBRecord, parse_dir
from preprocess.allocation import allocate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="IBO IID split",
    )
    parser.add_argument("--data_dir", help="raw data dir", type=str, required=True)
    parser.add_argument("--sub_dirs", help="raw data sub-dir", type=str, default=None)
    parser.add_argument(
        "--split_number", help="number to split", type=int, required=True
    )
    parser.add_argument(
        "--output_dir",
        help="output dir for json files",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    split_number = int(args.split_number)
    allocation: dict[int, list] = {i: [] for i in range(split_number)}
    data_dir = args.data_dir
    if args.sub_dirs is not None:
        data_dirs = [os.path.join(data_dir, s) for s in args.sub_dirs.split(":")]
    else:
        data_dirs = [data_dir]

    for data_dir in data_dirs:
        assert os.path.isdir(data_dir)
        print("check ", data_dir)
        result = parse_dir(data_dir, "bio")
        all_records = []
        for records in result.values():
            all_records += records
        assert all_records
        allocate(
            all_records=all_records, allocation=allocation, split_number=split_number
        )
    for records in allocation.values():
        total_counter: collections.Counter[str] = collections.Counter()
        for r in records:
            assert isinstance(r, IOBRecord)
            counter = collections.Counter(
                tag.removeprefix("B-").removeprefix("I-") for tag in r.token_tags
            )
            total_counter += counter
        print([(k, total_counter[k]) for k in sorted(total_counter.keys())])
    output_dir = os.path.join(args.output_dir, f"split_{split_number}")
    os.makedirs(output_dir, exist_ok=True)
    for worker_idx, records in allocation.items():
        with open(
            os.path.join(output_dir, f"worker_{worker_idx}.json"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump([r.to_json() for r in records], f)
