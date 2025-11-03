import argparse
import json
import os
import sys
from collections import Counter

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)

from preprocess.iob import IOBRecord, JSONRecord

from preprocess import parse_dir, parse_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Analyze IBO Distribution",
    )
    parser.add_argument("--data_dir", help="raw data dir", type=str, default=None)
    parser.add_argument("--sub_dirs", help="raw data sub-dir", type=str, default=None)
    parser.add_argument("--file", help="json file", type=str, default=None)
    parser.add_argument("--count", help="count phrases", type=bool, default=True)
    args = parser.parse_args()
    all_records: list[IOBRecord | JSONRecord] = []
    if args.file is not None:
        print("check ", args.file)
        all_records += parse_file(args.file)
    else:
        data_dir = args.data_dir
        if args.sub_dirs is not None:
            data_dirs = [os.path.join(data_dir, s) for s in args.sub_dirs.split(":")]
        else:
            data_dirs = [data_dir]
        result: dict = {}
        for data_dir in data_dirs:
            assert os.path.isdir(data_dir)
            print("check ", data_dir)
            result |= parse_dir(data_dir, "bio")
            result |= parse_dir(args.data_dir, "iob")
        for records in result.values():
            all_records += records
    assert all_records
    total_distribution: None | dict[str, Counter] = None
    for record in all_records:
        if isinstance(record, JSONRecord):
            record = IOBRecord(
                tokens=record.to_json()["tokens"], tags=record.to_json()["tags"]
            )
        record_result = record.get_tag_distribution()
        if total_distribution is None:
            total_distribution = record_result
        else:
            for k, v in record_result.items():
                if k not in total_distribution:
                    total_distribution[k] = v
                else:
                    total_distribution[k] = total_distribution[k] + v
    assert total_distribution is not None
    if args.count:
        total_counter = {k: v.total() for k, v in total_distribution.items()}
        print("tag counts", json.dumps(total_counter, indent=4))
    else:
        print("phrase distribution", total_distribution)
