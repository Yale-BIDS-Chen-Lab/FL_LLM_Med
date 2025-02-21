import argparse
from dataclasses import dataclass, field
import collections
import os
import sys

lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(lib_path)
from preprocess import parse_dir
from preprocess import IOBRecord
import heapq


@dataclass(order=True)
class WorkerCount:
    count: int
    index: int = field(compare=False)


@dataclass(order=True)
class RecordCount:
    count: int
    record: IOBRecord = field(compare=False)


def allocate(
    all_records: list[IOBRecord], tag: str, allocation: dict, split_number: int
) -> list[IOBRecord]:
    heap: list = []
    for idx in range(split_number):
        heapq.heappush(heap, WorkerCount(count=0, index=idx))
    assert all_records
    used_records: list[RecordCount] = []
    remain_records = []
    for r in all_records:
        assert isinstance(r, IOBRecord)
        counter = collections.Counter(
            tag.removeprefix("B-").removeprefix("I-") for tag in r.token_tags
        )
        if tag not in counter:
            remain_records.append(r)
        else:
            used_records.append(RecordCount(count=counter[r], record=r))
    assert used_records
    for used_record in sorted(used_records, reverse=True):
        worker_count: WorkerCount = heapq.heappop(heap)
        worker_count.count += used_record.count
        allocation[worker_count.index].append(used_record.record)
        heapq.heappush(heap, worker_count)
    assert allocation
    return remain_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Analyze IBO Distribution",
    )
    parser.add_argument("--data_dir", help="raw data dir", type=str, required=True)
    parser.add_argument(
        "--skip_empty", help="skip empty lines", type=bool, default=True
    )
    parser.add_argument(
        "--split_number", help="number to split", type=int, required=True
    )
    args = parser.parse_args()
    os.environ["SKIP_EMPTY_LINE"] = str(int(args.skip_empty))
    split_number = int(args.split_number)

    result = parse_dir(args.data_dir, "bio")
    result |= parse_dir(args.data_dir, "iob")
    all_records = []
    for records in result.values():
        all_records += records
    assert all_records
    total_counter = collections.Counter()
    for r in all_records:
        assert isinstance(r, IOBRecord)
        counter = collections.Counter(
            tag.removeprefix("B-").removeprefix("I-") for tag in r.token_tags
        )
        total_counter += counter
    counts = sorted(total_counter.values())
    allocation = {i: [] for i in range(split_number)}
    for sorted_count in counts:
        for tag, count in total_counter.items():
            if tag == "O":
                continue
            if count == sorted_count:
                print("allocate ", tag)
                all_records = allocate(all_records, tag, allocation, split_number)
    for idx, records in allocation.items():
        total_counter = collections.Counter()
        for r in records:
            assert isinstance(r, IOBRecord)
            counter = collections.Counter(
                tag.removeprefix("B-").removeprefix("I-") for tag in r.token_tags
            )
            total_counter += counter
        print([(k, total_counter[k]) for k in sorted(total_counter.keys())])
