import argparse
import json
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


def get_min_tag(all_records: list[IOBRecord]) -> str | None:
    assert all_records
    total_counter = collections.Counter()
    for r in all_records:
        assert isinstance(r, IOBRecord)
        counter = collections.Counter(
            tag.removeprefix("B-").removeprefix("I-") for tag in r.token_tags
        )
        total_counter += counter
    total_counter.pop("O")
    if not total_counter:
        return None
    counts = sorted(list(total_counter.items()), key=lambda a: a[1])
    return counts[0][0]


def allocate_impl(
    all_records: list[IOBRecord], checked_tag: str, allocation: dict, split_number: int
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
        if counter[checked_tag] == 0:
            remain_records.append(r)
        else:
            used_records.append(RecordCount(count=counter[checked_tag], record=r))
    assert used_records
    for used_record in sorted(used_records, reverse=True):
        worker_count: WorkerCount = heapq.heappop(heap)
        worker_count.count += used_record.count
        allocation[worker_count.index].append(used_record.record)
        heapq.heappush(heap, worker_count)
        assert len(heap) == split_number
    assert allocation
    return remain_records


def allocate(all_records: list[IOBRecord], allocation: dict, split_number: int) -> None:
    while True:
        tag = get_min_tag(all_records=all_records)
        if tag is None:
            return
        all_records = allocate_impl(all_records, tag, allocation, split_number)


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
    parser.add_argument(
        "--output_dir", help="output dir for json files", type=str, required=True
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
    allocation = {i: [] for i in range(split_number)}
    allocate(all_records=all_records, allocation=allocation, split_number=split_number)
    allocated_cnt = 0
    for records in allocation.values():
        total_counter = collections.Counter()
        allocated_cnt += len(records)

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
