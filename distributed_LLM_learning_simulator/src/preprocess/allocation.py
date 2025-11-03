import collections
import heapq
from dataclasses import dataclass, field

from cyy_preprocessing_pipeline.dataset import IOBRecord, JSONRecord


@dataclass(order=True)
class WorkerCount:
    count: int
    index: int = field(compare=False)


@dataclass(order=True)
class RecordCount:
    count: int
    record: IOBRecord | JSONRecord = field(compare=False)


def get_min_tag(all_records: list[IOBRecord | JSONRecord]) -> str | None:
    assert all_records
    total_counter: collections.Counter[str] = collections.Counter()
    for r in all_records:
        assert isinstance(r, IOBRecord | JSONRecord)
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
    all_records: list[IOBRecord | JSONRecord],
    checked_tag: str,
    allocation: dict,
    split_number: int,
) -> list[IOBRecord | JSONRecord]:
    heap: list = []
    for idx in range(split_number):
        heapq.heappush(heap, WorkerCount(count=0, index=idx))
    assert all_records
    used_records: list[RecordCount] = []
    remain_records = []
    for r in all_records:
        assert isinstance(r, IOBRecord | JSONRecord)
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


def allocate(
    all_records: list[IOBRecord | JSONRecord], allocation: dict, split_number: int
) -> None:
    while True:
        tag = get_min_tag(all_records=all_records)
        if tag is None:
            return
        all_records = allocate_impl(all_records, tag, allocation, split_number)
