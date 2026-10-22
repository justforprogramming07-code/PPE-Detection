import json
from collections import Counter


def count_classes(labels_by_file):
    """{filename: [label lines]} -> Counter of class id."""
    c = Counter()
    for lines in labels_by_file.values():
        for line in lines:
            c[int(line.split()[0])] += 1
    return c


def build_stats(before, after, id_map, keep, per_split):
    """per_split example: {"train": {"images": 1132, "kept": 1128, "dropped": 4}}"""
    return {
        "classes": {
            keep[new]: {"old_id": old, "new_id": new,
                        "before": before[old], "after": after[new]}
            for old, new in id_map.items()
        },
        "splits": per_split,
    }


def write_stats(path, stats):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)