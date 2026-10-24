"""Filter unwanted classes out of the YOLO PPE dataset, split by split.

Rules:
- A label file with no kept instances (only removed classes, or empty) is
  dropped, and its image is not copied.
- A mixed file keeps its wanted lines; unwanted lines are removed.
- Kept IDs are renumbered with ONE fixed table for the whole dataset.
- A kept label with no image is still written (counted as images_missing).
- Nothing moves between train/val/test. Raw is never modified.
"""
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import yaml

log = logging.getLogger("data_filtering")
ROOT = Path(__file__).resolve().parents[2]  # D:\YOLO (file is src\data\x.py)


def load_params(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_id_map(keep, remove, names):
    """One fixed table: old id -> new id, in the order of `keep`."""
    if set(keep) & set(remove):
        raise ValueError("An ID is in both keep and remove.")
    if set(keep) | set(remove) != set(names):
        raise ValueError("keep + remove must cover exactly the IDs in names.")
    return {old: new for new, old in enumerate(keep)}


def filter_label_lines(lines, id_map, remove):
    """Lines of ONE label file in, wanted + renumbered lines out."""
    kept = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected 5 numbers, got: {line!r}")
        old = int(parts[0])
        if old in id_map:
            kept.append(" ".join([str(id_map[old])] + parts[1:]))
        elif old not in remove:
            raise ValueError(f"Unknown class id {old}: {line!r}")
    return kept


def find_image(images_dir, stem, extensions):
    for ext in extensions:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def process_split(raw, out, split, id_map, inv_map, remove, names, extensions):
    labels_in = raw / "labels" / split
    images_in = raw / "images" / split
    (out / "labels" / split).mkdir(parents=True)
    (out / "images" / split).mkdir(parents=True)

    stats, before, after = Counter(), Counter(), Counter()
    for label_file in sorted(labels_in.glob("*.txt")):
        stats["label_files_seen"] += 1
        lines = [
            l.strip()
            for l in label_file.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        kept = filter_label_lines(lines, id_map, remove)

        before.update(names[int(l.split()[0])] for l in lines)
        stats["instances_dropped"] += len(lines) - len(kept)

        if not kept:
            stats["label_files_dropped"] += 1
            continue

        after.update(names[inv_map[int(l.split()[0])]] for l in kept)
        stats["label_files_kept"] += 1
        (out / "labels" / split / label_file.name).write_text(
            "\n".join(kept) + "\n", encoding="utf-8"
        )

        image = find_image(images_in, label_file.stem, extensions)
        if image is None:
            stats["images_missing"] += 1
        else:
            shutil.copy2(image, out / "images" / split / image.name)
            stats["images_copied"] += 1

    return {
        "counts": dict(stats),
        "instances_before_by_class": dict(before),
        "instances_after_by_class": dict(after),
    }


def run(params, root):
    raw = root / params["paths"]["raw"]
    out = root / params["paths"]["processed"]
    report_path = root / params["paths"]["report"]
    names = params["classes"]["names"]
    keep = params["classes"]["keep"]
    remove = params["classes"]["remove"]
    splits = params["splits"]
    extensions = params["image_extensions"]

    id_map = build_id_map(keep, remove, names)
    inv_map = {new: old for old, new in id_map.items()}

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty. Refusing to overwrite.")
    for split in splits:  # check all inputs before creating anything
        if not (raw / "labels" / split).is_dir():
            raise FileNotFoundError(f"Missing {raw / 'labels' / split}")

    report = {"splits": {}}
    total = {"counts": Counter(), "before": Counter(), "after": Counter()}
    for split in splits:
        res = process_split(raw, out, split, id_map, inv_map, remove, names, extensions)
        report["splits"][split] = res
        total["counts"].update(res["counts"])
        total["before"].update(res["instances_before_by_class"])
        total["after"].update(res["instances_after_by_class"])
        log.info("%s: %s", split, res["counts"])

    report["total"] = {
        "counts": dict(total["counts"]),
        "instances_before_by_class": dict(total["before"]),
        "instances_after_by_class": dict(total["after"]),
        "id_map_old_to_new": {f"{o} ({names[o]})": n for o, n in id_map.items()},
    }

    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {new: names[old] for old, new in id_map.items()},
    }
    (out / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    params = load_params(ROOT / "config" / "params.yaml")
    run(params, ROOT)


if __name__ == "__main__":
    main()