import shutil
import sys
from collections import Counter
from pathlib import Path

from .config import load_config
from .stats import build_stats, count_classes, write_stats
from .transform import build_id_map, parse_class_names, rewrite_label_lines
from .validate import (check_class_counts, check_id_range, check_pairing,
                       find_images_without_labels, find_orphan_labels,
                       orphan_problems)


def read_lines(path):
    text = path.read_text(encoding="utf-8")
    return [l.strip() for l in text.splitlines() if l.strip()]


def main():
    cfg = load_config()
    raw = Path(cfg["paths"]["raw"])
    final = Path(cfg["paths"]["processed"])
    tmp = final.with_name(final.name + ".tmp")
    exts = {e.lower() for e in cfg["image_extensions"]}
    keep, remove = cfg["keep"], cfg["remove"]

    if final.exists():
        print(f"{final} already exists. Move or delete it first.")
        return 1
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover from a crashed run; always ours

    names = parse_class_names(Path(cfg["paths"]["raw_data_yaml"]).read_text(encoding="utf-8"))
    id_map, drop_ids = build_id_map(names, keep, remove)

    problems = []
    before, after = Counter(), Counter()
    per_split = {}

    for split in cfg["splits"]:
        img_dir, lbl_dir = raw / "images" / split, raw / "labels" / split
        out_img, out_lbl = tmp / "images" / split, tmp / "labels" / split
        out_img.mkdir(parents=True)
        out_lbl.mkdir(parents=True)

        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in exts}
        label_stems = {p.stem for p in lbl_dir.glob("*.txt")}
        problems += orphan_problems(split, find_orphan_labels(images, label_stems),
                                    cfg["orphan_labels"])
        no_label = set(find_images_without_labels(images, label_stems))

        raw_labels, new_labels = {}, {}
        for stem, img in sorted(images.items()):
            if stem in no_label:
                continue
            lines = read_lines(lbl_dir / (stem + ".txt"))
            try:
                new = rewrite_label_lines(lines, id_map, drop_ids)
            except ValueError as e:
                problems.append(f"{split}/{stem}.txt: {e}")
                continue
            raw_labels[stem] = lines
            if not new:
                continue  # only removed classes: drop image and label
            shutil.copy2(img, out_img / img.name)
            (out_lbl / (stem + ".txt")).write_text("\n".join(new) + "\n", encoding="utf-8")
            new_labels[stem] = new

        before += count_classes(raw_labels)
        after += count_classes(new_labels)
        kept_images = {p.stem for p in out_img.iterdir()}
        problems += check_pairing(split, kept_images, set(new_labels))
        problems += check_id_range(split, new_labels, len(keep))
        per_split[split] = {"images": len(images), "kept": len(new_labels),
                            "dropped_only_removed": len(raw_labels) - len(new_labels),
                            "dropped_no_label": len(no_label)}

    problems += check_class_counts(before, after, id_map, keep)

    if problems:
        print("FAILED. processed was NOT created.")
        for p in problems:
            print(" -", p)
        shutil.rmtree(tmp)
        return 1

    yaml_text = (f"path: {final.resolve().as_posix()}\ntrain: images/train\n"
                 f"val: images/val\ntest: images/test\n\nnc: {len(keep)}\nnames:\n")
    yaml_text += "".join(f"  {i}: {n}\n" for i, n in enumerate(keep))
    (tmp / "data.yaml").write_text(yaml_text, encoding="utf-8")
    write_stats(tmp / "stats.json", build_stats(before, after, id_map, keep, per_split))

    tmp.rename(final)  # the only moment processed appears: all checks passed
    print("OK. wrote", final)
    for s, d in per_split.items():
        print(s, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())