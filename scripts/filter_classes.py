import re
import shutil
from pathlib import Path
from collections import Counter

DATA = Path(r"C:\yolo\clone\PPE-Detection\data")
RAW = DATA / "raw"
OUT = DATA / "processed"
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg"}  # compared in lowercase

A = [0, 1, 2, 3, 6]          # keep: helmet, gloves, vest, boots, Person
B = [4, 5, 7, 8, 9, 10]      # remove: goggles, none, no_helmet, no_goggle, no_gloves, no_boots
MAP = {old: new for new, old in enumerate(A)}  # {0:0, 1:1, 2:2, 3:3, 6:4}

# Read class names from the original data.yaml (inside raw) and verify them
old_names = {}
for line in (RAW / "data.yaml").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^\s+(\d+):\s*(\S+)\s*$", line)
    if m:
        old_names[int(m.group(1))] = m.group(2)
expected = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
            5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
            9: "no_gloves", 10: "no_boots"}
if old_names != expected:
    raise SystemExit(f"data.yaml names do not match what I expect:\n{old_names}")
NEW_NAMES = [old_names[o] for o in A]


def get_class_id(line):
    """Return only the class ID of one label line."""
    return int(line.split()[0])


def read_lines(p):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f"{OUT} is not empty. Empty it first so we start clean.")
OUT.mkdir(exist_ok=True)

before, after = Counter(), Counter()
problems = []

for split in SPLITS:
    img_dir = RAW / "images" / split
    lbl_dir = RAW / "labels" / split
    out_img = OUT / "images" / split
    out_lbl = OUT / "labels" / split
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    n_img = n_kept = n_dropped_b = n_dropped_nolabel = 0
    img_stems = set()

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        n_img += 1
        img_stems.add(img.stem)
        lbl = lbl_dir / (img.stem + ".txt")

        if not lbl.exists():
            n_dropped_nolabel += 1
            continue

        lines = read_lines(lbl)
        for l in lines:
            c = get_class_id(l)
            if c in A:
                before[c] += 1
            elif c not in B:
                problems.append(f"{lbl}: unknown class {c}")

        new_lines = []
        for l in lines:
            c = get_class_id(l)
            if c in MAP:
                parts = l.split()
                parts[0] = str(MAP[c])
                new_lines.append(" ".join(parts))

        if not new_lines:  # empty label, or only B classes
            n_dropped_b += 1
            continue

        shutil.copy2(img, out_img / img.name)
        (out_lbl / lbl.name).write_text("\n".join(new_lines) + "\n")
        n_kept += 1

    orphans = [p.name for p in lbl_dir.glob("*.txt") if p.stem not in img_stems]
    if orphans:
        problems.append(f"{split}: {len(orphans)} label files with no image")

    print(f"{split}: images={n_img} kept={n_kept} "
          f"dropped(only B or empty)={n_dropped_b} dropped(no label file)={n_dropped_nolabel}")

# New data.yaml (absolute path: change it on Colab)
yaml_text = (f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
             f"test: images/test\n\nnc: {len(A)}\nnames:\n")
yaml_text += "".join(f"  {i}: {n}\n" for i, n in enumerate(NEW_NAMES))
(OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

# ---- Checks ----
print("\nCHECKS")
for split in SPLITS:
    imgs = {p.stem for p in (OUT / "images" / split).iterdir()}
    lbls = {p.stem for p in (OUT / "labels" / split).glob("*.txt")}
    if imgs != lbls:
        problems.append(f"{split}: image/label mismatch")
    for p in (OUT / "labels" / split).glob("*.txt"):
        for l in read_lines(p):
            c = get_class_id(l)
            after[c] += 1
            if not 0 <= c < len(A):
                problems.append(f"{p}: id {c} out of range")

for old, new in MAP.items():
    ok = before[old] == after[new]
    print(f"  {NEW_NAMES[new]:7s} old={old} new={new} before={before[old]} after={after[new]} {'OK' if ok else 'MISMATCH'}")
    if not ok:
        problems.append(f"count mismatch for {NEW_NAMES[new]}")

print("\nPROBLEMS:", "none" if not problems else "")
for p in problems:
    print(" -", p)