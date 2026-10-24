

from pathlib import Path
from collections import Counter
from PIL import Image
import math
import json

# --------------------------------------------------
# BLOCK 1: CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path(r"D:\YOLO\data\raw")

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Keep this mapping consistent with your dataset's data.yaml.
CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}

NUM_CLASSES = len(CLASS_NAMES)

# Warn about unusually small boxes; this is a configurable heuristic.
MIN_BOX_SIZE = 0.005

# Small tolerance for floating-point boundary checks.
EPS = 1e-6


# --------------------------------------------------
# BLOCK 2: FIND FILES AND CHECK IMAGE PAIRS
# --------------------------------------------------

def get_images(folder):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_split(split):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    errors = []
    warnings = []
    class_counts = Counter()

    if not image_dir.is_dir():
        return {
            "errors": [f"Missing image directory: {image_dir}"],
            "warnings": [],
            "images": 0,
            "objects": 0,
            "class_counts": {},
        }

    if not label_dir.is_dir():
        return {
            "errors": [f"Missing label directory: {label_dir}"],
            "warnings": [],
            "images": 0,
            "objects": 0,
            "class_counts": {},
        }

    images = get_images(image_dir)

    # Match files by their relative path and filename stem.
    image_map = {
        p.relative_to(image_dir).with_suffix("").as_posix(): p
        for p in images
    }

    label_files = sorted(label_dir.rglob("*.txt"))
    label_map = {
        p.relative_to(label_dir).with_suffix("").as_posix(): p
        for p in label_files
    }

    if not images:
        errors.append(f"No images found in {image_dir}")

    duplicate_stems = set()
    seen_image_keys = set()

    for image in images:
        key = image.relative_to(image_dir).with_suffix("").as_posix()
        if key in seen_image_keys:
            duplicate_stems.add(key)
        seen_image_keys.add(key)

    for key in sorted(duplicate_stems):
        errors.append(f"Multiple images share the same filename stem: {key}")

    for key in sorted(image_map.keys() - label_map.keys()):
        errors.append(f"Image has no label file: {image_map[key]}")

    for key in sorted(label_map.keys() - image_map.keys()):
        errors.append(f"Label has no matching image: {label_map[key]}")


    # --------------------------------------------------
    # BLOCK 3: VALIDATE IMAGE FILES
    # --------------------------------------------------

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            errors.append(f"Unreadable or corrupt image: {image_path} ({exc})")


    # --------------------------------------------------
    # BLOCK 4: VALIDATE YOLO LABEL CONTENT
    # --------------------------------------------------

    for key, label_path in sorted(label_map.items()):
        if key not in image_map:
            continue

        try:
            content = label_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            errors.append(f"Cannot read label file {label_path}: {exc}")
            continue

        seen_boxes = set()

        for line_number, line in enumerate(content.splitlines(), start=1):
            line = line.strip()

            # Empty label files are valid for images with no objects.
            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label_path}:{line_number}: expected 5 values, "
                    f"found {len(parts)}. Possible wrong format or segmentation label."
                )
                continue

            try:
                class_value = float(parts[0])
                box = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(
                    f"{label_path}:{line_number}: non-numeric label value"
                )
                continue

            values = [class_value] + box

            if not all(math.isfinite(value) for value in values):
                errors.append(
                    f"{label_path}:{line_number}: NaN or infinity found"
                )
                continue

            if not class_value.is_integer():
                errors.append(
                    f"{label_path}:{line_number}: class ID must be an integer"
                )
                continue

            class_id = int(class_value)

            if class_id not in CLASS_NAMES:
                errors.append(
                    f"{label_path}:{line_number}: unknown class ID {class_id}"
                )
                continue

            x, y, width, height = box

            if width <= 0 or height <= 0:
                errors.append(
                    f"{label_path}:{line_number}: box width and height must be positive"
                )
                continue

            if any(v < -EPS or v > 1 + EPS for v in (x, y, width, height)):
                errors.append(
                    f"{label_path}:{line_number}: coordinates or dimensions outside [0, 1]"
                )
                continue

            # Check the actual box edges, not just its center.
            if (
                x - width / 2 < -EPS
                or y - height / 2 < -EPS
                or x + width / 2 > 1 + EPS
                or y + height / 2 > 1 + EPS
            ):
                errors.append(
                    f"{label_path}:{line_number}: bounding box extends outside the image"
                )
                continue

            if width < MIN_BOX_SIZE or height < MIN_BOX_SIZE:
                warnings.append(
                    f"{label_path}:{line_number}: unusually small box "
                    f"(w={width}, h={height})"
                )

            # Catch identical annotation rows within the same file.
            normalized_row = (class_id, x, y, width, height)

            if normalized_row in seen_boxes:
                warnings.append(
                    f"{label_path}:{line_number}: duplicate bounding box"
                )
            else:
                seen_boxes.add(normalized_row)

            class_counts[class_id] += 1


    # --------------------------------------------------
    # BLOCK 5: CHECK CLASS COVERAGE
    # --------------------------------------------------

    missing_classes = [
        CLASS_NAMES[class_id]
        for class_id in CLASS_NAMES
        if class_counts[class_id] == 0
    ]

    if missing_classes:
        warnings.append(
            "Classes with zero valid instances in this split: "
            + ", ".join(missing_classes)
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "images": len(images),
        "objects": sum(class_counts.values()),
        "class_counts": {
            CLASS_NAMES[class_id]: class_counts[class_id]
            for class_id in CLASS_NAMES
        },
    }


# --------------------------------------------------
# BLOCK 6: RUN ALL SPLITS AND SAVE THE REPORT
# --------------------------------------------------

def main():
    report = {
        "dataset": str(DATASET_DIR),
        "splits": {},
    }

    for split in SPLITS:
        print(f"\nValidating {split}...")

        result = check_split(split)
        report["splits"][split] = result

        print(f"Images: {result['images']}")
        print(f"Valid objects counted: {result['objects']}")
        print(f"Errors: {len(result['errors'])}")
        print(f"Warnings: {len(result['warnings'])}")

        for error in result["errors"]:
            print(f"  ERROR: {error}")

        for warning in result["warnings"]:
            print(f"  WARNING: {warning}")

        print("Class counts:")
        for name, count in result["class_counts"].items():
            print(f"  {name}: {count}")

    total_errors = sum(
        len(result["errors"]) for result in report["splits"].values()
    )

    total_warnings = sum(
        len(result["warnings"]) for result in report["splits"].values()
    )

    report_path = DATASET_DIR / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 45)
    print(f"TOTAL ERRORS: {total_errors}")
    print(f"TOTAL WARNINGS: {total_warnings}")
    print(f"Report saved to: {report_path}")

    if total_errors:
        print("RESULT: FAILED — fix errors before training.")
    else:
        print("RESULT: STRUCTURAL CHECKS PASSED — review warnings too.")


if __name__ == "__main__":
    main()
