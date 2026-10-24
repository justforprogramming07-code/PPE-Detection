# src/make_dataset.py
import sys
import json
import shutil
import hashlib
from pathlib import Path
from collections import Counter
import yaml


# =====================================================
# PART 1: small helper functions
# =====================================================

def get_file_fingerprint(file_path):
    # Read the file as raw bytes and turn it into a fingerprint (a long string).
    # If the file changes even a little, the fingerprint changes.
    # We store it in the report to prove which files built this dataset.
    file_bytes = file_path.read_bytes()
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    return fingerprint


def save_text_file(file_path, text):
    # Write text into a file.
    # newline="\n" stops Windows from using a different line ending,
    # so the files come out identical on Windows and on Colab.
    with open(file_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(text)


def read_label_file(label_path):
    # A label file has one object per line, like:
    #   3 0.499 0.801 0.082 0.087
    # The first number is the class id. The rest is the box position.
    # This function returns a list of rows.
    # Each row is a list of two things: [class_id, box_text]
    # Example: [3, "0.499 0.801 0.082 0.087"]

    rows = []

    # If the image has no label file, it simply has no objects.
    if not label_path.exists():
        return rows

    text = label_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for line in lines:
        # Skip empty lines
        if line.strip() == "":
            continue

        parts = line.split()           # "3 0.4 0.8 0.08 0.08" -> ["3","0.4","0.8","0.08","0.08"]
        class_id = int(parts[0])       # the first part, as a number
        box_text = " ".join(parts[1:]) # all the other parts joined back together

        rows.append([class_id, box_text])

    return rows


# =====================================================
# PART 2: the rule (this is the most important function)
# =====================================================

def decide(rows, old_names, remap, drop_class):
    # Look at ONE image and decide two things:
    #   1) should we drop this whole image? (True or False)
    #   2) which label lines should we keep?

    # RULE 1: the image has exactly one label, and that label is goggles.
    # Then we drop the whole image.
    if len(rows) == 1:
        class_id = rows[0][0]
        class_name = old_names[class_id]
        if class_name == drop_class:
            return True, []

    # RULE 2: otherwise we keep the image, but we clean its labels.
    # Go through each label one by one.
    kept_lines = []
    for row in rows:
        class_id = row[0]
        box_text = row[1]

        if class_id in remap:
            # This class is on our keep list.
            # Write it again using its NEW class id.
            new_id = remap[class_id]
            new_line = str(new_id) + " " + box_text
            kept_lines.append(new_line)
        # If the class is not on the keep list (goggles, none, no_helmet...),
        # we do nothing. Not adding it to kept_lines is what removes it.

    return False, kept_lines


# =====================================================
# PART 3: setup functions used by main()
# =====================================================

def read_config(config_path):
    text = Path(config_path).read_text(encoding="utf-8")
    config = yaml.safe_load(text)   # turns the yaml text into a Python dictionary
    return config


def read_old_class_names(source_folder):
    # The raw data.yaml has the class list: {0: 'helmet', 1: 'gloves', ...}
    yaml_path = source_folder / "data.yaml"
    text = yaml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data["names"]


def check_names_exist(names_to_check, old_names):
    # Make sure every class name in our config really exists in the raw data.
    # If we wrote "person" but the raw data says "Person", stop immediately.
    # Crashing now is better than silently deleting every Person label.
    all_raw_names = list(old_names.values())

    for name in names_to_check:
        if name not in all_raw_names:
            raise ValueError("This class is not in the raw data.yaml: " + name)


def build_remap(keep_classes, old_names):
    # Build a translation table: old class id -> new class id.
    # Example: Person is 6 in the raw data, but becomes 4 in the new data.
    remap = {}

    new_id = 0
    for class_name in keep_classes:
        # find the old id of this class name
        for old_id in old_names:
            if old_names[old_id] == class_name:
                remap[old_id] = new_id
        new_id = new_id + 1

    return remap


# =====================================================
# PART 4: the main program
# =====================================================

def main(config_path):
    # ---- Step 1: read our decisions and the raw class list ----
    config = read_config(config_path)

    source_folder = Path(config["source"])      # data/raw
    output_folder = Path(config["output"])      # data/processed/ppe_v1
    keep_classes = config["keep_classes"]       # helmet, gloves, vest, boots, Person
    drop_class = config["drop_single_annotation_class"]   # goggles
    splits = config["splits"]                   # train, val, test

    old_names = read_old_class_names(source_folder)

    # ---- Step 2: safety check on the names ----
    check_names_exist(keep_classes + [drop_class], old_names)

    # ---- Step 3: build the old-id -> new-id table ----
    remap = build_remap(keep_classes, old_names)

    # ---- Step 4: start clean ----
    # If the output folder is left over from an earlier run, delete it.
    # This way the result only depends on the raw data, the config and the code.
    if output_folder.exists():
        shutil.rmtree(output_folder)

    # ---- Step 5: places to collect numbers for the report ----
    count_before = Counter()          # labels per class before cleaning
    count_after = Counter()           # labels per class after cleaning
    images_kept = Counter()           # how many images kept, per split
    images_made_empty = Counter()     # images that lost all their labels, per split
    images_dropped = {}               # names of dropped images, per split
    orphan_labels = {}                # label files with no image, per split

    # ---- Step 6: go through train, then val, then test ----
    for split in splits:

        # make the empty output folders for this split
        (output_folder / "images" / split).mkdir(parents=True)
        (output_folder / "labels" / split).mkdir(parents=True)

        image_folder = source_folder / "images" / split
        label_folder = source_folder / "labels" / split

        images_dropped[split] = []
        orphan_labels[split] = []

        # get all image files, in sorted order
        image_files = []
        for path in sorted(image_folder.iterdir()):
            if path.is_file():
                image_files.append(path)

        # find label files that have no matching image (just for the report)
        image_stems = []
        for image_path in image_files:
            image_stems.append(image_path.stem)   # stem = file name without .jpg

        for label_path in sorted(label_folder.glob("*.txt")):
            if label_path.stem not in image_stems:
                orphan_labels[split].append(label_path.name)

        # ---- Step 7: handle every image, one at a time ----
        for image_path in image_files:

            # the label file has the same name as the image, but ends in .txt
            label_path = label_folder / (image_path.stem + ".txt")

            rows = read_label_file(label_path)

            # count the original labels before we change anything
            for row in rows:
                class_name = old_names[row[0]]
                count_before[class_name] += 1

            # ask the rule what to do with this image
            drop_image, kept_lines = decide(rows, old_names, remap, drop_class)

            if drop_image:
                # write down its name, then skip to the next image.
                # nothing gets copied, so it will not be in the new dataset.
                images_dropped[split].append(image_path.name)
                continue

            # the image had labels, but all of them were removed
            if len(rows) > 0 and len(kept_lines) == 0:
                images_made_empty[split] += 1

            # copy the image into the new dataset
            new_image_path = output_folder / "images" / split / image_path.name
            shutil.copy2(image_path, new_image_path)

            # write the cleaned label file next to it
            if len(kept_lines) > 0:
                label_text = "\n".join(kept_lines) + "\n"
            else:
                label_text = ""   # empty file = background image (no objects)

            new_label_path = output_folder / "labels" / split / (image_path.stem + ".txt")
            save_text_file(new_label_path, label_text)

            images_kept[split] += 1

            # count the labels we kept, using their class NAMES
            for line in kept_lines:
                new_id = int(line.split()[0])
                class_name = keep_classes[new_id]
                count_after[class_name] += 1

    # ---- Step 8: write the new data.yaml ----
    # No absolute "path:" line, so it works on both Windows and Colab.
    # Not verified for your Ultralytics version: a 1-epoch test on Colab will tell.
    new_names = {}
    new_id = 0
    for class_name in keep_classes:
        new_names[new_id] = class_name
        new_id = new_id + 1

    new_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": new_names,
    }
    yaml_text = yaml.safe_dump(new_yaml, sort_keys=False)
    save_text_file(output_folder / "data.yaml", yaml_text)

    # ---- Step 9: checks. If any check fails, the program crashes on purpose ----

    # Check A: the number of images and labels must match in every split
    for split in splits:
        number_of_images = len(list((output_folder / "images" / split).iterdir()))
        number_of_labels = len(list((output_folder / "labels" / split).iterdir()))
        assert number_of_images == number_of_labels, "Image and label counts differ in " + split
        assert number_of_images == images_kept[split], "Kept count is wrong in " + split

    # Check B: every label line must have 5 numbers and a valid class id
    for label_path in (output_folder / "labels").rglob("*.txt"):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, "Line does not have 5 numbers in " + str(label_path)
            class_id = int(parts[0])
            assert class_id >= 0, "Negative class id in " + str(label_path)
            assert class_id < len(keep_classes), "Class id too big in " + str(label_path)

    # Check C: only classes from the keep list may remain
    for class_name in count_after:
        assert class_name in keep_classes, "Unexpected class in output: " + class_name

    # ---- Step 10: write the report ----
    removed_class_counts = {}
    for class_name in count_before:
        if class_name not in keep_classes:
            removed_class_counts[class_name] = count_before[class_name]

    dropped_counts = {}
    orphan_counts = {}
    for split in splits:
        dropped_counts[split] = len(images_dropped[split])
        orphan_counts[split] = len(orphan_labels[split])

    version = config["dataset_version"]

    manifest = {
        "dataset_version": version,
        "config_fingerprint": get_file_fingerprint(Path(config_path)),
        "raw_data_yaml_fingerprint": get_file_fingerprint(source_folder / "data.yaml"),
        "images_kept_per_split": dict(images_kept),
        "images_dropped_per_split": dropped_counts,
        "images_made_label_empty_per_split": dict(images_made_empty),
        "orphan_label_files_per_split": orphan_counts,
        "instances_before": dict(count_before),
        "instances_after": dict(count_after),
        "removed_class_instances": removed_class_counts,
    }

    Path("reports").mkdir(exist_ok=True)
    save_text_file(Path("reports/" + version + "_manifest.json"),
                   json.dumps(manifest, indent=2))
    save_text_file(Path("reports/" + version + "_dropped_images.json"),
                   json.dumps(images_dropped, indent=2))

    print(json.dumps(manifest, indent=2))


# This runs main() only when you run the file directly
# (not when the tests import decide from it).
if __name__ == "__main__":
    main(sys.argv[0])