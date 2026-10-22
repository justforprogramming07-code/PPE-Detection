def find_orphan_labels(image_stems, label_stems):
    """Label files that have no matching image."""
    return sorted(set(label_stems) - set(image_stems))


def find_images_without_labels(image_stems, label_stems):
    """Images that have no label file."""
    return sorted(set(image_stems) - set(label_stems))


def orphan_problems(split, orphans, mode):
    """strict: orphans are problems. lenient: they are tolerated."""
    if mode == "strict" and orphans:
        return [f"{split}: {len(orphans)} label files with no image, e.g. {orphans[:3]}"]
    return []


def check_pairing(split, image_stems, label_stems):
    """After filtering, every kept image must have exactly one label file."""
    if set(image_stems) != set(label_stems):
        return [f"{split}: image/label mismatch after filtering"]
    return []


def check_id_range(split, labels_by_file, n_classes):
    """Every class ID in the output must be 0..n_classes-1."""
    problems = []
    for name, lines in labels_by_file.items():
        for line in lines:
            cls = int(line.split()[0])
            if not 0 <= cls < n_classes:
                problems.append(f"{split}/{name}: id {cls} out of range")
    return problems


def check_class_counts(before, after, id_map, keep):
    """Instances per kept class must be identical before and after."""
    problems = []
    for old, new in id_map.items():
        if before[old] != after[new]:
            problems.append(
                f"count mismatch for {keep[new]}: before={before[old]} after={after[new]}"
            )
    return problems