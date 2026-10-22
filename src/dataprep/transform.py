import yaml


def parse_class_names(yaml_text):
    """data.yaml text -> {id: name}."""
    names = yaml.safe_load(yaml_text)["names"]
    return {int(k): str(v) for k, v in names.items()}


def build_id_map(names, keep, remove):
    """Return (id_map, drop_ids). Fails unless keep+remove covers every
    class name exactly once. New IDs follow the order of `keep`."""
    all_listed = list(keep) + list(remove)
    if len(all_listed) != len(set(all_listed)):
        raise ValueError("a class name appears twice in keep/remove")
    if set(all_listed) != set(names.values()):
        missing = set(names.values()) - set(all_listed)
        extra = set(all_listed) - set(names.values())
        raise ValueError(f"keep+remove mismatch. missing={missing} extra={extra}")
    name_to_old = {n: i for i, n in names.items()}
    id_map = {name_to_old[n]: new for new, n in enumerate(keep)}
    drop_ids = {name_to_old[n] for n in remove}
    return id_map, drop_ids


def rewrite_label_lines(lines, id_map, drop_ids):
    """Drop removed-class lines, renumber kept ones. Returns new lines
    (empty list means the file has nothing left to train on)."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
        cls = int(parts[0])
        if cls in id_map:
            parts[0] = str(id_map[cls])
            out.append(" ".join(parts))
        elif cls not in drop_ids:
            raise ValueError(f"unknown class id {cls}")
    return out