import pytest
from collections import Counter

from src.dataprep.stats import count_classes
from src.dataprep.transform import (build_id_map, parse_class_names,
                                    rewrite_label_lines)
from src.dataprep.validate import (check_class_counts, check_id_range,
                                   check_pairing, find_images_without_labels,
                                   find_orphan_labels, orphan_problems)

NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
         5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
         9: "no_gloves", 10: "no_boots"}
KEEP = ["helmet", "gloves", "vest", "boots", "Person"]
REMOVE = ["goggles", "none", "no_helmet", "no_goggle", "no_gloves", "no_boots"]
BOX = "0.5 0.5 0.1 0.1"


def maps():
    return build_id_map(NAMES, KEEP, REMOVE)


def test_parse_class_names():
    assert parse_class_names("names:\n  0: helmet\n  6: Person\n") == {0: "helmet", 6: "Person"}


def test_id_map_matches_original_script():
    id_map, drop = maps()
    assert id_map == {0: 0, 1: 1, 2: 2, 3: 3, 6: 4}
    assert drop == {4, 5, 7, 8, 9, 10}


def test_build_id_map_fails_if_class_unlisted():
    with pytest.raises(ValueError):
        build_id_map(NAMES, KEEP, REMOVE[:-1])


def test_build_id_map_fails_on_duplicate_name():
    with pytest.raises(ValueError):
        build_id_map(NAMES, KEEP + ["helmet"], REMOVE)


def test_only_removed_classes_leaves_nothing():
    id_map, drop = maps()
    assert rewrite_label_lines([f"5 {BOX}", f"7 {BOX}"], id_map, drop) == []


def test_mixed_file_keeps_and_renumbers():
    id_map, drop = maps()
    out = rewrite_label_lines([f"6 {BOX}", f"5 {BOX}", f"0 {BOX}"], id_map, drop)
    assert out == [f"4 {BOX}", f"0 {BOX}"]


def test_unknown_class_id_raises():
    id_map, drop = maps()
    with pytest.raises(ValueError):
        rewrite_label_lines([f"99 {BOX}"], id_map, drop)


def test_wrong_field_count_raises():
    id_map, drop = maps()
    with pytest.raises(ValueError):
        rewrite_label_lines(["0 0.5 0.5"], id_map, drop)


def test_orphans_found():
    assert find_orphan_labels({"a"}, {"a", "b(1)"}) == ["b(1)"]
    assert find_images_without_labels({"a", "c"}, {"a"}) == ["c"]


def test_orphan_strict_fails_lenient_passes():
    assert len(orphan_problems("train", ["x"], "strict")) == 1
    assert orphan_problems("train", ["x"], "lenient") == []
    assert orphan_problems("train", [], "strict") == []


def test_pairing_mismatch_detected():
    assert check_pairing("train", {"a"}, {"a"}) == []
    assert len(check_pairing("train", {"a", "b"}, {"a"})) == 1


def test_id_range_out_of_range_fails():
    assert check_id_range("train", {"a.txt": [f"4 {BOX}"]}, 5) == []
    assert len(check_id_range("train", {"a.txt": [f"5 {BOX}"]}, 5)) == 1


def test_count_classes():
    assert count_classes({"a": [f"0 {BOX}", f"0 {BOX}"], "b": [f"4 {BOX}"]}) == Counter({0: 2, 4: 1})


def test_class_count_mismatch_detected():
    id_map, _ = maps()
    before = Counter({0: 2, 6: 1})
    assert check_class_counts(before, Counter({0: 2, 4: 1}), id_map, KEEP) == []
    assert len(check_class_counts(before, Counter({0: 1, 4: 1}), id_map, KEEP)) == 1