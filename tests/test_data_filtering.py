import json

import pytest
import yaml

from src.data import data_filtering as df

NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
         5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
         9: "no_gloves", 10: "no_boots"}
KEEP = [0, 1, 2, 3, 6]
REMOVE = [4, 5, 7, 8, 9, 10]
ID_MAP = df.build_id_map(KEEP, REMOVE, NAMES)


def test_id_map_is_fixed_table():
    assert ID_MAP == {0: 0, 1: 1, 2: 2, 3: 3, 6: 4}


def test_id_map_rejects_overlap():
    with pytest.raises(ValueError):
        df.build_id_map([0, 1], [1, 2], {0: "a", 1: "b", 2: "c"})


def test_mixed_file_keeps_wanted_and_renumbers():
    lines = ["6 0.46 0.48 0.35 0.87", "4 0.1 0.1 0.1 0.1"]
    assert df.filter_label_lines(lines, ID_MAP, REMOVE) == ["4 0.46 0.48 0.35 0.87"]


def test_only_removed_class_gives_nothing():
    assert df.filter_label_lines(["5 0.1 0.1 0.1 0.1"], ID_MAP, REMOVE) == []


def test_empty_file_gives_nothing():
    assert df.filter_label_lines([], ID_MAP, REMOVE) == []


def test_unknown_class_id_raises():
    with pytest.raises(ValueError):
        df.filter_label_lines(["99 0.1 0.1 0.1 0.1"], ID_MAP, REMOVE)


def test_malformed_line_raises():
    with pytest.raises(ValueError):
        df.filter_label_lines(["1 0.1 0.1 0.1"], ID_MAP, REMOVE)


def _params():
    return {
        "paths": {"raw": "data/raw", "processed": "data/processed",
                  "report": "reports/filter_report.json"},
        "splits": ["train", "val", "test"],
        "image_extensions": [".jpg", ".jpeg"],
        "classes": {"names": NAMES, "keep": KEEP, "remove": REMOVE},
    }


def _make_raw(root):
    raw = root / "data" / "raw"
    for kind in ("labels", "images"):
        for split in ("train", "val", "test"):
            (raw / kind / split).mkdir(parents=True)
    t = raw / "labels" / "train"
    (t / "a.txt").write_text("6 0.1 0.2 0.3 0.4\n4 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    (t / "b.txt").write_text("5 0.1 0.1 0.1 0.1\n", encoding="utf-8")
    (t / "c.txt").write_text("", encoding="utf-8")
    (t / "d.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")  # no image
    (raw / "labels" / "val" / "e.txt").write_text("3 0.3 0.3 0.1 0.1\n", encoding="utf-8")
    (raw / "images" / "train" / "a.jpg").write_bytes(b"x")
    (raw / "images" / "train" / "b.jpg").write_bytes(b"x")
    (raw / "images" / "val" / "e.jpeg").write_bytes(b"x")


def test_end_to_end(tmp_path):
    _make_raw(tmp_path)
    report = df.run(_params(), tmp_path)
    out = tmp_path / "data" / "processed"

    # remap + drop rules, and nothing crosses splits
    assert (out / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == "4 0.1 0.2 0.3 0.4\n"
    assert (out / "labels" / "train" / "d.txt").exists()
    assert not (out / "labels" / "train" / "b.txt").exists()
    assert not (out / "labels" / "train" / "c.txt").exists()
    assert (out / "labels" / "val" / "e.txt").exists()
    assert not (out / "labels" / "val" / "a.txt").exists()
    assert (out / "images" / "train" / "a.jpg").exists()
    assert not (out / "images" / "train" / "b.jpg").exists()
    assert (out / "images" / "val" / "e.jpeg").exists()

    # per-split report
    train = report["splits"]["train"]["counts"]
    assert train["label_files_seen"] == 4
    assert train["label_files_kept"] == 2
    assert train["label_files_dropped"] == 2
    assert train["images_copied"] == 1
    assert train["images_missing"] == 1
    assert train["instances_dropped"] == 2
    assert report["splits"]["val"]["counts"]["label_files_kept"] == 1

    # class names in the report must be the right ones (guards the remap lookup)
    assert report["total"]["instances_after_by_class"] == {"Person": 1, "vest": 1, "boots": 1}

    # generated data.yaml
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "Person"}
    assert data["train"] == "images/train"

    # report file on disk
    on_disk = json.loads((tmp_path / "reports" / "filter_report.json").read_text(encoding="utf-8"))
    assert on_disk["total"]["counts"]["label_files_kept"] == 3


def test_refuses_non_empty_output(tmp_path):
    _make_raw(tmp_path)
    junk = tmp_path / "data" / "processed"
    junk.mkdir(parents=True)
    (junk / "junk.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        df.run(_params(), tmp_path)


def test_missing_split_folder_fails_before_writing(tmp_path):
    _make_raw(tmp_path)
    (tmp_path / "data" / "raw" / "labels" / "test").rmdir()
    with pytest.raises(FileNotFoundError):
        df.run(_params(), tmp_path)
    assert not (tmp_path / "data" / "processed").exists()