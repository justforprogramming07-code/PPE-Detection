from pathlib import Path
import yaml


def load_config(path="config/filter.yaml"):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg["orphan_labels"] not in ("strict", "lenient"):
        raise ValueError("orphan_labels must be 'strict' or 'lenient'")
    return cfg