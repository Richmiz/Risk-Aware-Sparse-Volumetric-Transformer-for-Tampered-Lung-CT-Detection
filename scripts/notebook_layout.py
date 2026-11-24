"""Keep two active notebooks; route historical generators into the archive."""

from pathlib import Path

SNAPSHOT = "20261113_cleanup"
ACTIVE_NOTEBOOKS = ("01_RSVT_TRAINING.ipynb", "02_ANNOTATION_REVIEW.ipynb")


def legacy_output(root, name):
    if Path(name).name != name or not name.endswith(".ipynb"):
        raise ValueError("Expected a notebook basename")
    target = Path(root) / "notebooks/archive/generated" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def legacy_input(root, name):
    if Path(name).name != name or not name.endswith(".ipynb"):
        raise ValueError("Expected a notebook basename")
    base = Path(root) / "notebooks/archive"
    for target in (base / "generated" / name, base / SNAPSHOT / name):
        if target.is_file():
            return target
    raise FileNotFoundError("Archived notebook not found: " + name)
