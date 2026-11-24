"""Require an actual Colab kernel before starting public dataset downloads."""

import importlib.util
from pathlib import Path
import platform


def require_colab_runtime() -> None:
    """Fail before network/disk activity when a dataset entrypoint runs locally."""
    try:
        module_present = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        module_present = False
    if platform.system() != "Linux" or not Path("/content").is_dir() or not module_present:
        raise RuntimeError(
            "Dataset acquisition must run in a Google Colab kernel. "
            "Open notebooks/01_phase1_colab.ipynb in VS Code and select Colab; "
            "local dataset downloads are disabled."
        )
