"""Download the official metadata and retain original documentation/license notices."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsvt.data.acquisition import fetch_file, sha256_file  # noqa: E402
from rsvt.utils.runtime import require_colab_runtime  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("/content/rsvt-data/raw/m3dsynth/metadata"))
    args = parser.parse_args()
    require_colab_runtime()
    metadata = args.destination
    base = "https://raw.githubusercontent.com/grip-unina/M3Dsynth/main/"
    names = {name: "data/" + name for name in ("data.csv", "LIDC.csv", "centers.csv", "sets.csv", "dataset.py")}
    names.update({"README.md": "README.md", "LICENSE.md": "LICENSE.md"})
    for local, remote in names.items():
        target = metadata / local
        if target.exists():
            # Retain previously acquired bytes; do not silently refresh a moving branch.
            receipt = {"url": base + remote, "status": "existing_local_copy_not_refetched",
                       "bytes": target.stat().st_size, "sha256": sha256_file(target)}
            receipt_path = target.with_name(target.name + ".receipt.json")
            if receipt_path.exists():
                recorded = json.loads(receipt_path.read_text(encoding="utf-8"))
                if recorded.get("sha256") != receipt["sha256"]:
                    raise ValueError(f"Local metadata differs from its acquisition receipt: {local}")
            else:
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        else:
            receipt = fetch_file(base + remote, target, max_bytes=30 * 1024**2)
        print(json.dumps({"file": local, **receipt}), flush=True)
