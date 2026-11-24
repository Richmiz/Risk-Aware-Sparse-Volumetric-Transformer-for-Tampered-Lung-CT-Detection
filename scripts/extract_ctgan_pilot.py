"""Extract CT-GAN pilot data from an already downloaded local ZIP prefix."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rsvt.data.ct_gan import extract_ctgan_prefix  # noqa: E402

if __name__ == "__main__":
    result = extract_ctgan_prefix(ROOT / "data/downloads/ct_gan_uci.zip.part",
                                 ROOT / "data/raw/ct_gan/pilot")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
