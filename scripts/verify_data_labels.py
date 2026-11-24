"""Audit publisher labels and metadata without reading medical image pixels."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsvt.data.verification import run_metadata_verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest-path", "files-path", "series-path", "overlap-path", "label-root",
                 "readme-path", "m3-metadata-dir", "tcia-catalog-path", "policy-path", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = run_metadata_verification(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("scope", "metadata_checks_passed", "training_ready", "goal_verification_complete")}, indent=2))
    return 0 if result["metadata_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
