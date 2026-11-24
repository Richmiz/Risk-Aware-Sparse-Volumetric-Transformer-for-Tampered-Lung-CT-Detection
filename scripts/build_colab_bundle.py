"""Build a source-only Colab bundle; never include downloaded datasets or private keys."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("src", "configs", "docs", "notebooks", "scripts", "tests", "requirements", ".vscode")
ROOT_FILES = ("README.md", "AGENTS.md", "IMPLEMENTATION_PHASES.md", "pyproject.toml", ".gitignore", ".editorconfig")
EXPLICIT_DATA_FILES = ("data/public_sources.json", "data/manifests/manifest.template.csv",
                       # Author metadata tables (no images) needed to select and verify M3Dsynth samples.
                       *(f"data/raw/m3dsynth/metadata/{name}" for name in
                         ("data.csv", "LIDC.csv", "sets.csv", "centers.csv", "LICENSE.md", "README.md")),
                       # Audited LIDC series UIDs (11 Nov 2026) so downloads target the exact TCIA series.
                       "reports/supplementary/data_label_audit_20261111/metadata_review_v2/m3dsynth_source_groups.csv")
ALLOWED_SUFFIXES = {".py", ".toml", ".md", ".ipynb", ".json", ".txt"}


def build_bundle(root: Path, output: Path, *, include_notebooks=True) -> dict:
    members = []
    for name in (*ROOT_FILES, *EXPLICIT_DATA_FILES):
        path = root / name
        if path.is_file() and not path.is_symlink():
            members.append(path)
    for directory in DIRECTORIES:
        if directory == 'notebooks' and not include_notebooks:
            continue
        for path in (root / directory).rglob("*"):
            if (path.is_file() and not path.is_symlink() and path.suffix in ALLOWED_SUFFIXES
                    and "__pycache__" not in path.parts and not path.name.startswith(".env")
                    and not (directory == "notebooks" and "archive" in path.relative_to(root / directory).parts)):
                members.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Replace only this generated code bundle; medical data is outside the allowlist.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(members)):
            name = path.relative_to(root).as_posix()
            if path.suffix == ".ipynb":
                # Executed notebooks can contain medical previews and logs.
                # Package source cells only, preserving the original file.
                notebook = json.loads(path.read_text(encoding="utf-8"))
                notebook["metadata"] = {key: value for key, value in notebook.get("metadata", {}).items()
                                        if key in {"kernelspec", "language_info"}}
                for cell in notebook["cells"]:
                    cell["metadata"] = {}
                    cell.pop("attachments", None)
                    if cell["cell_type"] == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None
                archive.writestr(name, json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
            else:
                archive.write(path, name)
    content = output.read_bytes()
    result = {"bundle": str(output), "files": len(set(members)), "bytes": len(content),
              "sha256": hashlib.sha256(content).hexdigest(), "contains_datasets": False}
    output.with_name(output.name + ".sha256").write_text(result["sha256"] + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rsvt-colab-code.zip")
    parser.add_argument('--source-only', action='store_true', help='Exclude notebooks; keep runtime bundle independent of notebook checksums')
    args = parser.parse_args()
    print(json.dumps(build_bundle(ROOT, args.output, include_notebooks=not args.source_only), indent=2))
