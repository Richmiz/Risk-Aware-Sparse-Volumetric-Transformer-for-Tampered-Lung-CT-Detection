"""Download public research data with byte limits, receipts, and safe ZIP extraction."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
from urllib.request import Request, urlopen
import zipfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_file(url: str, destination: Path, *, max_bytes: int, progress=None) -> dict:
    """Download once; retain incomplete .part files and never overwrite existing files.

    A computed SHA256 is an acquisition fingerprint, not publisher verification.
    No remote code is executed. Callers supply URLs from a reviewed source registry.
    """
    if not url.startswith("https://"):
        raise ValueError("Public downloads require HTTPS")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".part")
    receipt_path = destination.with_name(destination.name + ".receipt.json")
    if destination.exists() or partial.exists() or receipt_path.exists():
        raise FileExistsError(f"Download destination or receipt already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    started = time.monotonic()
    last_notice = started
    request = Request(url, headers={"User-Agent": "RSVT-Research/0.1"})
    with urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        expected = int(length) if length else None
        if expected is not None and expected > max_bytes:
            raise ValueError(f"Download exceeds configured limit: {expected} > {max_bytes}")
        if expected and shutil.disk_usage(destination.parent).free < expected + 1024**3:
            raise ValueError("Insufficient storage for download and 1 GiB reserve")
        resolved_url = response.geturl()
        if not resolved_url.startswith("https://"):
            raise ValueError("Download redirect downgraded HTTPS")
        with partial.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("Download exceeded configured byte limit; .part retained")
                if shutil.disk_usage(destination.parent).free < len(chunk) + 64 * 1024**2:
                    raise ValueError("Storage reserve reached; .part retained")
                stream.write(chunk)
                digest.update(chunk)
                now = time.monotonic()
                if progress and now - last_notice >= 10:
                    progress({"downloaded_bytes": count, "expected_bytes": expected})
                    last_notice = now
        if expected is not None and count != expected:
            raise ValueError("Incomplete response; .part retained")
    if not count:
        raise ValueError("Empty download; .part retained")
    partial.rename(destination)
    receipt = {
        "url": url, "resolved_url": resolved_url,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": count, "sha256": digest.hexdigest(),
        "publisher_checksum_verified": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def safe_member_path(root: Path, name: str) -> Path:
    """Reject archive traversal, Windows drive/ADS paths, and absolute paths."""
    parts = PurePosixPath(name.replace("\\", "/"))
    if parts.is_absolute() or not parts.parts or ".." in parts.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    if any(":" in part for part in parts.parts):
        raise ValueError(f"Unsafe Windows archive member: {name}")
    target = (root / Path(*parts.parts)).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError(f"Archive member escapes destination: {name}")
    return target


def extract_zip(archive: Path, destination: Path, *, max_uncompressed_bytes: int) -> dict:
    """Extract a reviewed archive into a new directory, checking every path first."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        total = sum(member.file_size for member in members)
        if total > max_uncompressed_bytes:
            raise ValueError("Archive exceeds uncompressed-size limit")
        targets = []
        seen = set()
        for member in members:
            target = safe_member_path(destination, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive symlinks are not supported")
            normalized = str(target).casefold()
            if normalized in seen:
                raise ValueError("Archive contains duplicate destination paths")
            seen.add(normalized)
            targets.append((member, target))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < total + 1024**3:
            raise ValueError("Insufficient storage for extraction and 1 GiB reserve")
        destination.mkdir()
        for member, target in targets:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return {"archive": str(archive), "files": sum(not m.is_dir() for m in members),
            "uncompressed_bytes": total, "destination": str(destination)}
