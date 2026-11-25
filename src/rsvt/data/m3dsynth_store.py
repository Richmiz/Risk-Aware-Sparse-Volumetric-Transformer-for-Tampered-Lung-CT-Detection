"""M3Dsynth acquisition for Colab: LIDC real series plus verified manipulation boxes.

Storage design
--------------
A manipulated M3Dsynth scan is its LIDC source series with a local region replaced.
The three author archives are 142-171 GB each and are not seekable, so they are
streamed once. For each selected sample we keep only

* the real source series once (uint16, author TIFF intensity convention), and
* a box of the manipulated scan around the declared manipulation centre plus the
  author voxel mask inside that box.

While a sample streams, every manipulated slice is compared with the real slice.
Any difference outside the stored box, a missing slice, a shape mismatch or mask
voxels outside the box makes the sample ineligible: a reconstructed volume is
never silently different from the archive. The same comparison is the global
shortcut check (manipulated and authentic volumes must be identical outside the
edited region).

Real series reproduce the authors' conversion (utils/dicom_utils.py,
convert_real_to_tiff.py): `.dcm` files sorted by descending ImagePositionPatient z,
raw stored pixels (no rescale), then `scan_to_uint16`. Exactness is verified by
the outside-box identity check above, not assumed.

Nothing here opens test pixels unless the caller selects test patients.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import http.client
import io
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np

GENERATORS = ("cycle", "pix2pix", "diffusion")
PUBLISHER_SPLITS = {"train": "train", "valid": "validation", "test": "test"}
OPERATIONS = {"rem": "removal", "inj": "insertion"}
FORENSIC_LABEL = {"rem": "FB", "inj": "FM"}
SITE_LABEL = {"rem": "TM", "inj": "TB"}
CANDIDATE_OFFSETS = (0, 512, 1024, 2000, 2048)
TCIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
ARCHIVE_BASE = "https://www.grip.unina.it/download/prog/M3Dsynth"
ARCHIVE_SIZES_GB = {"cycle": 171, "pix2pix": 142, "diffusion": 161}
# Boxes made under the 24 Nov 2026 rule stay untouched in "boxes"; newer ones go to their own folder.
BOX_FOLDER = "boxes_v2"
# Author metadata tables (no images) at a pinned commit of github.com/grip-unina/M3Dsynth.
# The SHA-256 values are those of the copies audited on 11 Nov 2026, unchanged at this commit.
METADATA_COMMIT = "f1461b74abac878fed45a3f0caf7bb4b72ff00d3"
METADATA_FILES = {
    "data.csv": ("data/data.csv", "fbacb7e2d3edbb9ca4581b2f5fb86b776fa9299cb8b3d629999faa49dbf24c44"),
    "LIDC.csv": ("data/LIDC.csv", "1da68f5fca3d8dab984af987fcbdda7493239522a80871929580b844b236851d"),
    "sets.csv": ("data/sets.csv", "f16458a8ca90b68bfb9bd67e453ffce955641ce0d1f2729c1c0badd4db6c841e"),
    "centers.csv": ("data/centers.csv", "1a27ee50264b944d8bee55c0400156092a3d28e3a2ed8b0ade7e48d48b82b2ee"),
    "LICENSE.md": ("LICENSE.md", "aed4e157a210c033bb296584a509d95224c4b441b94a5c2fe200eb7206cd43a7"),
    "README.md": ("README.md", "3bcb8dfd278631e385291b95480714e8c775cf340a4ecaec057c14a63aa6ec83"),
}
SLIDE = re.compile(r"slide(\d{4,})\.tiff?$")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_json(path, value):
    """Atomic JSON write with readback."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    if path.read_text(encoding="utf-8") != text:
        raise OSError("JSON readback failed: " + str(path))
    return value


def box_rule(half_mm, cube_half_mm):
    """Identifies how a box was cut and checked; a receipt under another rule is streamed again."""
    return f"v2;half_mm={float(half_mm):g};cube_half_mm={float(cube_half_mm):g};label_tail=inspected"


def series_key(orig_id, sdir_id):
    if not re.fullmatch(r"LIDC-IDRI-\d{4}", orig_id) or not re.fullmatch(r"[0-9]+", str(sdir_id)):
        raise ValueError(f"Unexpected LIDC identity {orig_id}/{sdir_id}")
    return f"{orig_id}__{sdir_id}"


# --------------------------------------------------------------------------- metadata


def _table(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid metadata columns: {Path(path).name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty metadata: {Path(path).name}")
    return rows


def load_uid_map(path):
    """Canonical TCIA SeriesInstanceUIDs from the 11 Nov 2026 metadata audit (744 series)."""
    rows = _table(path, ("publisher_source_id", "canonical_patient_id", "canonical_series_uid"))
    mapping = {}
    for row in rows:
        patient, sdir = row["publisher_source_id"].split(":")
        if patient != row["canonical_patient_id"] or not re.fullmatch(r"[0-9.]+", row["canonical_series_uid"]):
            raise ValueError("Invalid audited UID row " + row["publisher_source_id"])
        mapping[series_key(patient, sdir)] = row["canonical_series_uid"]
    return mapping


def load_metadata(metadata_dir, uid_map=None):
    """Author tables with explicit integrity checks; returns plain dictionaries.

    `uid_map` (optional CSV) attaches the audited TCIA SeriesInstanceUID to each
    series, so downloads use the exact series and headers are checked against it.
    """
    folder = Path(metadata_dir)
    data = _table(folder / "data.csv", ("img_id", "ty", "mod", "orig_id", "sdir_id", "coord_z", "coord_y", "coord_x"))
    lidc = _table(folder / "LIDC.csv", ("orig_id", "sdir_id", "spacing_z", "spacing_y", "spacing_x", "dicom"))
    sets = _table(folder / "sets.csv", ("orig_id", "set"))
    series, split_of, seen, sites = {}, {}, set(), {}
    for row in lidc:
        key = series_key(row["orig_id"], row["sdir_id"])
        if key in series:
            raise ValueError("Duplicate LIDC series " + key)
        series[key] = {"series_key": key, "patient_id": row["orig_id"], "sdir_id": row["sdir_id"],
                       "dicom_path": row["dicom"], **{k: float(row[k]) for k in ("spacing_z", "spacing_y", "spacing_x")}}
    for row in sets:
        if row["set"] not in PUBLISHER_SPLITS or row["orig_id"] in split_of:
            raise ValueError("Unknown or repeated publisher patient partition")
        split_of[row["orig_id"]] = PUBLISHER_SPLITS[row["set"]]
    samples = []
    for row in data:
        identity = (row["mod"], row["img_id"])
        if identity in seen:
            raise ValueError(f"Duplicate M3Dsynth sample {identity}")
        seen.add(identity)
        if row["mod"] not in ("real", *GENERATORS) or row["ty"] not in OPERATIONS:
            raise ValueError("Unknown generator or operation")
        key = series_key(row["orig_id"], row["sdir_id"])
        if key not in series or row["orig_id"] not in split_of:
            raise ValueError("Sample without LIDC series or publisher split: " + row["img_id"])
        if not row["img_id"].startswith(row["ty"] + "_"):
            raise ValueError("Site identifier does not encode its operation: " + row["img_id"])
        center = [int(row[k]) for k in ("coord_z", "coord_y", "coord_x")]
        site = (row["img_id"], key, tuple(center))
        if sites.setdefault(row["img_id"], site) != site:
            raise ValueError("A site identifier changes series or coordinates across generators: " + row["img_id"])
        samples.append({"img_id": row["img_id"], "variant": row["mod"], "operation": OPERATIONS[row["ty"]],
                        "op_code": row["ty"], "patient_id": row["orig_id"], "series_key": key,
                        "split": split_of[row["orig_id"]], "center_zyx": center})
    digests = {name: sha256_path(folder / name) for name in ("data.csv", "LIDC.csv", "sets.csv")}
    if uid_map is not None:
        mapping = load_uid_map(uid_map)
        missing = sorted(set(series) - set(mapping))
        if missing:
            raise ValueError(f"Audited UID map lacks {len(missing)} series, e.g. {missing[0]}")
        for key in series:
            series[key]["series_instance_uid"] = mapping[key]
        digests["uid_map"] = sha256_path(uid_map)
    return {"series": series, "split_of": split_of, "samples": samples, "sha256": digests}


def select_subset(meta, *, patients_per_split, generators=GENERATORS, seed=13):
    """Deterministic patient-level subset; all sites/generators of a patient move together.

    `patients_per_split` maps train/validation/test to a patient count; None means all.
    Selection order is a seeded hash of the LIDC patient id, so it is stable and
    independent of file order. Publisher partitions are patient-disjoint.
    """
    if set(patients_per_split) - set(PUBLISHER_SPLITS.values()):
        raise ValueError("Use train/validation/test patient caps")
    if not generators or set(generators) - set(GENERATORS) or len(set(generators)) != len(generators):
        raise ValueError("Choose distinct generators from " + ", ".join(GENERATORS))
    chosen = {}
    for split in PUBLISHER_SPLITS.values():
        cap = patients_per_split.get(split, 0)
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 0):
            raise ValueError("Patient caps must be nonnegative integers or None")
        patients = sorted({p for p, s in meta["split_of"].items() if s == split},
                          key=lambda p: digest_json([seed, "m3dsynth_patient", p]))
        for patient in patients[:len(patients) if cap is None else cap]:
            chosen[patient] = split
    used = {s["series_key"] for s in meta["samples"] if s["patient_id"] in chosen}
    series = [meta["series"][k] | {"split": chosen[meta["series"][k]["patient_id"]]} for k in sorted(used)]
    manipulated = [s for s in meta["samples"] if s["patient_id"] in chosen and s["variant"] in generators]
    sites = [s for s in meta["samples"] if s["patient_id"] in chosen and s["variant"] == "real"]
    return {"patients": dict(sorted(chosen.items())), "series": series, "samples": manipulated, "sites": sites,
            "generators": list(generators), "seed": seed, "patients_per_split": dict(patients_per_split),
            "counts": {"patients": dict(Counter(chosen.values())), "series": dict(Counter(s["split"] for s in series)),
                       "manipulated": dict(Counter(f"{s['split']}:{s['variant']}:{s['operation']}" for s in manipulated)),
                       "real_sites": dict(Counter(s["split"] for s in sites))}}


def storage_estimate(selection, *, real_mb=55., box_mb=2.4):
    """Rough planning numbers shown before any download (not measured sizes)."""
    archives = {g: ARCHIVE_SIZES_GB[g] for g in selection["generators"]}
    return {"real_series": len(selection["series"]), "manipulated_samples": len(selection["samples"]),
            "estimated_drive_gb": round((len(selection["series"]) * real_mb + len(selection["samples"]) * box_mb) / 1024, 1),
            "estimated_lidc_download_gb": round(len(selection["series"]) * 0.11, 1),
            "archive_stream_gb": archives, "archive_stream_total_gb": sum(archives.values()),
            "note": "Archives are gzip tar streams: the full archive is transferred even for a subset."}


# --------------------------------------------------------------------------- real series


def authors_uint16(scan):
    """Replicate M3Dsynth `scan_to_uint16` (offset choice and int16 arithmetic)."""
    if scan.dtype == np.uint16:
        return scan.copy(), {"rule": "already_uint16", "offset": 0}
    if scan.dtype != np.int16:
        raise ValueError("M3Dsynth conversion accepts int16/uint16 stored pixels only")
    vmin = -int(scan.min())
    rule = "negated_minimum"
    if vmin not in CANDIDATE_OFFSETS:
        k = 5
        blocks = [scan[z, y, x] for z in (slice(None, k), slice(-k, None))
                  for y in (slice(None, k), slice(-k, None)) for x in (slice(None, k), slice(-k, None))]
        corners = np.concatenate([b.ravel() for b in blocks])
        median = -float(np.median(corners))
        if median in CANDIDATE_OFFSETS:
            vmin, rule = int(median), "negated_corner_median"
        else:
            counts = [int(np.count_nonzero(corners == -c)) for c in CANDIDATE_OFFSETS]
            vmin, rule = CANDIDATE_OFFSETS[int(np.argmax(counts))], "most_frequent_corner_candidate"
    clipped = np.maximum(scan, np.int16(-vmin))
    with np.errstate(over="ignore"):
        shifted = (clipped + np.int16(vmin)).astype(np.int16)
    return shifted.astype(np.uint16), {"rule": rule, "offset": int(vmin)}


def read_dicom_series(directory):
    """Authors' ordering: `.dcm` files, descending ImagePositionPatient[2], raw pixels."""
    import pydicom
    files = [p for p in Path(directory).rglob("*") if p.is_file() and p.name.endswith(".dcm")]
    if not files:
        raise ValueError("No .dcm files in series download")
    slices = []
    for path in files:
        header = pydicom.dcmread(str(path))
        slices.append((-float(header.ImagePositionPatient[2]), path.name, header))
    slices.sort(key=lambda item: item[0])
    keys = [item[0] for item in slices]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate slice positions; the authors' sort would be ambiguous")
    headers = [item[2] for item in slices]
    scan = np.stack([h.pixel_array for h in headers])
    return scan, headers


def series_geometry(headers):
    """voxel (x,y,z) -> LPS mm affine for the descending-z stack; spacing regularity."""
    orientation = np.asarray(headers[0].ImageOrientationPatient, dtype=float)
    if any(not np.allclose(np.asarray(h.ImageOrientationPatient, dtype=float), orientation, atol=1e-4) for h in headers):
        raise ValueError("Slice orientation changes within the series")
    row, col = orientation[:3], orientation[3:]
    spacing = np.asarray(headers[0].PixelSpacing, dtype=float)
    positions = np.asarray([np.asarray(h.ImagePositionPatient, dtype=float) for h in headers])
    steps = np.diff(positions, axis=0) if len(positions) > 1 else np.asarray([np.cross(row, col)])
    step = np.median(steps, axis=0)
    deviation = float(np.abs(steps - step).max()) if len(steps) else 0.
    affine = np.eye(4)
    affine[:3, 0] = row * spacing[1]
    affine[:3, 1] = col * spacing[0]
    affine[:3, 2] = step
    affine[:3, 3] = positions[0]
    return {"voxel_to_lps": affine.tolist(), "spacing_zyx": [float(np.linalg.norm(step)), float(spacing[0]), float(spacing[1])],
            "uniform_spacing": deviation < 1e-2, "max_step_deviation_mm": deviation}


def convert_series(directory, expected):
    """Convert one downloaded series and verify its identity against LIDC.csv."""
    scan, headers = read_dicom_series(directory)
    first = headers[0]
    problems = []
    if str(first.PatientID) != expected["patient_id"]:
        problems.append("PatientID differs")
    if str(getattr(first, "Modality", "")) != "CT":
        problems.append("Modality is not CT")
    if any(str(h.SeriesInstanceUID) != str(first.SeriesInstanceUID) for h in headers):
        problems.append("Mixed SeriesInstanceUID")
    number = getattr(first, "SeriesNumber", None)
    if number is None or int(float(number)) != int(expected["sdir_id"]):
        problems.append("SeriesNumber differs from LIDC.csv sdir_id")
    if expected.get("series_instance_uid") and str(first.SeriesInstanceUID) != expected["series_instance_uid"]:
        problems.append("SeriesInstanceUID differs from the audited TCIA mapping")
    if problems:
        raise ValueError("; ".join(problems))
    u16, rule = authors_uint16(scan)
    slope = float(getattr(first, "RescaleSlope", 1.))
    intercept = float(getattr(first, "RescaleIntercept", 0.))
    if any(float(getattr(h, "RescaleSlope", 1.)) != slope or float(getattr(h, "RescaleIntercept", 0.)) != intercept
           for h in headers):
        raise ValueError("Rescale changes within the series; one intensity mapping cannot be declared")
    geometry = series_geometry(headers)
    spacing_check = (abs(geometry["spacing_zyx"][1] - expected["spacing_y"]) < 1e-3
                     and abs(geometry["spacing_zyx"][2] - expected["spacing_x"]) < 1e-3)
    meta = {"series_key": expected["series_key"], "patient_id": expected["patient_id"], "sdir_id": expected["sdir_id"],
            "series_instance_uid": str(first.SeriesInstanceUID), "shape_zyx": list(u16.shape),
            "stored_dtype": str(scan.dtype), "uint16_rule": rule,
            # HU = (u16 - offset) * slope + intercept for the int16 path; the
            # already-uint16 path keeps raw values (offset 0).
            "u16_to_hu": {"offset": rule["offset"], "slope": slope, "intercept": intercept},
            "geometry": geometry, "inplane_spacing_matches_metadata": spacing_check,
            "metadata_spacing_zyx": [expected["spacing_z"], expected["spacing_y"], expected["spacing_x"]],
            "content_sha256": sha256_bytes(np.ascontiguousarray(u16).tobytes())}
    return u16, meta


def _fetch(url, timeout=120, *, attempts=3, sleep=time.sleep):
    """GET a whole response; transient network errors are retried with a growing pause."""
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "rsvt-research-client"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, http.client.HTTPException):
            if attempt == attempts:
                raise
            sleep(15 * attempt)


def folder_series_hint(dicom_path):
    """NBIA descriptive folder '<SeriesNumber>.000000-<Description>-<UID last 5>'."""
    last = PurePosixPath(dicom_path.replace("\\", "/")).name
    match = re.match(r"^(\d+)\.\d+-.*-(\d{5})$", last)
    return (int(match.group(1)), match.group(2)) if match else (None, None)


def find_tcia_series(expected, *, base=TCIA_BASE, fetch=_fetch):
    query = urllib.parse.urlencode({"Collection": "LIDC-IDRI", "PatientID": expected["patient_id"]})
    candidates = json.loads(fetch(f"{base}/getSeries?{query}").decode("utf-8"))
    number, suffix = folder_series_hint(expected["dicom_path"])
    hits = [c for c in candidates if str(c.get("Modality", "CT")) == "CT"
            and c.get("SeriesNumber") is not None and int(float(c["SeriesNumber"])) == int(expected["sdir_id"])]
    if len(hits) > 1 and suffix:
        hits = [c for c in hits if str(c["SeriesInstanceUID"]).replace(".", "").endswith(suffix)] or hits
    if len(hits) != 1:
        raise ValueError(f"TCIA series for {expected['series_key']} is not unique ({len(hits)} matches)")
    return hits[0]["SeriesInstanceUID"]


def fetch_metadata(destination, *, fetch=_fetch):
    """Download the author metadata tables at the pinned commit; each file must match its SHA-256.

    Copies that already match are kept. No image is downloaded.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    fetched = []
    for name, (remote, expected) in METADATA_FILES.items():
        target = destination / name
        if target.is_file() and sha256_path(target) == expected:
            continue
        data = fetch(f"https://raw.githubusercontent.com/grip-unina/M3Dsynth/{METADATA_COMMIT}/{remote}")
        if sha256_bytes(data) != expected:
            raise ValueError(f"M3Dsynth metadata {name} differs from the audited copy; nothing was replaced")
        part = target.with_name(target.name + ".part")
        part.write_bytes(data)
        part.replace(target)
        fetched.append(name)
    return {"commit": METADATA_COMMIT, "fetched": fetched, "files": sorted(METADATA_FILES)}


def safe_extract_zip(data, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(BytesIO(data)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe path in series archive")
        archive.extractall(destination)


def acquire_real_series(expected, store_root, work_root, *, lidc_root=None, base=TCIA_BASE, fetch=_fetch,
                        coarse_builder=None):
    """Return verified metadata; writes Drive npz + JSON and a local memmap copy.

    Reuses an existing verified store entry (resume). `lidc_root` may point to an
    existing LIDC-IDRI download in the NBIA descriptive layout; otherwise the
    series is fetched from the public TCIA API.
    """
    store_root, work_root = Path(store_root), Path(work_root)
    key = expected["series_key"]
    npz, receipt = store_root / "real" / (key + ".npz"), store_root / "real" / (key + ".json")
    local = work_root / "real" / (key + ".npy")
    if receipt.exists() and npz.exists():
        meta = json.loads(receipt.read_text(encoding="utf-8"))
        if sha256_path(npz) == meta["file_sha256"]:
            if not local.exists():
                materialize_local(npz, local, meta["content_sha256"])
            return meta | {"reused": True}
    temporary = work_root / "downloads" / key
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        if lidc_root is not None and (Path(lidc_root) / expected["dicom_path"]).is_dir():
            source = Path(lidc_root) / expected["dicom_path"]
            origin = {"source": "existing_lidc_root", "path": expected["dicom_path"]}
        else:
            uid = expected.get("series_instance_uid") or find_tcia_series(expected, base=base, fetch=fetch)
            payload = fetch(f"{base}/getImage?" + urllib.parse.urlencode({"SeriesInstanceUID": uid}))
            safe_extract_zip(payload, temporary)
            source = temporary
            origin = {"source": "tcia_nbia_api", "series_instance_uid": uid, "zip_bytes": len(payload),
                      "zip_sha256": sha256_bytes(payload)}
        u16, meta = convert_series(source, expected)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    npz.parent.mkdir(parents=True, exist_ok=True)
    part = npz.with_name(npz.name + ".part")
    with part.open("wb") as stream:
        np.savez_compressed(stream, scan=u16)
    part.replace(npz)
    local.parent.mkdir(parents=True, exist_ok=True)
    np.save(local, u16)
    meta.update(origin=origin, file_sha256=sha256_path(npz), file_bytes=npz.stat().st_size,
                created_utc=datetime.now(timezone.utc).isoformat())
    if coarse_builder is not None:
        meta["coarse"] = coarse_builder(u16, meta, store_root)
    write_json(receipt, meta)
    return meta


def materialize_local(npz, local, content_sha256):
    """Decompress a Drive npz to a local .npy for memory-mapped random access."""
    with np.load(npz, allow_pickle=False) as data:
        scan = data["scan"]
    if scan.dtype != np.uint16 or sha256_bytes(np.ascontiguousarray(scan).tobytes()) != content_sha256:
        raise ValueError("Stored real series differs from its receipt: " + Path(npz).name)
    local = Path(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    part = local.with_name(local.name + ".part.npy")
    np.save(part, scan)
    part.replace(local)
    return local


# --------------------------------------------------------------------------- archives


def decode_tiff(data):
    """Decode one author TIFF slice (LZMA, I;16 scan or 1-bit label) with Pillow/libtiff."""
    from PIL import Image
    with Image.open(BytesIO(data)) as image:
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError("Expected a single 2-D TIFF slice")
    if array.dtype == bool:
        return array
    if array.dtype in (np.uint16, np.int32, np.uint8):
        return array.astype(np.uint16, copy=False)
    raise ValueError(f"Unexpected TIFF dtype {array.dtype}")


def parse_member(name):
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if len(parts) < 3 or parts[-3] not in ("scan", "label"):
        return None
    match = SLIDE.fullmatch(parts[-1])
    if not match:
        return None
    return parts[-3], parts[-2], int(match.group(1))


def box_for(center_zyx, shape_zyx, spacing_zyx, half_mm, cube_half_mm=0.):
    """Index box [start, stop) of +/- half_mm around the centre, clipped to the volume.

    Changed voxels were observed to fill a cube of about 41 mm in plane with the same number
    of voxels along z (26 receipts, 25 Nov 2026), so on thick slices the edit is far deeper
    than `half_mm`. Every axis therefore also covers `cube_half_mm` measured in in-plane pixels.
    """
    cube = int(math.ceil(cube_half_mm / max(min(float(s) for s in spacing_zyx[1:]), 1e-3)))
    start, stop = [], []
    for c, n, s in zip(center_zyx, shape_zyx, spacing_zyx):
        half = max(int(math.ceil(half_mm / max(float(s), 1e-3))), cube)
        start.append(max(0, int(c) - half))
        stop.append(min(int(n), int(c) + half + 1))
    if any(b <= a for a, b in zip(start, stop)):
        raise ValueError("Manipulation centre lies outside the scan")
    return start, stop


def label_tail_problem(voxels):
    return f"{voxels} mask voxels in label slices past the scan end"


class _SampleState:
    def __init__(self, sample, real_path, real_meta, half_mm, cube_half_mm=0.):
        self.sample = sample
        self.real = np.load(real_path, mmap_mode="r")
        self.shape = tuple(real_meta["shape_zyx"])
        self.start, self.stop = box_for(sample["center_zyx"], self.shape, real_meta["geometry"]["spacing_zyx"], half_mm,
                                         cube_half_mm)
        size = [b - a for a, b in zip(self.start, self.stop)]
        self.box = np.zeros(size, dtype=np.uint16)
        self.mask = np.zeros(size, dtype=bool)
        self.seen = {"scan": set(), "label": set()}
        self.diff_voxels = 0
        self.diff_outside = 0
        self.diff_bbox = None
        self.mask_voxels = 0
        self.mask_outside = 0
        self.mask_bbox = None
        self.label_tail_slices = 0
        self.label_tail_voxels = 0
        self.problems = []

    @staticmethod
    def _grow(bbox, z, rows, cols):
        lo = [z, int(rows.min()), int(cols.min())]
        hi = [z + 1, int(rows.max()) + 1, int(cols.max()) + 1]
        if bbox is None:
            return [lo, hi]
        return [[min(a, b) for a, b in zip(bbox[0], lo)], [max(a, b) for a, b in zip(bbox[1], hi)]]

    def _outside(self, z, rows, cols):
        (z0, y0, x0), (z1, y1, x1) = self.start, self.stop
        if not z0 <= z < z1:
            return len(rows)
        return int(np.count_nonzero((rows < y0) | (rows >= y1) | (cols < x0) | (cols >= x1)))

    def add(self, kind, z, array):
        if kind == "label" and z >= self.shape[0]:
            # The authors' loader takes the depth from scan/ and never reads these slices. Count
            # what they contain instead of assuming it: only empty ones are accepted (see result).
            self.label_tail_slices += 1
            self.label_tail_voxels += int(np.count_nonzero(array))
            return
        if not 0 <= z < self.shape[0] or z in self.seen[kind]:
            self.problems.append(f"{kind} slice {z} outside the real depth or repeated")
            return
        if array.shape != self.shape[1:]:
            self.problems.append(f"{kind} slice {z} shape {array.shape} differs from real {self.shape[1:]}")
            return
        self.seen[kind].add(z)
        (z0, y0, x0), (z1, y1, x1) = self.start, self.stop
        if kind == "scan":
            rows, cols = np.nonzero(array != self.real[z])
            if len(rows):
                self.diff_voxels += len(rows)
                self.diff_outside += self._outside(z, rows, cols)
                self.diff_bbox = self._grow(self.diff_bbox, z, rows, cols)
            if z0 <= z < z1:
                self.box[z - z0] = array[y0:y1, x0:x1]
        else:
            mask = array.astype(bool)
            rows, cols = np.nonzero(mask)
            if len(rows):
                self.mask_voxels += len(rows)
                self.mask_outside += self._outside(z, rows, cols)
                self.mask_bbox = self._grow(self.mask_bbox, z, rows, cols)
            if z0 <= z < z1:
                self.mask[z - z0] = mask[y0:y1, x0:x1]

    def complete(self):
        return all(len(self.seen[k]) == self.shape[0] for k in ("scan", "label"))

    def half_complete(self):
        """One kind finished while the other has not started (archive stores scan/ and label/ apart)."""
        done = [len(self.seen[k]) == self.shape[0] for k in ("scan", "label")]
        started = [bool(self.seen[k]) for k in ("scan", "label")]
        return any(done) and not all(started)

    _COUNTERS = ("diff_voxels", "diff_outside", "diff_bbox", "mask_voxels", "mask_outside", "mask_bbox",
                 "label_tail_slices", "label_tail_voxels", "problems")

    def spill(self, path):
        """Persist a half-finished sample so RAM holds only samples currently streaming."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), box=self.box, mask=self.mask)
        write_json(path.with_suffix(".json"), {"seen": {k: sorted(v) for k, v in self.seen.items()},
                                                **{k: getattr(self, k) for k in self._COUNTERS}})

    def restore(self, path):
        path = Path(path)
        with np.load(path.with_suffix(".npz")) as data:
            self.box, self.mask = data["box"].copy(), data["mask"].copy()
        state = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.seen = {k: set(v) for k, v in state["seen"].items()}
        for key in self._COUNTERS:
            setattr(self, key, state[key])
        path.with_suffix(".npz").unlink()
        path.with_suffix(".json").unlink()

    def result(self):
        problems = list(self.problems)
        for kind in ("scan", "label"):
            if len(self.seen[kind]) != self.shape[0]:
                problems.append(f"{kind} slices {len(self.seen[kind])}/{self.shape[0]}")
        if self.diff_outside:
            problems.append(f"{self.diff_outside} changed voxels outside the stored box")
        if self.mask_outside:
            problems.append(f"{self.mask_outside} mask voxels outside the stored box")
        if self.label_tail_voxels:
            problems.append(label_tail_problem(self.label_tail_voxels))
        if not self.mask_voxels:
            problems.append("empty author mask")
        if not self.diff_voxels:
            problems.append("manipulated scan identical to the real series")
        center = self.sample["center_zyx"]
        inside = self.mask_bbox is not None and all(lo <= c < hi for c, lo, hi in zip(center, *self.mask_bbox))
        return {"diff_voxels": self.diff_voxels, "diff_outside_box": self.diff_outside, "diff_bbox": self.diff_bbox,
                "mask_voxels": self.mask_voxels, "mask_outside_box": self.mask_outside, "mask_bbox": self.mask_bbox,
                "label_tail_slices": self.label_tail_slices, "label_tail_voxels": self.label_tail_voxels,
                "center_inside_mask_bbox": inside, "box_start_zyx": self.start, "box_stop_zyx": self.stop,
                "problems": problems, "eligible": not problems}


class StreamResumeError(OSError):
    """The server cannot continue the same file at the same byte; resuming would splice different data."""


class ResumingHTTPStream(io.RawIOBase):
    """Read-only HTTP stream that reconnects at the same byte offset after a network error.

    The author archives are 153-183 GB gzip streams that must be read in one pass.
    Without this, a dropped connection restarted the whole archive from byte 0.
    A reconnect asks for `Range: bytes=<offset>-`. It continues only if the server
    answers 206 starting at exactly that offset, with the same total size and the
    same ETag or Last-Modified as the first response; otherwise the stream stops
    instead of joining bytes from a different file. Decompression above this
    reader never sees the interruption. A VM restart still starts the current
    archive again (finished samples are skipped).
    """

    def __init__(self, url, *, timeout=120, retries=10, backoff=10., opener=urllib.request.urlopen, sleep=time.sleep):
        self.url, self.timeout, self.retries, self.backoff = str(url), timeout, retries, backoff
        self._opener, self._sleep = opener, sleep
        self.position, self.reconnects, self.total = 0, 0, None
        self._identity, self._response = None, None
        self._open()

    def readable(self):
        return True

    def _open(self):
        headers = {"User-Agent": "rsvt-research-client"}
        if self.position:
            headers["Range"] = f"bytes={self.position}-"
        response = self._opener(urllib.request.Request(self.url, headers=headers), timeout=self.timeout)
        status = getattr(response, "status", None) or response.getcode()
        if self.position:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", str(response.headers.get("Content-Range", "")).strip())
            if status != 206 or not match or int(match.group(1)) != self.position:
                response.close()
                raise StreamResumeError(f"Server did not resume at byte {self.position} (HTTP {status}); "
                                        "the archive must be streamed again from the start")
            total = int(match.group(3))
        else:
            if status != 200:
                response.close()
                raise StreamResumeError(f"Unexpected HTTP {status} for {self.url}")
            length = response.headers.get("Content-Length")
            total = int(length) if length else None
        identity = (total, response.headers.get("ETag") or response.headers.get("Last-Modified"))
        if self._identity is not None and identity != self._identity:
            response.close()
            raise StreamResumeError("The archive changed on the server while it was being streamed")
        self._identity, self.total, self._response = identity, total, response

    def _drop(self):
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
        self._response = None

    def readinto(self, buffer):
        failures = 0
        while True:
            try:
                if self._response is None:
                    self._open()
                count = self._response.readinto(buffer)
                if not count and self.total is not None and self.position < self.total:
                    raise http.client.IncompleteRead(b"", self.total - self.position)
                self.position += count
                return count
            except StreamResumeError:
                raise
            except (OSError, http.client.HTTPException) as exc:
                failures += 1
                self._drop()
                if failures > self.retries:
                    raise OSError(f"Archive stream failed at byte {self.position:,} after "
                                  f"{failures - 1} reconnect attempts") from exc
                self.reconnects += 1
                self._sleep(min(self.backoff * failures, 120.))

    def close(self):
        self._drop()
        super().close()


def open_archive_stream(source, *, timeout=120, opener=None, sleep=None):
    """URL or local .tgz path -> buffered binary stream (caller closes). URLs reconnect where they stopped."""
    if str(source).startswith(("http://", "https://")):
        raw = ResumingHTTPStream(source, timeout=timeout, opener=opener or urllib.request.urlopen,
                                 sleep=sleep or time.sleep)
        return io.BufferedReader(raw, buffer_size=1 << 20)
    return Path(source).open("rb")


def iter_extracted_root(root, generator):
    """Already-extracted author folders: <root>/<generator>/{scan,label}/<img_id>/slide####.tiff."""
    base = Path(root) / generator
    for kind in ("scan", "label"):
        for path in sorted((base / kind).glob("*/slide*.tif*")):
            yield f"{generator}/{kind}/{path.parent.name}/{path.name}", path.read_bytes


def stream_generator(generator, samples, real_meta, store_root, work_root, *, source=None, extracted_root=None,
                     half_mm=48., cube_half_mm=0., progress=None, progress_every=500):
    """Stream one author archive and store verified boxes for the selected samples.

    Resumable: samples with a verified receipt made under the same box rule are skipped.
    The archive is still transferred because gzip tar streams cannot seek.
    """
    if generator not in GENERATORS:
        raise ValueError("Unknown generator")
    store_root, work_root = Path(store_root), Path(work_root)
    rule = box_rule(half_mm, cube_half_mm)
    folder = store_root / BOX_FOLDER / generator
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {}
    for sample in samples:
        if sample["variant"] != generator:
            continue
        receipt = folder / (sample["img_id"] + ".json")
        if receipt.exists():
            stored = json.loads(receipt.read_text(encoding="utf-8"))
            box = folder / (sample["img_id"] + ".npz")
            if stored.get("box_rule") == rule and box.exists() and sha256_path(box) == stored.get("file_sha256"):
                continue
        wanted[sample["img_id"]] = sample
    states, finished, members, started = {}, Counter(), 0, time.time()
    reader = {}  # the HTTP reader, for bytes read and reconnects
    done, tails = set(), Counter()  # samples finished in this run; what their label tails held
    partial = work_root / "partial" / generator
    if partial.exists():
        shutil.rmtree(partial)  # a previous interrupted stream restarts from the archive start

    def finish(img_id):
        state = states.pop(img_id)
        outcome = state.result()
        box = folder / (img_id + ".npz")
        part = box.with_name(box.name + ".part")
        with part.open("wb") as stream:
            np.savez_compressed(stream, scan=state.box, mask=state.mask)
        part.replace(box)
        record = {"img_id": img_id, "generator": generator, "series_key": state.sample["series_key"],
                  "patient_id": state.sample["patient_id"], "split": state.sample["split"],
                  "operation": state.sample["operation"], "center_zyx": state.sample["center_zyx"],
                  "half_mm": half_mm, "cube_half_mm": cube_half_mm, "box_rule": rule, **outcome, "file_sha256": sha256_path(box), "file_bytes": box.stat().st_size,
                  "box_content_sha256": sha256_bytes(np.ascontiguousarray(state.box).tobytes()),
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_json(folder / (img_id + ".json"), record)
        finished["eligible" if outcome["eligible"] else "ineligible"] += 1
        done.add(img_id)
        if outcome["label_tail_slices"]:
            tails["nonempty" if outcome["label_tail_voxels"] else "empty"] += 1

    def after_finish(img_id, kind, z, array):
        """A slice that arrives once its sample is stored must correct the receipt, not restart the sample."""
        path = folder / (img_id + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        was_eligible, had_tail, had_voxels = record["eligible"], record["label_tail_slices"], record["label_tail_voxels"]
        depth = real_meta[record["series_key"]]["shape_zyx"][0]
        if kind == "label" and z >= depth:
            record["label_tail_slices"] += 1
            record["label_tail_voxels"] += int(np.count_nonzero(array))
            if record["label_tail_voxels"]:
                old = label_tail_problem(had_voxels)
                record["problems"] = [p for p in record["problems"] if p != old] + [label_tail_problem(record["label_tail_voxels"])]
            if had_tail:
                tails["nonempty" if had_voxels else "empty"] -= 1
            tails["nonempty" if record["label_tail_voxels"] else "empty"] += 1
        else:
            record["problems"].append(f"{kind} slice {z} outside the real depth or repeated")
        record["eligible"] = not record["problems"]
        if was_eligible and not record["eligible"]:
            finished["eligible"] -= 1
            finished["ineligible"] += 1
        write_json(path, record)

    def handle(name, read):
        nonlocal members
        parsed = parse_member(name)
        members += 1
        if parsed is None:
            return
        kind, img_id, z = parsed
        if img_id not in wanted:
            return
        if img_id in done:
            after_finish(img_id, kind, z, decode_tiff(read()))
            return
        if img_id not in states:
            sample = wanted[img_id]
            meta = real_meta[sample["series_key"]]
            states[img_id] = _SampleState(sample, work_root / "real" / (sample["series_key"] + ".npy"), meta, half_mm,
                                          cube_half_mm)
            spilled = partial / img_id
            if spilled.with_suffix(".json").exists():
                states[img_id].restore(spilled)
        state = states[img_id]
        state.add(kind, z, decode_tiff(read()))
        if state.complete():
            finish(img_id)
        elif state.half_complete():
            state.spill(partial / img_id)
            del states[img_id]
        if progress and members % progress_every == 0:
            event = {"stage": "archive", "generator": generator, "members": members, "open_samples": len(states),
                     "finished": dict(finished), "remaining": len(wanted) - sum(finished.values()),
                     "label_tails": {"empty": tails["empty"], "nonempty": tails["nonempty"]},
                     "elapsed_s": round(time.time() - started)}
            raw = reader.get("raw")
            if raw is not None:
                event.update(archive_gb_read=round(raw.position / 1e9, 2), reconnects=raw.reconnects,
                             archive_gb_total=round(raw.total / 1e9, 1) if raw.total else None,
                             mb_per_s=round(raw.position / 1e6 / max(time.time() - started, 1e-6), 1))
            progress(event)

    if not wanted:
        summary = {"generator": generator, "status": "already_complete", "finished": {}}
        write_json(folder / "stream_summary.json", summary)
        return summary
    if extracted_root is not None:
        for name, read in iter_extracted_root(extracted_root, generator):
            handle(name, read)
    else:
        stream = open_archive_stream(source or f"{ARCHIVE_BASE}/{generator}.tgz")
        raw = getattr(stream, "raw", None)
        reader["raw"] = raw if isinstance(raw, ResumingHTTPStream) else None
        try:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    handle(member.name, lambda m=member: archive.extractfile(m).read())
        finally:
            stream.close()
    for spilled in sorted(partial.glob("*.json")) if partial.exists() else []:
        img_id = spilled.stem
        sample = wanted[img_id]
        states[img_id] = _SampleState(sample, work_root / "real" / (sample["series_key"] + ".npy"),
                                      real_meta[sample["series_key"]], half_mm, cube_half_mm)
        states[img_id].restore(partial / img_id)
    for img_id in list(states):
        finish(img_id)  # incomplete samples are stored as ineligible with their problems
    missing = sorted(set(wanted) - {p.stem for p in folder.glob("*.json") if p.name != "stream_summary.json"})
    summary = {"generator": generator, "status": "complete" if not missing else "missing_samples",
               "members_read": members, "finished": dict(finished), "missing_from_archive": missing,
               "elapsed_s": round(time.time() - started), "half_mm": half_mm, "box_rule": rule,
               "label_tails": {"empty": tails["empty"], "nonempty": tails["nonempty"]},
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    if reader.get("raw") is not None:
        summary.update(archive_bytes_read=reader["raw"].position, reconnects=reader["raw"].reconnects)
    write_json(folder / "stream_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- manifest


def build_manifest(store_root, selection, meta):
    """One JSON manifest of authentic volumes, manipulated volumes and authentic sites."""
    store_root = Path(store_root)
    current = store_root / "manifest.json"
    if current.exists():  # keep every earlier manifest; the reports refer to its checksum
        kept = store_root / "manifest_history" / f"manifest_{sha256_path(current)[:12]}.json"
        if not kept.exists():
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(current, kept)
            if sha256_path(kept) != sha256_path(current):
                raise OSError("Earlier manifest was not preserved; the new one was not written")
    rules = set()
    series, volumes, sites, excluded = {}, [], [], []
    for item in selection["series"]:
        receipt = store_root / "real" / (item["series_key"] + ".json")
        if not receipt.exists():
            excluded.append({"series_key": item["series_key"], "reason": "real series not acquired"})
            continue
        stored = json.loads(receipt.read_text(encoding="utf-8"))
        series[item["series_key"]] = stored | {"split": item["split"]}
        volumes.append({"volume_id": "real:" + item["series_key"], "kind": "authentic", "series_key": item["series_key"],
                        "patient_id": item["patient_id"], "split": item["split"], "generator": "", "operation": "",
                        "label": "AUTHENTIC", "eligible": True})
    for sample in selection["samples"]:
        receipt = store_root / BOX_FOLDER / sample["variant"] / (sample["img_id"] + ".json")
        if sample["series_key"] not in series or not receipt.exists():
            excluded.append({"volume_id": f"{sample['variant']}:{sample['img_id']}",
                             "reason": "real series or manipulated box not acquired"})
            continue
        box = json.loads(receipt.read_text(encoding="utf-8"))
        rules.add(box.get("box_rule", ""))
        volumes.append({"volume_id": f"{sample['variant']}:{sample['img_id']}", "kind": "manipulated",
                        "series_key": sample["series_key"], "patient_id": sample["patient_id"], "split": sample["split"],
                        "generator": sample["variant"], "operation": sample["operation"],
                        "label": FORENSIC_LABEL[sample["op_code"]], "img_id": sample["img_id"],
                        "center_zyx": sample["center_zyx"], "box_file": f"{BOX_FOLDER}/{sample['variant']}/{sample['img_id']}.npz",
                        "box_file_sha256": box["file_sha256"], "box_start_zyx": box["box_start_zyx"],
                        "box_stop_zyx": box["box_stop_zyx"], "mask_bbox": box["mask_bbox"], "mask_voxels": box["mask_voxels"],
                        "diff_bbox": box["diff_bbox"], "label_tail_slices": box["label_tail_slices"],
                        "label_tail_voxels": box["label_tail_voxels"],
                        "eligible": box["eligible"], "problems": box["problems"]})
    for site in selection["sites"]:
        if site["series_key"] in series:
            sites.append({"img_id": site["img_id"], "series_key": site["series_key"], "split": site["split"],
                          "site_label": SITE_LABEL[site["op_code"]], "center_zyx": site["center_zyx"]})
    eligible = [v for v in volumes if v["eligible"]]
    assignments = sorted((v["volume_id"], v["patient_id"], v["split"]) for v in eligible)
    patients_by_split = defaultdict(set)
    for v in volumes:
        patients_by_split[v["split"]].add(v["patient_id"])
    overlap = set()
    for a in patients_by_split:
        for b in patients_by_split:
            if a < b:
                overlap |= patients_by_split[a] & patients_by_split[b]
    if overlap:
        raise ValueError("A patient appears in more than one partition: " + ", ".join(sorted(overlap)[:5]))
    document = {"schema_version": "m3dsynth_store_v1", "dataset": "M3Dsynth (Zingarini et al., ICASSP 2024)",
                "box_rules": sorted(rules), "metadata_sha256": meta["sha256"], "selection": {k: selection[k] for k in
                ("patients_per_split", "generators", "seed", "counts")},
                "series": series, "volumes": volumes, "sites": sites, "excluded": excluded,
                "assignments_sha256": digest_json(assignments), "patient_overlap_across_splits": sorted(overlap),
                "counts": {"volumes": dict(Counter(f"{v['split']}:{v['kind']}" for v in volumes)),
                           "eligible": dict(Counter(f"{v['split']}:{v['kind']}:{v['generator']}:{v['operation']}" for v in eligible)),
                           "ineligible": sum(not v["eligible"] for v in volumes), "sites": dict(Counter(s["split"] for s in sites))},
                "test_policy": "test volumes are acquired only when the caller selects test patients; training never reads them",
                "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(store_root / "manifest.json", document)
    return document


def verify_store(store_root, manifest=None):
    """Re-read every stored file against its receipt checksum (Drive readback)."""
    store_root = Path(store_root)
    manifest = manifest or json.loads((store_root / "manifest.json").read_text(encoding="utf-8"))
    checked, failures = 0, []
    for key, item in manifest["series"].items():
        checked += 1
        if sha256_path(store_root / "real" / (key + ".npz")) != item["file_sha256"]:
            failures.append(key)
    for volume in manifest["volumes"]:
        if volume["kind"] == "manipulated":
            checked += 1
            if sha256_path(store_root / volume["box_file"]) != volume["box_file_sha256"]:
                failures.append(volume["volume_id"])
    return {"files_checked": checked, "failures": failures, "ok": not failures}
