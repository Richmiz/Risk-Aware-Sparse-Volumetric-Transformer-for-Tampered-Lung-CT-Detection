"""Source-header evidence and human coordinate QA; no model or split changes.

Medical pixels are handled by the Colab entry point only. The exported JSON/CSV
receipt contains no images; the HTML previews stay on Colab/Drive.
"""

from __future__ import annotations

import base64
from collections import Counter, defaultdict
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import struct
import zipfile
import zlib

import numpy as np
import pydicom

from rsvt.data.acquisition import sha256_file
from rsvt.data.verification import read_csv, scan_key, verify_labels
from rsvt.utils.runtime import require_colab_runtime


def header_evidence(headers, catalog, image_count):
    """Only emit identifiers already present in the public catalog.

    A study UID is patient evidence, not an exact source-series match. Image
    count and SeriesNumber are candidate filters and never establish identity.
    Conflicting identifiers are retained as a conflict rather than voted away.
    """
    hits, candidates = [], set()
    for header in headers:
        for element in header.iterall():
            if element.keyword not in {"SeriesInstanceUID", "StudyInstanceUID", "PatientID"}:
                continue
            field = element.keyword
            matches = [r for r in catalog if str(r.get(field, "")) == str(element.value) and str(element.value)]
            if matches:
                patients = sorted({r["PatientID"] for r in matches})
                # PatientID alone is a publisher-declared candidate.
                hits.append({"field": field, "public_value": str(element.value),
                             "public_patients": patients,
                             "unique_patient_UID_evidence": field != "PatientID" and len(patients) == 1})
        number = str(getattr(header, "SeriesNumber", ""))
        if number:
            candidates.update(str(r["SeriesInstanceUID"]) for r in catalog
                              if str(r.get("SeriesNumber", "")) == number and int(r["ImageCount"]) == image_count)
    hits = [json.loads(s) for s in sorted({json.dumps(h, sort_keys=True) for h in hits})]
    all_patients = set().union(*(set(h["public_patients"]) for h in hits)) if hits else set()
    proven = {h["public_patients"][0] for h in hits if h["unique_patient_UID_evidence"]}
    conflict = len(all_patients) > 1
    exact_series = sorted({h["public_value"] for h in hits if h["field"] == "SeriesInstanceUID"})
    status = ("conflicting_public_header_evidence" if conflict else
              "exact_catalog_series_UID_evidence" if exact_series else
              "exact_catalog_study_UID_patient_evidence" if proven else
              "public_patient_ID_candidate" if hits else
              "metadata_candidates_only" if candidates else "unresolved")
    return {"status": status, "public_patient_UID_evidence": next(iter(proven)) if len(proven) == 1 and not conflict else "",
            "exact_public_series_UIDs": exact_series, "public_header_hits": hits,
            "candidate_series_UIDs": sorted(candidates), "conflict": conflict,
            "pixel_source_match_performed": False, "independent_patient_verified": False}


def coordinate_hypotheses(point):
    """Explicit alternatives; source slice filename and XY index bases are separate."""
    result = []
    for slice_base in (0, 1):
        for xy_base in (0, 1):
            for swap in (False, True):
                x, y = int(point["x"]) - xy_base, int(point["y"]) - xy_base
                if swap:
                    x, y = y, x
                result.append({"hypothesis": f"slice_base{slice_base}_xy_base{xy_base}_" + ("swap" if swap else "native"),
                               "slice_filename": str(int(point["slice"]) - slice_base) + ".dcm", "x": x, "y": y})
    return result


def native_xy_to_lps(header, x, y):
    orientation = np.asarray(header.ImageOrientationPatient, dtype=float)
    spacing = np.asarray(header.PixelSpacing, dtype=float)
    position = np.asarray(header.ImagePositionPatient, dtype=float)
    if (orientation.shape != (6,) or spacing.shape != (2,) or position.shape != (3,)
            or not all(np.isfinite(a).all() for a in (orientation, spacing, position)) or (spacing <= 0).any()
            or not np.allclose(np.linalg.norm(orientation.reshape(2, 3), axis=1), 1, atol=1e-5)
            or abs(np.dot(orientation[:3], orientation[3:])) > 1e-5):
        raise ValueError("Invalid source geometry for coordinate QA")
    return (position + orientation[:3] * spacing[1] * x + orientation[3:] * spacing[0] * y).tolist()


def grayscale_png(hu):
    """Lossless PNG of a declared lung-window display, without image dependencies."""
    gray = np.rint(np.clip((np.asarray(hu, dtype=float) + 1000) / 1400, 0, 1) * 255).astype(np.uint8)
    if gray.ndim != 2 or not np.isfinite(hu).all():
        raise ValueError("Finite 2D HU image required")
    height, width = gray.shape
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    scanlines = b"".join(b"\0" + row.tobytes() for row in gray)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def checked_dicom(root, row, *, pixels=False):
    relative = PurePosixPath(row["relative_path"].replace("\\", "/"))
    root = Path(root).resolve()
    raw_target = root / Path(*relative.parts)
    target = raw_target.resolve()
    if (relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(root)
            or any(p.is_symlink() for p in (raw_target, *raw_target.parents) if p != root)):
        raise ValueError("Unsafe source image path")
    content = target.read_bytes()
    if len(content) != int(row["bytes"]) or hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("Source bytes differ from F01")
    header = pydicom.dcmread(BytesIO(content), stop_before_pixels=not pixels)
    if not pixels:
        return header
    array = header.pixel_array
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected finite single-frame source pixels")
    hu = array.astype(float) * float(header.RescaleSlope) + float(header.RescaleIntercept)
    if not np.isfinite(hu).all():
        raise ValueError("Nonfinite native HU")
    return header, hu


def render_point(point, source_rows, dataset_root):
    panels, hypotheses, decoded = [], [], {}
    for candidate in coordinate_hypotheses(point):
        filename = candidate["slice_filename"]
        title = escape(candidate["hypothesis"])
        if filename not in source_rows:
            panels.append(f"<section><h3>{title}</h3><p>Source filename unavailable.</p></section>")
            hypotheses.append({**candidate, "status": "source_filename_unavailable"})
            continue
        if filename not in decoded:
            decoded[filename] = checked_dicom(dataset_root, source_rows[filename], pixels=True)
        header, hu = decoded[filename]
        h, w = hu.shape
        x, y = candidate["x"], candidate["y"]
        inside = 0 <= x < w and 0 <= y < h
        hypotheses.append({**candidate, "status": "review_pending" if inside else "out_of_bounds",
                           "point_LPS_mm": native_xy_to_lps(header, x, y) if inside else None,
                           "source_sha256": source_rows[filename]["sha256"]})
        data = base64.b64encode(grayscale_png(hu)).decode("ascii")
        # Pixel centers sit at SVG coordinates x+.5/y+.5 for this unflipped native raster.
        marker = (f'<path d="M {x+.5-8},{y+.5} h 16 M {x+.5},{y+.5-8} v 16" '
                  'stroke="#00ff66" stroke-width="1"/>' if inside else "")
        panels.append(f'<section><h3>{title}</h3><p>file={escape(filename)}; x={x}, y={y}; '
                      f'{"pending" if inside else "OUT OF BOUNDS"}</p><svg viewBox="0 0 {w} {h}" '
                      f'xmlns="http://www.w3.org/2000/svg"><image width="{w}" height="{h}" '
                      f'href="data:image/png;base64,{data}"/>{marker}</svg></section>')
    title = escape(f'{point["case_key"]} {point["label"]}: slice={point["slice"]}, x={point["x"]}, y={point["y"]}')
    html = ('<!doctype html><meta charset="utf-8"><title>Coordinate QA</title><style>'
            'body{font:14px sans-serif;background:#171717;color:white}main{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}'
            'svg{width:100%;image-rendering:pixelated}h3{font-size:12px}section{min-width:0}</style>'
            f'<h1>{title}</h1><p>Native DICOM pixel rows/columns, no display flip. Window -1000..400 HU. '
            'Green cross: candidate point. Hypotheses require visual evidence; small index differences may remain ambiguous. '
            'FB denotes a removal site, so absence of a nodule alone does not reject the coordinate. '
            'No lesion masks or automatic coordinate choice are produced.</p><main>' + "".join(panels) + '</main>')
    return html, hypotheses


def persist_receipt(run, drive_results_root):
    """Read every artifact back before publishing a successful persistence receipt."""
    run, root = Path(run).resolve(), Path(drive_results_root).resolve()
    if not run.is_relative_to(root) or run == root:
        raise ValueError("QA results must be below the project Drive results directory")
    paths = [p for p in sorted(run.rglob("*")) if p.is_file() and p.name != "drive_persistence_receipt.json"]
    files = {p.relative_to(run).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in paths}
    for name, item in files.items():
        p = run / name
        if p.is_symlink() or sha256_file(p) != item["sha256"] or p.stat().st_size != item["bytes"]:
            raise ValueError("Drive artifact failed readback")
    receipt = {"schema": "coordinate_qa_drive_receipt_v1", "run": str(run), "files": files,
               "all_artifacts_read_back": True, "manual_transfer_required": False,
               "training_ready": False, "remote_execution_evidenced_by_this_receipt_only": True}
    path = run / "drive_persistence_receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("Drive receipt failed readback")
    return receipt


def validate_coordinate_reviews(points, decisions):
    """Validate the existing human-decision contract without writing files."""
    seen = set()
    for decision in decisions:
        identifier = decision.get("point_id")
        if identifier not in points or identifier in seen:
            raise ValueError("Unknown or repeated review point")
        seen.add(identifier)
        status = decision.get("status")
        if (status not in {"accepted", "ambiguous", "rejected"}
                or any(not isinstance(decision.get(k), str) or not decision[k].strip()
                       for k in ("reviewer", "evidence"))):
            raise ValueError("Human status, reviewer and evidence are required")
        allowed = {h["hypothesis"] for h in points[identifier]["hypotheses"] if h["status"] == "review_pending"}
        selected = decision.get("selected_hypothesis")
        if (status == "accepted" and selected not in allowed) or (status != "accepted" and selected is not None):
            raise ValueError("Accepted coordinates require an in-bounds hypothesis; ambiguous/rejected points select none")
    return {"decisions": decisions, "reviewed_points": len(seen), "total_preview_points": len(points),
              "all_preview_points_accepted": bool(points) and len(seen) == len(points) and all(d["status"] == "accepted" for d in decisions),
              "test_points_reviewed": False, "training_ready": False,
              "scope": "human coordinate decisions only; source identity/protocol and global convention remain separate"}


def record_coordinate_reviews(run, decisions, *, drive_results_root):
    """Append documented human decisions; never unlock training or change targets."""
    run = Path(run)
    if not run.resolve().is_relative_to(Path(drive_results_root).resolve()):
        raise ValueError("Reviews must stay in the project Drive QA directory")
    points = {p["point_id"]: p for p in json.loads((run / "points.json").read_text(encoding="utf-8"))}
    result = validate_coordinate_reviews(points, decisions)
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    target = run / ("human_review_" + hashlib.sha256(content.encode()).hexdigest()[:12] + ".json")
    if not target.exists():
        with target.open("x", encoding="utf-8") as stream:
            stream.write(content)
    elif target.read_text(encoding="utf-8") != content:
        raise ValueError("Existing human review differs")
    persist_receipt(run, drive_results_root)
    return result


def run_annotation_qa(*, dataset_root, manifest_path, files_path, catalog_path, policy_path,
                      output_dir, drive_results_root, code_sha256, progress=None):
    require_colab_runtime()
    run = Path(output_dir)
    if not run.resolve().is_relative_to(Path(drive_results_root).resolve()) or run.exists():
        raise ValueError("Use a fresh project Drive QA run directory")
    records = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["records"]
    record_map = {r["case_key"]: r for r in records}
    if len(record_map) != len(records):
        raise ValueError("Duplicate manifest cases")
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    label_root = Path(dataset_root) / "Tampered Scans"
    _, annotations = verify_labels({e: label_root / f"labels_{e}.csv" for e in ("exp1", "exp2")}, policy)
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    by_case = defaultdict(list)
    for row in read_csv(files_path):
        if row["dataset"] == "ct_gan" and row["kind"] == "dicom":
            by_case[scan_key(PurePosixPath(row["relative_path"]).parent)].append(row)
    if set(by_case) != set(record_map):
        raise ValueError("Inventory and manifest case coverage differs")
    run.mkdir(parents=True)
    (run / "headers").mkdir()
    (run / "previews").mkdir()
    identities, sampled_count = {}, 0
    for key, rows in sorted(by_case.items()):
        rows.sort(key=lambda r: int(PurePosixPath(r["relative_path"]).stem))
        sampled = [rows[i] for i in sorted({0, len(rows)//2, len(rows)-1})]
        headers = [checked_dicom(dataset_root, row) for row in sampled]
        result = header_evidence(headers, catalog, len(rows))
        result.update(case_key=key, split=record_map[key]["split"],
                      sampled_header_files=len(sampled), sampled_source_sha256=[r["sha256"] for r in sampled])
        identities[key] = result
        sampled_count += len(sampled)
        (run / "headers" / (key.replace(":", "_") + ".json")).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        if progress:
            progress({"stage": "header_evidence", "case": key, "status": result["status"]})
    points, decisions, links = [], [], []
    nonzero = [r for r in annotations if not r["duplicate"] and not r["zero_location_placeholder"]]
    for point in nonzero:
        key = point["case_key"]
        if record_map[key]["split"] == "test":
            continue
        if record_map[key]["split"] not in {"train", "validation"}:
            raise ValueError("Unexpected manifest partition")
        point_id = hashlib.sha256(json.dumps([key, point["slice"], point["x"], point["y"]]).encode()).hexdigest()[:20]
        source_rows = {PurePosixPath(r["relative_path"]).name: r for r in by_case[key]}
        html, hypotheses = render_point(point, source_rows, dataset_root)
        relative = "previews/point_" + point_id + ".html"
        (run / relative).write_text(html, encoding="utf-8")
        points.append({**point, "point_id": point_id, "split": record_map[key]["split"],
                       "preview": relative, "hypotheses": hypotheses})
        decisions.append({"point_id": point_id, "status": "pending", "selected_hypothesis": None,
                          "reviewer": "", "evidence": ""})
        links.append(f'<li><a href="{relative}">{escape(key)} / {point_id}</a></li>')
        if progress:
            progress({"stage": "coordinate_preview", "completed": len(points)})
    (run / "points.json").write_text(json.dumps(points, indent=2)+"\n", encoding="utf-8")
    (run / "coordinate_review_template.json").write_text(json.dumps(decisions, indent=2)+"\n", encoding="utf-8")
    (run / "index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Coordinate QA: train/validation</h1>'
                                    '<p>Pending review. Test previews excluded.</p><ul>' + "".join(links) + '</ul>', encoding="utf-8")
    public_groups = defaultdict(list)
    for key, result in identities.items():
        if result["public_patient_UID_evidence"]:
            public_groups[result["public_patient_UID_evidence"]].append(key)
    edges = [{"public_patient": patient, "cases": keys,
              "splits": sorted({record_map[k]["split"] for k in keys})}
             for patient, keys in sorted(public_groups.items()) if len(keys) > 1]
    summary = {"schema": "source_coordinate_QA_v1", "case_count": len(records),
               "sampled_header_files_checked_against_F01": sampled_count,
               "source_header_status_counts": dict(Counter(r["status"] for r in identities.values())),
               "public_patient_UID_evidence_count": sum(bool(r["public_patient_UID_evidence"]) for r in identities.values()),
               "public_header_conflict_count": sum(r["conflict"] for r in identities.values()),
               "public_patient_header_edges": edges,
               "cross_split_public_header_evidence": any(len(e["splits"]) > 1 for e in edges),
               "unique_nonzero_points": len(nonzero), "preview_points": len(points),
               "test_points_excluded": sum(record_map[r["case_key"]]["split"] == "test" for r in nonzero),
               "annotation_axis_visual_review_completed": False,
               "new_dataset_download_bytes": 0, "training_performed": False,
               "model_loaded": False, "test_pixels_decoded": False, "test_performance_evaluated": False,
               "historical_manifests_modified": False, "training_ready": False,
               "code_bundle_sha256": code_sha256,
               "input_sha256": {str(p): sha256_file(p) for p in map(Path, (manifest_path, files_path, catalog_path, policy_path))},
               "pending": ["review 8 coordinate hypotheses; document ambiguity rather than selecting by appearance alone",
                           "original public source pixel matching for unresolved identities",
                           "publisher/archive 484-slice discrepancy", "separate baseline optimization after QA review"]}
    (run / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    # Metadata review ZIP deliberately excludes HTML previews (which contain CT pixels).
    with zipfile.ZipFile(run / "metadata.review.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(run.glob("*.json")) + sorted((run / "headers").glob("*.json")):
            archive.write(p, p.relative_to(run).as_posix())
    persist_receipt(run, drive_results_root)
    return summary
