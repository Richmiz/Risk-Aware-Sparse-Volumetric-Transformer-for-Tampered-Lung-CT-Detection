"""Audit rescale metadata and numerical agreement without guessing calibration."""

from collections import Counter
import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut
from pydicom.uid import CTImageStorage


def header_intensity_evidence(header) -> dict:
    declared = str(getattr(header, "RescaleType", "") or "").strip().upper()
    raw_type = getattr(header, "ImageType", []) or []
    image_type = [part.strip().upper() for part in (str(raw_type).split("\\") if isinstance(raw_type, str) else map(str, raw_type))]
    energy_element = header.get((0x0018, 0x9361))
    multi = str(energy_element.value if energy_element is not None else "").strip().upper()
    conventional = str(getattr(header, "SOPClassUID", "")) == str(CTImageStorage)
    if declared:
        status = "explicit_hu" if declared == "HU" else "declared_non_hu"
    elif (conventional and len(image_type) >= 3 and image_type[0] == "ORIGINAL"
          and image_type[2] and image_type[2] != "LOCALIZER" and multi in {"", "NO"}):
        status = "hu_by_ct_image_module_conditions"
    else:
        status = "units_unspecified_requires_review"
    return {"status": status, "rescale_type": declared, "image_type": image_type,
            "multi_energy_ct": multi, "conventional_ct_sop": conventional,
            "rule": "DICOM_PS3.3_C.8.2.1", "calibration_verified": False}


def series_intensity_evidence(geometry) -> dict:
    evidence = [record.intensity_evidence or {"status": "units_unspecified_requires_review"}
                for record in geometry.slices]
    counts = Counter(item["status"] for item in evidence)
    supported = {"explicit_hu", "hu_by_ct_image_module_conditions"}
    status = ("hu_supported_by_header_evidence" if set(counts).issubset(supported)
              else "units_unspecified_requires_review")
    return {"status": status, "slice_status_counts": dict(counts),
            "rescale_pairs": sorted({(r.slope, r.intercept) for r in geometry.slices}),
            "declared_padding_slices": sum(r.padding_low is not None for r in geometry.slices),
            "calibration_verified": False,
            "interpretation": "Header evidence only; altered CT metadata is not proof of physical calibration or provenance."}


def sample_rescale_qa(geometry, hu_zyx=None, max_samples_per_slice=4096) -> dict:
    """Compare first/middle/last slice samples against pydicom's reference transform.

    If a native array is supplied, check that array too. This is a numerical
    transformation check, not verification against a phantom or pathology.
    """
    if max_samples_per_slice < 1:
        raise ValueError("Intensity sample count must be positive")
    indices = sorted({0, len(geometry.slices) // 2, len(geometry.slices) - 1})
    checks = []
    for index in indices:
        record = geometry.slices[index]
        stat = record.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record.byte_size, record.mtime_ns):
            raise ValueError("Source changed before intensity QA")
        dataset = pydicom.dcmread(record.path)
        raw = dataset.pixel_array.ravel()
        selected = np.linspace(0, raw.size - 1, min(raw.size, max_samples_per_slice), dtype=int)
        reference = np.asarray(apply_modality_lut(raw[selected], dataset), dtype=np.float64)
        actual = (raw[selected].astype(np.float32) * record.slope + record.intercept
                  if hu_zyx is None else np.asarray(hu_zyx[index]).ravel()[selected])
        if not np.all(np.isfinite(reference)) or not np.allclose(actual, reference, rtol=1e-6, atol=1e-3):
            raise ValueError("Rescale sample differs from pydicom reference")
        after = record.path.stat()
        if (after.st_size, after.st_mtime_ns) != (record.byte_size, record.mtime_ns):
            raise ValueError("Source changed during intensity QA")
        checks.append({"native_z": index, "filename": record.path.name, "samples": len(selected),
                       "raw_min_sample": float(raw[selected].min()), "raw_max_sample": float(raw[selected].max()),
                       "rescaled_min_sample": float(reference.min()), "rescaled_max_sample": float(reference.max()),
                       "max_abs_reference_error": float(np.max(np.abs(actual.astype(np.float64) - reference)))})
    return {"reference": "pydicom.pixels.apply_modality_lut", "sample_checks": checks,
            "sampled_slices": len(checks), "numerical_rescale_check": "passed",
            "native_array_checked": hu_zyx is not None, "calibration_verified": False}
