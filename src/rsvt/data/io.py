"""Conventional single-frame CT IO in physical LPS mm; NumPy arrays use z/y/x."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pydicom
from rsvt.data.acquisition import sha256_file
from rsvt.data.inventory import file_kind
from rsvt.data.intensity import header_intensity_evidence


class SeriesValidationError(ValueError):
    """Input cannot be represented as the supported regular 3D CT grid."""


@dataclass(frozen=True)
class GeometryPolicy:
    position_tolerance_mm: float = 0.05
    direction_tolerance: float = 1e-4
    spacing_relative_tolerance: float = 0.01
    max_input_voxels: int = 300_000_000

    def __post_init__(self):
        values = (self.position_tolerance_mm, self.direction_tolerance,
                  self.spacing_relative_tolerance)
        if not all(not isinstance(value, bool) and np.isfinite(value) and value > 0 for value in values):
            raise ValueError("Geometry tolerances must be finite and positive")
        if (isinstance(self.max_input_voxels, bool) or not isinstance(self.max_input_voxels, int)
                or self.max_input_voxels <= 0):
            raise ValueError("max_input_voxels must be positive")


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    position_lps: tuple[float, float, float]
    slope: float
    intercept: float
    byte_size: int
    mtime_ns: int
    padding_low: float | None
    padding_high: float | None
    intensity_evidence: dict | None = None


@dataclass(frozen=True)
class SeriesGeometry:
    slices: tuple[SliceRecord, ...]
    size_xyz: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    origin_lps_mm: tuple[float, float, float]
    direction: tuple[float, ...]
    warnings: tuple[str, ...]
    max_spacing_deviation_mm: float
    max_inplane_drift_mm: float

    def as_dict(self) -> dict:
        return {"size_xyz": list(self.size_xyz), "array_order": "zyx",
                "spacing_xyz_mm": list(self.spacing_xyz_mm),
                "origin_lps_mm": list(self.origin_lps_mm), "direction": list(self.direction),
                "coordinate_system": "LPS_mm", "warnings": list(self.warnings),
                "max_spacing_deviation_mm": self.max_spacing_deviation_mm,
                "max_inplane_drift_mm": self.max_inplane_drift_mm}


@dataclass
class CTVolume:
    hu_zyx: np.ndarray
    valid_zyx: np.ndarray
    geometry: SeriesGeometry

    def as_images(self):
        import SimpleITK as sitk
        image = sitk.GetImageFromArray(self.hu_zyx)
        image.SetSpacing(self.geometry.spacing_xyz_mm)
        image.SetOrigin(self.geometry.origin_lps_mm)
        image.SetDirection(self.geometry.direction)
        valid = sitk.GetImageFromArray(self.valid_zyx.astype(np.uint8))
        valid.CopyInformation(image)
        return image, valid


def _vector(header, keyword: str, length: int) -> np.ndarray:
    try:
        result = np.asarray(getattr(header, keyword), dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise SeriesValidationError(f"Missing/invalid {keyword}") from exc
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise SeriesValidationError(f"Invalid {keyword}")
    return result


def _number(header, keyword: str) -> float:
    try:
        result = float(getattr(header, keyword))
    except (AttributeError, TypeError, ValueError) as exc:
        raise SeriesValidationError(f"Missing/invalid {keyword}") from exc
    if not np.isfinite(result):
        raise SeriesValidationError(f"Non-finite {keyword}")
    return result


def inspect_series(directory: str | Path, policy: GeometryPolicy | None = None) -> SeriesGeometry:
    """Sort slice planes by physical position; reject ambiguous or unsupported geometry.

    Multi-frame CT, mixed series, duplicate planes, irregular spacing, nonlinear
    modality LUTs, and sheared gantry-tilt grids require dedicated adapters.
    """
    policy = policy or GeometryPolicy()
    folder = Path(directory)
    if not folder.is_dir():
        raise SeriesValidationError("Series directory does not exist")
    paths = [p for p in sorted(folder.iterdir()) if p.is_file() and not p.is_symlink()
             and not p.name.startswith(".") and file_kind(p) == "dicom"]
    if len(paths) < 2:
        raise SeriesValidationError("At least two DICOM slices are required")
    tags = ["Modality", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID",
            "PatientID", "Rows", "Columns", "SamplesPerPixel", "NumberOfFrames",
            "PhotometricInterpretation", "PixelSpacing", "ImagePositionPatient",
            "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept", "RescaleType",
            "PixelPaddingValue", "PixelPaddingRangeLimit", "ModalityLUTSequence",
            "AnatomicalOrientationType", "SOPClassUID", "ImageType", (0x0018, 0x9361)]
    records, projections = [], []
    series_ids, patient_ids, frame_ids, sop_ids = set(), set(), set(), set()
    base_orientation = base_spacing = normal = dimensions = None
    warnings = set()
    for path in paths:
        initial = path.stat()
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
        except Exception as exc:
            raise SeriesValidationError(f"Invalid DICOM header: {path.name}") from exc
        if str(getattr(header, "Modality", "")) != "CT":
            raise SeriesValidationError("Series includes a non-CT image")
        if int(getattr(header, "NumberOfFrames", 1)) != 1:
            raise SeriesValidationError("Enhanced/multi-frame CT is not supported")
        if int(getattr(header, "SamplesPerPixel", 1)) != 1:
            raise SeriesValidationError("Only scalar grayscale CT is supported")
        if str(getattr(header, "PhotometricInterpretation", "")) not in {"MONOCHROME1", "MONOCHROME2"}:
            raise SeriesValidationError("Unsupported CT photometric interpretation")
        if str(getattr(header, "AnatomicalOrientationType", "BIPED")) != "BIPED":
            raise SeriesValidationError("Only BIPED patient coordinates are supported")
        if "ModalityLUTSequence" in header:
            raise SeriesValidationError("Non-linear modality LUT requires a dedicated adapter")
        series_id = str(getattr(header, "SeriesInstanceUID", ""))
        sop_id = str(getattr(header, "SOPInstanceUID", ""))
        if not series_id or not sop_id:
            raise SeriesValidationError("Missing series or SOP instance identity")
        if sop_id in sop_ids:
            raise SeriesValidationError("Repeated SOP instance within one series directory")
        sop_ids.add(sop_id)
        series_ids.add(series_id)
        patient_ids.add(str(getattr(header, "PatientID", "")))
        frame_ids.add(str(getattr(header, "FrameOfReferenceUID", "")))
        if len(series_ids) > 1 or len(patient_ids) > 1 or len(frame_ids) > 1:
            raise SeriesValidationError("Mixed series, patient, or frame identities in one directory")
        if "" in patient_ids:
            warnings.add("PatientID_missing_source_identity_requires_review")
        if "" in frame_ids:
            warnings.add("FrameOfReferenceUID_missing")
        try:
            shape = (int(header.Rows), int(header.Columns))
        except (AttributeError, TypeError, ValueError) as exc:
            raise SeriesValidationError("Missing image dimensions") from exc
        if min(shape) < 1:
            raise SeriesValidationError("Image dimensions must be positive")
        orientation = _vector(header, "ImageOrientationPatient", 6)
        spacing = _vector(header, "PixelSpacing", 2)
        position = _vector(header, "ImagePositionPatient", 3)
        if np.any(spacing <= 0):
            raise SeriesValidationError("Pixel spacing must be positive")
        row_direction, column_direction = orientation[:3], orientation[3:]
        if (abs(np.linalg.norm(row_direction) - 1) > policy.direction_tolerance
                or abs(np.linalg.norm(column_direction) - 1) > policy.direction_tolerance
                or abs(np.dot(row_direction, column_direction)) > policy.direction_tolerance):
            raise SeriesValidationError("Direction cosines are not orthonormal")
        if base_orientation is None:
            base_orientation, base_spacing, dimensions = orientation, spacing, shape
            normal = np.cross(row_direction, column_direction)
            normal /= np.linalg.norm(normal)
        if shape != dimensions:
            raise SeriesValidationError("Inconsistent slice dimensions")
        if not np.allclose(orientation, base_orientation, rtol=0, atol=policy.direction_tolerance):
            raise SeriesValidationError("Slice orientations differ")
        if not np.allclose(spacing, base_spacing, rtol=policy.spacing_relative_tolerance, atol=1e-6):
            raise SeriesValidationError("Inconsistent pixel spacing")
        slope, intercept = _number(header, "RescaleSlope"), _number(header, "RescaleIntercept")
        if slope == 0:
            raise SeriesValidationError("Rescale slope cannot be zero")
        units = str(getattr(header, "RescaleType", "") or "").strip().upper()
        if units and units != "HU":
            raise SeriesValidationError("Declared rescale units are not HU")
        intensity = header_intensity_evidence(header)
        if not units:
            warnings.add("RescaleType_absent_CT_Image_Module_HU_conditions_met" if intensity["status"] == "hu_by_ct_image_module_conditions"
                         else "RescaleType_missing_CT_HU_assumption_requires_review")
        padding_low = padding_high = None
        if "PixelPaddingRangeLimit" in header and "PixelPaddingValue" not in header:
            raise SeriesValidationError("Padding range has no padding value")
        if "PixelPaddingValue" in header:
            padding_low = _number(header, "PixelPaddingValue")
            padding_high = float(getattr(header, "PixelPaddingRangeLimit", padding_low))
            if not np.isfinite(padding_high):
                raise SeriesValidationError("Non-finite pixel padding range")
            padding_low, padding_high = sorted((padding_low, padding_high))
        final = path.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise SeriesValidationError("Source changed during header inspection")
        records.append(SliceRecord(path, tuple(position), slope, intercept,
                                   final.st_size, final.st_mtime_ns, padding_low, padding_high, intensity))
        projections.append(float(np.dot(position, normal)))
    records = [records[int(i)] for i in np.argsort(projections)]
    positions = np.asarray([record.position_lps for record in records])
    projected = positions @ normal
    gaps = np.diff(projected)
    if np.any(gaps <= policy.position_tolerance_mm):
        raise SeriesValidationError("Duplicate or indistinguishable slice planes")
    slice_spacing = float(np.median(gaps))
    max_deviation = float(np.max(np.abs(gaps - slice_spacing)))
    allowed = max(policy.position_tolerance_mm, policy.spacing_relative_tolerance * slice_spacing)
    if max_deviation > allowed:
        raise SeriesValidationError("Irregular slice spacing or missing internal slices")
    steps = projected - projected[0]
    drift = float(np.max(np.linalg.norm(positions - positions[0] - np.outer(steps, normal), axis=1)))
    if drift > policy.position_tolerance_mm:
        raise SeriesValidationError("Sheared/gantry-tilt geometry needs a dedicated reconstruction")
    if float(np.max(np.abs(steps - np.arange(len(records)) * slice_spacing))) > allowed:
        raise SeriesValidationError("Slice positions do not match a regular affine grid")
    size = (dimensions[1], dimensions[0], len(records))
    if int(np.prod(size, dtype=object)) > policy.max_input_voxels:
        raise SeriesValidationError("Input voxel count exceeds configured memory limit")
    direction = np.column_stack((base_orientation[:3], base_orientation[3:], normal))
    return SeriesGeometry(tuple(records), size,
                          (float(base_spacing[1]), float(base_spacing[0]), slice_spacing),
                          tuple(positions[0]), tuple(direction.ravel()), tuple(sorted(warnings)),
                          max_deviation, drift)


def source_fingerprint(geometry: SeriesGeometry) -> str:
    """F01 sorted file-digest multiset scheme; not a decoded voxel hash."""
    digests = []
    for record in geometry.slices:
        before = record.path.stat()
        if (before.st_size, before.st_mtime_ns) != (record.byte_size, record.mtime_ns):
            raise SeriesValidationError("Source changed since header inspection")
        digests.append(sha256_file(record.path))
        after = record.path.stat()
        if (after.st_size, after.st_mtime_ns) != (record.byte_size, record.mtime_ns):
            raise SeriesValidationError("Source changed while hashing")
    return hashlib.sha256("\n".join(sorted(digests)).encode()).hexdigest()


def read_hu_volume(geometry: SeriesGeometry) -> CTVolume:
    """Decode pixels and apply each slice's linear rescale exactly once."""
    x, y, z = geometry.size_xyz
    hu = np.empty((z, y, x), dtype=np.float32)
    valid = np.ones((z, y, x), dtype=bool)
    for index, record in enumerate(geometry.slices):
        before = record.path.stat()
        if (before.st_size, before.st_mtime_ns) != (record.byte_size, record.mtime_ns):
            raise SeriesValidationError("Source changed before decoding")
        try:
            header = pydicom.dcmread(record.path)
            pixels = header.pixel_array
        except Exception as exc:
            raise SeriesValidationError(f"Pixel decode failed for {record.path.name}; verify transfer-syntax decoder") from exc
        if pixels.shape != (y, x):
            raise SeriesValidationError("Decoded dimensions do not match audited header")
        hu[index] = pixels.astype(np.float32) * record.slope + record.intercept
        if not np.all(np.isfinite(hu[index])):
            raise SeriesValidationError("Non-finite HU after rescale")
        if record.padding_low is not None:
            valid[index] = (pixels < record.padding_low) | (pixels > record.padding_high)
        after = record.path.stat()
        if (after.st_size, after.st_mtime_ns) != (record.byte_size, record.mtime_ns):
            raise SeriesValidationError("Source changed during decoding")
    if not np.any(valid):
        raise SeriesValidationError("Volume contains only declared padding")
    return CTVolume(hu, valid, geometry)
