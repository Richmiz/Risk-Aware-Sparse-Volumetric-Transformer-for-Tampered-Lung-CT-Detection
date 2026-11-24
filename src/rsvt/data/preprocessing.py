"""Physical-space CT preprocessing with explicit geometry and annotation transforms."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from itertools import product
import hashlib
import json
import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class PreprocessingSpec:
    target_spacing_xyz_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hu_window: tuple[float, float] = (-1000.0, 400.0)
    background_hu: float = -1024.0
    max_output_voxels: int = 300_000_000
    save_native_hu: bool = True

    def __post_init__(self):
        if len(self.target_spacing_xyz_mm) != 3 or not all(not isinstance(x, bool) and np.isfinite(x) and x > 0 for x in self.target_spacing_xyz_mm):
            raise ValueError("Target spacing must contain three positive finite values")
        if len(self.hu_window) != 2 or not all(np.isfinite(x) for x in self.hu_window):
            raise ValueError("HU window must contain two finite values")
        if self.hu_window[0] >= self.hu_window[1]:
            raise ValueError("HU window lower bound must be smaller than upper bound")
        if (not np.isfinite(self.background_hu) or isinstance(self.max_output_voxels, bool)
                or not isinstance(self.max_output_voxels, int) or self.max_output_voxels <= 0):
            raise ValueError("Invalid background or output voxel limit")
        if not isinstance(self.save_native_hu, bool):
            raise ValueError("save_native_hu must be boolean")

    def fingerprint(self) -> str:
        value = {"implementation": "f02_physical_grid_v1", **asdict(self)}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def image_geometry(image: sitk.Image) -> dict:
    return {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_lps_mm": list(image.GetOrigin()), "direction": list(image.GetDirection()),
            "coordinate_system": "LPS_mm", "array_order": "zyx"}


def _require_3d(image: sitk.Image) -> None:
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar 3D image")
    if not np.all(np.isfinite(image.GetSpacing())) or min(image.GetSpacing()) <= 0:
        raise ValueError("Invalid source image spacing")
    if not np.all(np.isfinite(image.GetOrigin())) or not np.all(np.isfinite(image.GetDirection())):
        raise ValueError("Invalid source physical geometry")


def canonical_reference(image: sitk.Image, spec: PreprocessingSpec) -> sitk.Image:
    """An axis-aligned LPS grid enclosing all native voxel-support corners."""
    _require_3d(image)
    grid = canonical_grid_geometry(image_geometry(image), spec)
    reference = sitk.Image(grid["size_xyz"], sitk.sitkFloat32)
    reference.SetSpacing(tuple(grid["spacing_xyz_mm"]))
    reference.SetOrigin(tuple(grid["origin_lps_mm"]))
    reference.SetDirection(tuple(grid["direction"]))
    return reference


def canonical_grid_geometry(native: dict, spec: PreprocessingSpec) -> dict:
    """Compute the target grid and storage footprint without allocating an image."""
    size_in = np.asarray(native["size_xyz"], dtype=np.float64)
    spacing_in = np.asarray(native["spacing_xyz_mm"], dtype=np.float64)
    origin_in = np.asarray(native["origin_lps_mm"], dtype=np.float64)
    direction = np.asarray(native["direction"], dtype=np.float64).reshape(3, 3)
    if (size_in.shape != (3,) or spacing_in.shape != (3,) or origin_in.shape != (3,)
            or np.any(size_in < 1) or np.any(spacing_in <= 0)
            or not all(np.all(np.isfinite(a)) for a in (size_in, spacing_in, origin_in, direction))):
        raise ValueError("Invalid native grid description")
    corners = np.asarray([origin_in + direction @ (spacing_in * np.asarray(index))
                          for index in product(*[(-.5, float(n)-.5) for n in size_in])])
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    spacing = np.asarray(spec.target_spacing_xyz_mm)
    requested = np.maximum(1, np.ceil((upper - lower) / spacing - 1e-8))
    if not np.all(np.isfinite(requested)) or np.any(requested > spec.max_output_voxels):
        raise ValueError("Output voxel count exceeds configured memory limit")
    size = requested.astype(np.int64)
    if int(np.prod(size, dtype=object)) > spec.max_output_voxels:
        raise ValueError("Output voxel count exceeds configured memory limit")
    return {"size_xyz": [int(value) for value in size],
            "spacing_xyz_mm": [float(value) for value in spacing],
            "origin_lps_mm": [float(value) for value in lower + spacing / 2],
            "direction": np.eye(3).ravel().tolist(), "coordinate_system": "LPS_mm", "array_order": "zyx"}


def resample_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Nearest-neighbor resampling of already registered categorical/validity labels."""
    _require_3d(mask)
    _require_3d(reference)
    return sitk.Resample(mask, reference, sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkNearestNeighbor, 0, mask.GetPixelID())


def preprocess_volume(image: sitk.Image, valid: sitk.Image, spec: PreprocessingSpec) -> dict:
    """Keep full field of view; do not derive a crop from ground-truth lesions."""
    _require_3d(image)
    if image_geometry(valid) != image_geometry(image):
        raise ValueError("Validity mask does not share the native image geometry")
    valid = sitk.Cast(valid != 0, sitk.sitkUInt8)
    if not np.any(sitk.GetArrayViewFromImage(valid)):
        raise ValueError("No valid voxels for preprocessing")
    reference = canonical_reference(image, spec)
    masked = sitk.Mask(sitk.Cast(image, sitk.sitkFloat32), valid, outsideValue=float(spec.background_hu))
    hu = sitk.Resample(masked, reference, sitk.Transform(3, sitk.sitkIdentity),
                       sitk.sitkLinear, float(spec.background_hu), sitk.sitkFloat32)
    valid_out = resample_mask(valid, reference)
    if not np.any(sitk.GetArrayViewFromImage(valid_out)):
        raise ValueError("Resampling eliminated all valid voxels")
    low, high = spec.hu_window
    normalized = sitk.Cast((sitk.Clamp(hu, lowerBound=float(low), upperBound=float(high)) - low) / (high - low), sitk.sitkFloat32)
    return {"hu": hu, "normalized": normalized, "valid": valid_out,
            "transform": {"kind": "identity_in_physical_LPS_space", "native": image_geometry(image),
                          "processed": image_geometry(hu), "hu_interpolation": "linear",
                          "mask_interpolation": "nearest_neighbor", "full_field_of_view": True,
                          "lung_crop_applied": False, "hu_window": list(spec.hu_window),
                          "padding_replaced_with_background_before_resample": True,
                          "spec_fingerprint": spec.fingerprint()}}


def transform_points(points_xyz, source: sitk.Image, target: sitk.Image) -> dict:
    """Map continuous zero-based voxel coordinates through physical LPS space."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("Points must have shape (N, 3) and finite x/y/z values")
    _require_3d(source)
    _require_3d(target)
    physical, transformed, inside = [], [], []
    source_size, target_size = np.asarray(source.GetSize()), np.asarray(target.GetSize())
    for point in points:
        if np.any(point < 0) or np.any(point > source_size - 1):
            raise ValueError("Source point lies outside native voxel centers")
        world = source.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in point))
        out = target.TransformPhysicalPointToContinuousIndex(world)
        physical.append(world)
        transformed.append(out)
        inside.append(bool(np.all(np.asarray(out) >= -0.5 - 1e-6) and np.all(np.asarray(out) <= target_size - 0.5 + 1e-6)))
    return {"physical_lps_mm": physical, "target_index_xyz": transformed, "inside_target_support": inside}


def transform_box(lower_xyz, upper_xyz, source: sitk.Image, target: sitk.Image) -> dict:
    """Bound all eight mapped box corners without silent clipping."""
    lower, upper = np.asarray(lower_xyz, dtype=float), np.asarray(upper_xyz, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
        raise ValueError("Invalid box bounds")
    mapped = transform_points(list(product(*zip(lower, upper))), source, target)
    coords = np.asarray(mapped["target_index_xyz"])
    return {"lower_index_xyz": coords.min(axis=0).tolist(),
            "upper_index_xyz": coords.max(axis=0).tolist(),
            "all_corners_inside": all(mapped["inside_target_support"])}
