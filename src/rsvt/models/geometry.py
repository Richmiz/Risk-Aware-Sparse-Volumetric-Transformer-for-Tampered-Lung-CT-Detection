"""F05 geometry: C/Z/Y/X tensors, voxel X/Y/Z centers, physical LPS millimetres."""

import math
import torch
from torch.nn import functional as F


def power_of_two(value, minimum=4, maximum=64):
    return isinstance(value,int) and not isinstance(value,bool) and minimum <= value <= maximum and value & (value-1) == 0


def grid_shape(shape_zyx, patch_size):
    return tuple((int(n)+patch_size-1)//patch_size for n in shape_zyx)


def pad_volume(image, valid, multiple):
    extra = [(multiple-int(n)%multiple)%multiple for n in image.shape[-3:]]
    padding = (0,extra[2],0,extra[1],0,extra[0])
    return F.pad(image,padding),F.pad(valid,padding)


def sum_pool(volume, size):
    return F.avg_pool3d(volume.float(),size,stride=size,divisor_override=1)


def masked_pool_tokens(features, support, size, shape_zyx):
    count = sum_pool(support,size)
    pooled = sum_pool(features.float()*support.float(),size)/count.clamp_min(1.)
    z,y,x = shape_zyx
    return pooled[:,:,:z,:y,:x].flatten(2).transpose(1,2)[0]


def masked_max_pool_tokens(features, support, size, shape_zyx):
    """Per-patch maximum over supported positions; empty patches return zero.

    Mean pooling dilutes a small local anomaly by the patch volume. The maximum
    keeps the strongest local response, so a few-voxel trace is not averaged away.
    """
    mask = support.bool().expand_as(features)
    filled = features.float().masked_fill(~mask, float("-inf"))
    pooled = F.max_pool3d(filled, size, stride=size)
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    z,y,x = shape_zyx
    return pooled[:,:,:z,:y,:x].flatten(2).transpose(1,2)[0]


def patch_layout(shape_zyx, patch_size, voxel_to_lps, *, position_scale_mm=100.):
    """Border patches use clipped support; ends_xyz are exclusive voxel indices."""
    grid = grid_shape(shape_zyx,patch_size)
    device = voxel_to_lps.device
    axes = [torch.arange(n,device=device,dtype=torch.int64)*patch_size for n in grid]
    z,y,x = torch.meshgrid(*axes,indexing="ij")
    starts = torch.stack((x,y,z),-1).reshape(-1,3)
    size_xyz = torch.tensor(tuple(reversed(shape_zyx)),device=device,dtype=torch.int64)
    ends = torch.minimum(starts+patch_size,size_xyz)
    centers = (starts.double()+ends.double()-1.)/2.
    affine = voxel_to_lps.detach().double()
    physical = centers@affine[:3,:3].T+affine[:3,3]
    spacing = torch.linalg.vector_norm(affine[:3,:3],dim=0)
    edge_lengths = (ends-starts).double()*spacing
    fov_center = ((size_xyz.double()-1.)/2.)@affine[:3,:3].T+affine[:3,3]
    normalized_center = 2.*(centers+.5)/size_xyz.double()-1.
    normalized_size = (ends-starts).double()/size_xyz.double()
    position = torch.cat(((physical-fov_center)/position_scale_mm,edge_lengths/position_scale_mm,
                          normalized_center,normalized_size),1).float()
    return {"centers_voxel_xyz":centers,"centers_lps_mm":physical,"starts_xyz":starts,"ends_xyz":ends,
            "size_mm_xyz":edge_lengths,"position_features":position,"grid_shape_zyx":grid}


def validate_inputs(image, valid, affine, padding_mask, max_batch_voxels):
    if (image.ndim!=5 or image.shape[1]!=1 or image.dtype!=torch.float32 or min(image.shape)<=0
            or image.numel()>max_batch_voxels):
        raise ValueError("RSVT expects bounded nonempty float32 B/1/Z/Y/X normalized inputs")
    if valid.shape!=image.shape or valid.dtype!=torch.bool or valid.device!=image.device:
        raise ValueError("Validity mask must match image shape/device and have bool dtype")
    if padding_mask is not None and (padding_mask.shape!=image.shape or padding_mask.dtype!=torch.bool
                                     or padding_mask.device!=image.device):
        raise ValueError("Padding support must match image shape/device and have bool dtype")
    if not valid.flatten(1).any(1).all():
        raise ValueError("Every volume must contain valid voxels")
    values = image[valid]
    if not torch.isfinite(values).all() or (values<0).any() or (values>1).any():
        raise ValueError("Valid image voxels must be finite normalized values in [0,1]")
    if affine.shape!=(len(image),4,4) or not affine.is_floating_point():
        raise ValueError("Provide B/4/4 voxel-XYZ-to-LPS affine metadata")
    affine = affine.detach().to(device=image.device,dtype=torch.float64)
    last = torch.tensor([0.,0.,0.,1.],device=image.device,dtype=torch.float64).expand(len(image),4)
    if (not torch.isfinite(affine).all() or not torch.allclose(affine[:,3],last,atol=1e-8,rtol=0)
            or (torch.linalg.det(affine[:,:3,:3]).abs()<=torch.finfo(torch.float64).eps).any()):
        raise ValueError("Affines must be finite, nonsingular homogeneous transforms")
    shapes = []
    for index in range(len(image)):
        if padding_mask is None:
            shapes.append(tuple(image.shape[-3:]));continue
        support = padding_mask[index,0]
        if (valid[index,0]&~support).any() or not support.any():
            raise ValueError("Valid voxels must be inside nonempty padding support")
        extents = []
        for axis in range(3):
            active = support.any(dim=tuple(i for i in range(3) if i!=axis)).nonzero().flatten()
            extents.append(int(active[-1])+1)
        z,y,x = extents
        if int(support.sum())!=math.prod(extents) or not support[:z,:y,:x].all():
            raise ValueError("Padding support must be an origin-anchored rectangular box (F03 right padding)")
        shapes.append(tuple(extents))
    return shapes,affine
