import torch
from typing import Tuple

# ---------------------------
# Helpers
# ---------------------------
def _qrange(dtype: torch.dtype) -> Tuple[int, int]:
    """Return (q_min, q_max) for given integer dtype."""
    info = torch.iinfo(dtype)
    return info.min, info.max  # int8: (-128, 127), uint8: (0, 255)

def _reduce_min_max(x: torch.Tensor, per_channel: bool, ch_axis: int):
    """Return min/max per-tensor or per-channel (keepdim=True for broadcasting)."""
    if not per_channel:
        rmin = x.min().reshape([1] * x.ndim)
        rmax = x.max().reshape([1] * x.ndim)
        return rmin, rmax
    reduce_dims = [d for d in range(x.ndim) if d != ch_axis]
    rmin = x.amin(dim=reduce_dims, keepdim=True)
    rmax = x.amax(dim=reduce_dims, keepdim=True)
    return rmin, rmax

# ---------------------------
# Param computation (scale, zero-point)
# ---------------------------
def compute_params(
    x: torch.Tensor,
    scheme: str = "symmetric",          # "symmetric" | "asymmetric"
    dtype: torch.dtype = torch.int8,
    per_channel: bool = False,
    ch_axis: int = 0,
    eps: float = 1e-12,
):
    """
    Compute (scale, zero_point).
    - symmetric: zero_point is 0 (int8) or 128 (uint8); scale = amax/127
    - asymmetric: min/max affine; zero_point is computed and clamped
    - When per_channel=True, shape is broadcast-friendly along ch_axis.
    """
    assert dtype in (torch.int8, torch.uint8), "Only int8/uint8 are supported."
    scheme = scheme.lower()
    rmin, rmax = _reduce_min_max(x, per_channel, ch_axis)

    if scheme == "symmetric":
        amax = torch.maximum(rmax.abs(), rmin.abs())
        qmax_sym = 127.0
        scale = torch.where(amax > 0, amax / qmax_sym, torch.full_like(amax, eps))
        if dtype is torch.int8:
            zp = torch.zeros_like(scale)
        else:
            zp = torch.full_like(scale, 128.0)
        return scale, zp

    elif scheme == "asymmetric":
        q_min, q_max = _qrange(dtype)
        q_range = float(q_max - q_min)
        range_fp = (rmax - rmin).clamp_min(0.0)
        scale = torch.where(range_fp > 0, range_fp / q_range, torch.full_like(range_fp, eps))
        zp = q_min - (rmin / scale)
        zp = torch.round(zp)
        zp = torch.clamp(zp, min=float(q_min), max=float(q_max))
        return scale, zp

    else:
        raise ValueError("scheme must be 'symmetric' or 'asymmetric'.")