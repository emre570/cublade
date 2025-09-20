import torch

from .tensor import QuantizedTensor
# ---------------------------
# Helper
# ---------------------------
def _prep_group_params(scale: torch.Tensor, zero_point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    if zero_point.ndim == 1:
        zero_point = zero_point.unsqueeze(1)
    return scale, zero_point

# ---------------------------
# Dequantize
# ---------------------------
def dequantize_int8(
    qt: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
):
    """Inverse of affine INT8/UINT8 quantization.

    Parameters
    ----------
    qt : torch.Tensor
        Quantized tensor.
    scale : torch.Tensor
        Scaling factors broadcastable to ``qt``.
    zero_point : torch.Tensor
        Zero-point offsets broadcastable to ``qt``.

    Returns
    -------
    torch.Tensor
        Floating-point approximation of the original tensor.
    """
    return scale * (qt.to(torch.float32) - zero_point)

def dequantize_per_group_int8(
    qt: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
    axis: int = -1,
):
    """Undo symmetric per-group quantization.

    Parameters
    ----------
    qt : torch.Tensor
        Quantized tensor. The target ``axis`` must have been grouped by
        ``group_size`` during quantization.
    scale : torch.Tensor
        Per-group scale factors.
    zero_point : torch.Tensor
        Per-group zero points.
    group_size : int
        Number of elements in each quantized group.
    axis : int, default -1
        Axis that was grouped during quantization.

    Returns
    -------
    torch.Tensor
        Dequantized tensor matching the shape of ``qt``.
    """
    q_perm = qt.movedim(axis, -1).contiguous()
    *lead, C = q_perm.shape
    assert C % group_size == 0, "group_size must evenly divide the target axis."
    qg = q_perm.view(-1, group_size)
    
    scale, zero_point = _prep_group_params(scale, zero_point)
    xg = dequantize_int8(qg, scale, zero_point)
    x_perm = xg.view(*lead, C)
    return x_perm.movedim(-1, axis)

# ---------------------------
# High-level wrapper
# ---------------------------
def dequantize_tensor(
    q: QuantizedTensor
):
    """High-level entry point that consumes ``QuantizedTensor`` metadata."""
    if q.mode in ('tensor', 'channel'):
        if q.dtype in (torch.int8, torch.uint8):
            return dequantize_int8(q.data, q.scale, q.zero_point)
    elif q.mode == 'group':
        if q.dtype in (torch.int8, torch.uint8):
            return dequantize_per_group_int8(q.data, q.scale, q.zero_point, q.group_size, q.axis)

    raise ValueError(f"Unsupported mode '{q.mode}' for dequantization.")
    
