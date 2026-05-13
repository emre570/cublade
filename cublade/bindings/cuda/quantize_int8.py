# cublade/bindings/cuda/quantize_int8.py

import torch

from cublade._loader import CSRC, load_cuda_module

_SOURCES = [CSRC / "quantization" / "quantize_int8.cu"]
_MODULE_NAME = "cublade_quantize_int8"


def _module():
    # IEEE-precise math is required for the bit-exact gate: --use_fast_math
    # rewrites a/b as a*frcp(b) and may fuse multiplies, drifting by 1 ULP
    # and flipping round-half-to-even ties versus the torch reference.
    return load_cuda_module(name=_MODULE_NAME, sources=_SOURCES, use_fast_math=False)


def quantize_per_tensor_int8(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric INT8 quantization.

    Math:
        amax  = max(|x|)
        scale = max(amax, eps) / 127.0
        q     = round_half_even(x / scale) clamped to [-127, 127]

    Args:
        x: (..) contiguous CUDA tensor, dtype in {fp16, bf16, fp32}.
        eps: Numerical floor on amax to avoid divide-by-zero.

    Returns:
        (q, scale) - q is int8 with the same shape as x; scale is a 0-d
        float32 tensor on the same device. The dequant convention is
        y ~= q.to(float) * scale.
    """
    return _module().quantize_per_tensor_int8(x, eps)


def quantize_per_channel_int8(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT8 quantization along axis 0.

    The first dimension is the channel axis; each channel gets its own scale
    derived from the channel's amax. v1 supports ch_axis=0 only because that
    matches the W8A16 linear layer's usage (out_features-major weights).

    Args:
        x: (C, ...) contiguous CUDA tensor, dtype in {fp16, bf16, fp32}.
            The product of trailing dims must satisfy
            F * element_size % 16 == 0 (i.e. F % 8 == 0 for half/bf16,
            F % 4 == 0 for fp32) so the 16-byte vectorised loads stay aligned.
        eps: Numerical floor on per-row amax.

    Returns:
        (q, scale) - q is int8 with the same shape as x; scale is float32
        with shape (C,) on the same device.
    """
    return _module().quantize_per_channel_int8(x, eps)


def quantize_per_group_int8(
    x_grouped: torch.Tensor,
    group_size: int,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group symmetric INT8 quantization.

    The kernel assumes the caller has already reshaped to a 2D contiguous
    layout of (num_groups, group_size); see `quantize_tensor(..., mode='group')`
    for the convenience wrapper that handles the movedim + view.

    Args:
        x_grouped: (num_groups, group_size) contiguous CUDA tensor,
            dtype in {fp16, bf16, fp32}.
        group_size: number of elements per group. Must equal x_grouped.size(1)
            and lie in (0, 4096].
        eps: Numerical floor on per-group amax.

    Returns:
        (q, scale) - q is int8 with shape (num_groups, group_size); scale is
        float32 with shape (num_groups,) on the same device.
    """
    return _module().quantize_per_group_int8(x_grouped, group_size, eps)
