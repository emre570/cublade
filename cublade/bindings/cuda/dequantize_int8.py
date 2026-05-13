# cublade/bindings/cuda/dequantize_int8.py

import torch

from cublade._loader import CSRC, load_cuda_module

_SOURCES = [CSRC / "quantization" / "dequantize_int8.cu"]
_MODULE_NAME = "cublade_dequantize_int8"


def _module():
    # IEEE-precise math (no --use_fast_math) for bit-exact dequant tests. The
    # per-element multiply by `scale` would otherwise fuse with other ops and
    # drift by 1 ULP.
    return load_cuda_module(name=_MODULE_NAME, sources=_SOURCES, use_fast_math=False)


def dequantize_per_tensor_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Per-tensor symmetric INT8 dequantization.

    Math:
        y = (q.to(float) * scale).to(dtype)

    Args:
        q: (..) torch.int8 contiguous CUDA tensor.
        scale: 0-d torch.float32 CUDA tensor produced by
            `quantize_per_tensor_int8`.
        dtype: target output dtype, torch.float16 or torch.bfloat16
            (default torch.bfloat16).

    Returns:
        y: tensor with dtype matching `dtype`, same shape as q.
    """
    return _module().dequantize_per_tensor_int8(q, scale, dtype)


def dequantize_per_channel_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Per-channel symmetric INT8 dequantization along axis 0.

    Args:
        q: (C, ...) torch.int8 contiguous CUDA tensor. The product of trailing
            dims must be a multiple of 8 (16-byte alignment for the vec store).
        scale: (C,) torch.float32 CUDA tensor.
        dtype: target output dtype, torch.float16 or torch.bfloat16
            (default torch.bfloat16).

    Returns:
        y: tensor with dtype matching `dtype`, same shape as q.
    """
    return _module().dequantize_per_channel_int8(q, scale, dtype)


def dequantize_per_group_int8(
    q_grouped: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Per-group symmetric INT8 dequantization.

    Args:
        q_grouped: (num_groups, group_size) torch.int8 contiguous CUDA tensor.
        scale: (num_groups,) torch.float32 CUDA tensor.
        group_size: must equal q_grouped.size(1) and lie in (0, 4096].
        dtype: target output dtype, torch.float16 or torch.bfloat16
            (default torch.bfloat16).

    Returns:
        y: tensor with dtype matching `dtype`, same shape as q_grouped.
    """
    return _module().dequantize_per_group_int8(q_grouped, scale, group_size, dtype)
