# cublade/bindings/cuda/dequantize_fp8.py

import torch

from cublade._loader import CSRC, load_cuda_module

_SOURCES = [CSRC / "quantization" / "dequantize_per_tensor_fp8.cu"]
_MODULE_NAME = "cublade_dequantize_per_tensor_fp8"


def _module():
    return load_cuda_module(name=_MODULE_NAME, sources=_SOURCES)


def dequantize_per_tensor_fp8(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Per-tensor FP8 (E4M3) dequantization.

    Math:
        y = (q.to(float) * scale).to(dtype)

    Args:
        q: (..) torch.float8_e4m3fn contiguous CUDA tensor (output of
            `quantize_per_tensor_fp8`).
        scale: 0-d torch.float32 CUDA tensor (= amax / 448).
        dtype: target output dtype, torch.float16 or torch.bfloat16
            (default torch.bfloat16).

    Returns:
        y: tensor with dtype matching `dtype`, same shape as q.
    """
    return _module().dequantize_per_tensor_fp8(q, scale, dtype)
