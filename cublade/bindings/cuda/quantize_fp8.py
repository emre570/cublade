# cublade/bindings/cuda/quantize_fp8.py

import torch

from cublade._loader import CSRC, load_cuda_module

_SOURCES = [CSRC / "quantization" / "quantize_per_tensor_fp8.cu"]
_MODULE_NAME = "cublade_quantize_per_tensor_fp8"


def _module():
    return load_cuda_module(name=_MODULE_NAME, sources=_SOURCES)


def quantize_per_tensor_fp8(
    x: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 (E4M3) quantization.

    Math:
        amax  = max(|x|).clamp_min(eps)
        scale = amax / 448.0
        q     = round(x * 448 / amax) clipped to E4M3

    Args:
        x: (..) torch.float16 or torch.bfloat16 contiguous CUDA tensor.
        eps: Numerical floor on amax to avoid divide-by-zero.

    Returns:
        (q, scale) - q is float8_e4m3fn, scale is a 0-d float32 scalar
        on the same device.
    """
    return _module().quantize_per_tensor_fp8(x, eps)
