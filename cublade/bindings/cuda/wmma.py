# cublade/bindings/cuda/wmma.py

import torch
from cublade._kernels import cublade_wmma as _cpp_wmma


def wmma_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    acc_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """
    WMMA-based GEMM with multi-precision support (16x16x16 fragments).

    Supported precision combinations:
        | Input dtype | Accumulator dtype |
        |-------------|-------------------|
        | float16     | float32 (default) |
        | float16     | float16           |
        | bfloat16    | float32           |
        | int8        | int32             |
        | uint8       | int32             |

    Args:
        A: Input matrix (M x K)
        B: Input matrix (K x N), must match A's dtype
        acc_dtype: Optional accumulator dtype. If None, uses default for input type.

    Returns:
        C: Output matrix (M x N) with accumulator dtype

    Note:
        Dimensions M, N, K must be multiples of 16 (WMMA tile constraint).
    """
    return _cpp_wmma.wmma_gemm(A, B, acc_dtype)
