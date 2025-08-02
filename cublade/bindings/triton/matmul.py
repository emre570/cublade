import torch
import triton

from cublade.kernels.triton import matmul_kernel
from cublade.benchmark.triton_configs_checkers import get_autotune_config

def matmul(A: torch.Tensor, B: torch.Tensor, activation: str="", override_config=None, use_autotune=True):
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = A.shape
    N = B.shape[1]
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)
    
    # --- 1D GRID ---
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    strides = (A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1))

    # --- OVERRIDE MODE ---
    if override_config is not None:
        meta = override_config.copy()
        matmul_kernel[grid(meta)](
            A, B, C,
            M, N, K,
            *strides,
            **meta,
            ACTIVATION=activation
        )
        return C

    # --- AUTOTUNE MODE ---
    if use_autotune:
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            *strides,
            ACTIVATION=activation
        )
        return C

    # --- DEFAULT FIRST CONFIG ---
    configs = get_autotune_config()
    meta = configs[0].kwargs.copy()
    matmul_kernel[grid(meta)](
        A, B, C,
        M, N, K,
        *strides,
        **meta,
        ACTIVATION=activation
    )
    return C