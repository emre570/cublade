import torch
import triton
import triton.language as tl

from cublade.kernels.triton import matmul_kernel, matmul_kernel_autotuned
from cublade.benchmark.triton_configs_checkers import get_autotune_config

PREC_TORCH_TO_TRITON = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float8_e5m2: tl.float8e5,
    torch.int8: tl.int8,
    torch.int32: tl.int32
}

def matmul(A: torch.Tensor, B: torch.Tensor, activation: str="", override_config=None, use_autotune=True):
    assert A.shape[1] == B.shape[0], f"Incompatible dimensions"
    
    A = A.contiguous()
    B = B.contiguous()
    # Device preconditions: Triton backend requires CUDA tensors on the same device
    if A.device.type != "cuda" or B.device.type != "cuda":
        raise ValueError(f"Triton matmul requires CUDA tensors. Got A on {A.device}, B on {B.device}.")
    if A.device != B.device:
        raise ValueError(f"A and B must be on the same device. Got A on {A.device}, B on {B.device}.")
    
    M, K = A.shape
    N = B.shape[1]
    
    c_dtype = None
    acc_dtype = None
    acc_is_float = None
    
    if A.dtype == B.dtype:
        #FP32 check
        if (A.dtype == torch.float32):
            c_dtype = PREC_TORCH_TO_TRITON[torch.float32]
            acc_dtype = PREC_TORCH_TO_TRITON[torch.float32]
            acc_is_float = True
            C = torch.empty((M, N), device=A.device, dtype=torch.float32)
        #FP16 check
        elif (A.dtype == torch.float16):
            c_dtype = PREC_TORCH_TO_TRITON[torch.float16]
            acc_dtype = PREC_TORCH_TO_TRITON[torch.float32]
            acc_is_float = True
            C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        #BF16 check
        elif (A.dtype == torch.bfloat16):
            c_dtype = PREC_TORCH_TO_TRITON[torch.bfloat16]
            acc_dtype = PREC_TORCH_TO_TRITON[torch.float32]
            acc_is_float = True
            C = torch.empty((M, N), device=A.device, dtype=torch.bfloat16)
        #FP8 check
        elif (A.dtype == torch.float8_e5m2):
            c_dtype = PREC_TORCH_TO_TRITON[torch.float16]
            acc_dtype = PREC_TORCH_TO_TRITON[torch.float32]
            acc_is_float = True
            C = torch.empty((M, N), device=A.device, dtype=torch.float16)
        #INT8 check
        elif (A.dtype == torch.int8):
            activation = ""
            c_dtype = PREC_TORCH_TO_TRITON[torch.int32]
            acc_dtype = PREC_TORCH_TO_TRITON[torch.int32]
            acc_is_float = False
            C = torch.empty((M, N), device=A.device, dtype=torch.int32)
    else:
        raise ValueError(f"Unsupported dtype combination: A={A.dtype}, B={B.dtype}")
       
    if (c_dtype is None):
        raise ValueError("C tensor's dtype is None")
    if (acc_dtype is None):
        raise ValueError("acc tensor's dtype is None")
    if (acc_is_float is None):
        raise ValueError("acc_is_float is not specified.")
        
    # --- 1D GRID ---
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )
    strides = (A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1))

    # --- OVERRIDE MODE ---
    if override_config is not None:
        meta = override_config.copy()
        num_warps = meta.pop("num_warps", None)
        num_stages = meta.pop("num_stages", None)
        matmul_kernel[grid(meta)](
            A, B, C,
            M, N, K,
            *strides,
            **meta,
            ACTIVATION=activation,
            C_DTYPE=c_dtype,
            ACC_DTYPE=acc_dtype, ACC_IS_FLOAT=acc_is_float,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return C

    # --- AUTOTUNE MODE ---
    if use_autotune:
        matmul_kernel_autotuned[grid](
            A, B, C,
            M, N, K,
            *strides,
            ACTIVATION=activation,
            C_DTYPE=c_dtype,
            ACC_DTYPE=acc_dtype, ACC_IS_FLOAT=acc_is_float
        )
        return C

    # --- DEFAULT FIRST CONFIG ---
    configs = get_autotune_config()
    config = configs[0]
    meta = config.kwargs.copy()
    num_warps = config.num_warps
    num_stages = config.num_stages
    matmul_kernel[grid(meta)](
        A, B, C,
        M, N, K,
        *strides,
        **meta,
        ACTIVATION=activation,
        C_DTYPE=c_dtype,
        ACC_DTYPE=acc_dtype, ACC_IS_FLOAT=acc_is_float,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C
