import triton
import triton.language as tl

from cublade.kernels.triton.activations import leaky_relu
from cublade.benchmark.triton_configs_checkers import get_autotune_config

AUTOTUNE_CONFIGS = get_autotune_config()

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  # meta parameters 
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr,
                  ACTIVATION: tl.constexpr,
                  C_DTYPE: tl.constexpr, 
                  ACC_DTYPE: tl.constexpr, ACC_IS_FLOAT: tl.constexpr):
    """Kernel for compute Matrix Multiplication of given two A and B matrices.
    A shape must be: (M, K)
    B shape must be: (K, N)
    C shape will be: (M, N) after A @ B"""
    
    # Program ID
    pid = tl.program_id(axis=0)
    # Number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_M)
    # Number of programs ids along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Number of programs in group
    num_pid_in_group = GROUP_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group
    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_M
    # If `num_pid_m` isn't divisible by `GROUP_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Let compiler assume these values
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    
    # Pointer arithmetic for blocks of A and B.
    # Assign pointers of values for blocks
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # Assign tile's pointers since this is tiled matmul
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Compute block of C matrix with iterating tiles over main matrix
    # For higher accuracy, dot product accumulates as FP32, then converted back to FP16.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        
    # Can fuse activation functions too, for now we have leaky relu.
    # Other activations are WIP.
    if (ACTIVATION == "leaky_relu" and ACC_IS_FLOAT):
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(C_DTYPE)
    
    # Write the block of output matrix C with masks.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
        
matmul_kernel_autotuned = triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K',
         'ACC_DTYPE', 'C_DTYPE'],
)(matmul_kernel)

__all__ = ['matmul_kernel', 'matmul_kernel_autotuned']