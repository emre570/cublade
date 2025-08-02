import triton

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

# Checker
def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

# Autotune configs for NVIDIA CUDA devices
def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4)
    ]

# Autotune configs for AMD ROCm devices
def get_hip_autotune_config():
    sizes = [
        {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 6},
        {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_M': 4},
        {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 6},
        {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 6},
        {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4},
        {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4},
        {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4},
        {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 6},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]