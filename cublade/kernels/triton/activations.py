import triton
import triton.language as tl

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)