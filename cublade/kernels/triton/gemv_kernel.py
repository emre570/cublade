import triton
import triton.language as tl

@triton.jit
def gemv_kernel(
    A_ptr, x_ptr, y_ptr,
    m, n,
    stride_am, stride_an,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    acc = 0.0
    
    for off in range(0, n, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n
        
        A_offset = row_id * stride_am + cols * stride_an
        x_offset = cols
        
        a = tl.load(A_ptr + A_offset, mask=mask)
        x = tl.load(x_ptr + x_offset, mask=mask)
        
        acc += tl.sum(a*x, axis=0)
    
    tl.store(y_ptr + row_id, acc)