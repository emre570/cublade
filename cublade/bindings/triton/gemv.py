import torch
import triton

from cublade.kernels.triton import gemv_kernel

device = triton.runtime.driver.active.get_active_torch_device()

def gemv(A: torch.Tensor, x: torch.Tensor):
    m, n = A.shape
    stride_am, stride_an = A.stride()
    output = torch.empty(m, device=device)
    
    assert x.shape[0] == A.shape[1], "Vector and input tensor shape mismatch"
    assert A.device == device and x.device == device and output.device == device, "Tensors must be on CUDA"
    grid = lambda meta: (m, )
    
    gemv_kernel[grid](A, x, output, m, n, stride_am, stride_an, BLOCK_SIZE=1024)
    
    return output