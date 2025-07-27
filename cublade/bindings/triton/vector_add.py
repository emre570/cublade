import torch
import triton
from cublade.kernels.triton import vector_add_kernel

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.device == DEVICE and y.device == DEVICE, "Tensors must be on CUDA"
    
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    
    return output