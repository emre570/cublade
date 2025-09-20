# cublade/bindings/pooling/avgpool1d.py

import torch
from cublade._kernels import cublade_avg_pool_1d as _cpp_avg_pool_1d

def avg_pool_1d(input: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Python binding: Performs 1D average pooling on the input tensor.
    """
    return _cpp_avg_pool_1d(input, kernel_size, stride)
