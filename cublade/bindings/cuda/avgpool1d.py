# cublade/bindings/cuda/avgpool1d.py

import torch

from cublade._loader import CSRC, load_cuda_module

_SOURCES = [CSRC / "avg_pool_1d.cu"]
_MODULE_NAME = "cublade_avg_pool_1d"


def _module():
    return load_cuda_module(name=_MODULE_NAME, sources=_SOURCES)


def avg_pool_1d(input: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Python binding: Performs 1D average pooling on the input tensor.
    """
    return _module().avg_pool_1d(input, kernel_size, stride)
