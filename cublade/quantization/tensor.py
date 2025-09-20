import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class QuantizedTensor:
    data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    mode: str
    axis: int
    group_size: Optional[int]
    dtype: torch.dtype
