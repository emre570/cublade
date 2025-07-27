import torch
import triton
from cublade.bindings.triton import vector_add

torch.manual_seed(0)

size = 98432
DEVICE = torch.device("cuda")

# Random input tensors
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)

# Triton kernel
output_triton = vector_add(x, y)

# PyTorch baseline
output_torch = x + y

# Comparison
max_diff = torch.max(torch.abs(output_triton - output_torch))
print("Maximum difference between Triton and Torch:", max_diff.item())

# Optional: print first few elements
print("\nTriton output[:5]:", output_triton[:5])
print("Torch output[:5]:  ", output_torch[:5])
