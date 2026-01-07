# setup.py
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # Set to your GPU arch, or remove for auto-detect

base_dir = Path(__file__).parent
cuda_kernel_dir = base_dir / "cublade" / "kernels" / "cuda"

# Build each .cu file as a separate extension module
# cublade/kernels/cuda/wmma.cu -> cublade._kernels.cublade_wmma
ext_modules = []
for cu_file in cuda_kernel_dir.glob("*.cu"):
    module_name = f"cublade._kernels.cublade_{cu_file.stem}"
    ext_modules.append(
        CUDAExtension(
            name=module_name,
            sources=[str(cu_file.relative_to(base_dir))],
            extra_compile_args={"nvcc": ["--use_fast_math", "-O2"]},
        )
    )

print("Building CUDA extensions:")
for ext in ext_modules:
    print(f"  {ext.name} <- {ext.sources}")

setup(
    name="cublade",
    version="0.3.0",
    packages=find_packages(exclude=["tests*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
