# setup.py
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0+PTX"

base_dir = Path(__file__).parent  # /workspace/cublade
kernel_dir = base_dir / "cublade" / "kernels"

cu_sources = [
    # /workspace/cublade/cublade/kernels/...  ->  cublade/kernels/...
    str(path.relative_to(base_dir).as_posix())
    for path in kernel_dir.rglob("*.cu")
]

print("Going to compile these CUDA sources:", cu_sources)

setup(
    name="cublade",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    ext_modules=[
        CUDAExtension(
            name="cublade._kernels",
            sources=cu_sources,
            extra_compile_args={"nvcc": ["--use_fast_math", "-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
