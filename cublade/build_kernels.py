# cublade/build_kernels.py
# prod only

import os
from pathlib import Path
from torch.utils.cpp_extension import load
from cublade.utils.path_utils import get_base_paths

ALL_PATHS: dict[Path] = get_base_paths()

KERNELS: dict = {
    "avg_pool_1d": os.path.join(ALL_PATHS["KERNELS_PATH"], "cuda", "avg_pool_1d.cu"),
    # Other kernel paths can be added here
}

OUTPUTS: dict = {}

def build_all() -> dict:
    for name, path in KERNELS.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Kernel source not found: {path}")
        mod = load(
            name=f"cublade_{name}",
            sources=[str(path)],
            extra_cuda_cflags=["--use_fast_math"],
            verbose=True
        )
        OUTPUTS[name] = mod

    return OUTPUTS

if __name__ == "__main__":
    build_all()
