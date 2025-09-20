import shutil
from pathlib import Path
from torch.utils.cpp_extension import load

BASE_DIR = Path(__file__).parent.resolve()
KERNELS_PATH = BASE_DIR / "cublade" / "_kernels"
CUDA_SRC_PATH = BASE_DIR / "cublade" / "kernels" / "cuda" 

KERNELS_PATH.mkdir(exist_ok=True)
(CUDA_SRC_PATH).mkdir(parents=True, exist_ok=True)

KERNELS = {
    "avg_pool_1d": [str(CUDA_SRC_PATH / "avg_pool_1d.cu")],
    "wmma_matmul": [str(CUDA_SRC_PATH / "wmma_matmul.cu")],
}

OUTPUTS: dict = {}

def build_all():
    for name, sources in KERNELS.items():
        mod = load(
            name=f"cublade_{name}",
            sources=sources,
            extra_cuda_cflags=["--use_fast_math"],
            verbose=True
        )
        so_path = Path(mod.__file__)
        shutil.copy(so_path, KERNELS_PATH / so_path.name)
        OUTPUTS[name] = mod
    return OUTPUTS

if __name__ == "__main__":
    build_all()
