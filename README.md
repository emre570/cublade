# cuBlade: Modular CUDA kernel library

## Updates

Aug 3: Triton Tiled Matrix Multiplication Kernel

* Achieved 44 TFLOPS with RTX 3070 Ti, with FP16 and FP8 precisions (FP8 needs SM89 and above for NVIDIA).
* Supports both NVIDIA CUDA and AMD ROCm GPUs
* See `/examples` and try the kernel yourself

A growing toolbox of GPU blades:
from matmul to softmax, each operator is hand-tuned, lightweight, and production-ready.
Plug in only what you need, or forge your own.

## Installation

uv is recommended for installation, since it's faster than pip.

First, create a virtual enviroment:

`uv venv blade`

Then, activate the enviroment:

`source blade/bin/activate`

Then, you must install PyTorch on your own based on your CUDA version:

`uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

Then, install dependencies:

`uv sync`

Finally, install the package locally in your enviroment:

`uv pip install -e . --no-build-isolation`

You should be all done!

## Usage

You can find example usage scripts inside examples folder.
