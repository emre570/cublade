# cuBlade: Modular CUDA kernel library

## Updates

### **Sep 20-21** 
#### Quantizations and Precisions Update!
* Introduced `QuantizedTensor` and high-level int8 quantize/dequant helpers as `cublade.quantization` with a new `examples/quant_dequant_int8.py` walkthrough.

* Triton matmul now supports FP32/FP16/BF16/FP8/INT8 through a unified wrapper, with optional autotune bypass for manual configs.

* Added Rich-based benchmarking script (`examples/matmul_simple.py`) that compares Triton against `torch.matmul`, reports TFLOPS/error, and skips FP8 when hardware is missing.

## Description
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
