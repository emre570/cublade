# cuBlade: Modular CUDA kernel library

## Updates

### **Jan 7**

#### WMMA Multi-Precision GEMM Kernel

* New CUDA WMMA kernel with async pipeline and double buffering
* Supports all 16x16x16 fragment precision combinations:
  - `float16 -> float32` (default)
  - `float16 -> float16`
  - `bfloat16 -> float32`
  - `int8 -> int32`
  - `uint8 -> int32`
* Python binding with automatic dtype dispatch
* Benchmark results on RTX 3090:
  - int8: **96 TFLOPS** (exact precision)
  - float16->float16: **88 TFLOPS** (beats cuBLAS at large sizes)
* New example: `examples/kernels/wmma_simple.py`
* Improved install script with CUDA auto-detection

### **Oct 18-19**

#### W8A16 Quantization with LLM.int8 Support!

* Introduced `quantize_model()` high-level API for model quantization
* `cubladeW8A16LinearLayer`: INT8 weights, FP16 activations with ~18% memory savings
* Optional outlier handling (LLM.int8 style) for quality preservation
* Two new examples: `w8a16_toy_model.py` (simple) and `w8a16_llm_inference.py` (full LLM benchmark)
* See `examples/quantization/` for usage

### **Sep 20-21**

#### Quantizations and Precisions Update!

* Introduced `QuantizedTensor` and high-level int8 quantize/dequant helpers as `cublade.quantization` with a new `examples/quant_dequant_int8.py` walkthrough.
* Triton matmul now supports FP32/FP16/BF16/FP8/INT8 through a unified wrapper, with optional autotune bypass for manual configs.
* Added Rich-based benchmarking script (`examples/matmul_simple.py`) that compares Triton against `torch.matmul`, reports TFLOPS/error, and skips FP8 when hardware is missing.

## Description

A growing toolbox of GPU blades:
from matmul to softmax, each operator is hand-tuned, lightweight, and production-ready.
Plug in only what you need, or forge your own.

## Installation

### Quick Install (Recommended)

The install script auto-detects your CUDA version and installs everything:

```bash
./install.sh
```

This creates a `.venv`, installs PyTorch with matching CUDA, and builds all CUDA kernels.

### Manual Install

If you prefer manual control:

```bash
# Create and activate venv
uv venv && source .venv/bin/activate

# Install PyTorch for your CUDA version (check yours with: nvcc --version)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130  # CUDA 13.0
# Other options: cu118, cu121, cu124, cu128

# Install cublade (builds CUDA kernels)
uv pip install -e . --no-build-isolation
```

## Usage

Activate the environment:

```bash
source .venv/bin/activate
```

Example usage scripts are in the `examples/` folder.
