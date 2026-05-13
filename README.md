# cuBlade: Modular CUDA kernel library

## Updates

### **May 13**

#### Per-tensor FP8 (E4M3) quantizer + dequantizer, INT8 CUDA kernels

* New per-tensor **FP8 (E4M3)** quantizer and dequantizer in `cublade.quantization`:
  - BF16 default path, FP16 retained as legacy (pass `dtype=torch.float16` to dequant for the FP16 path)
  - Vectorized 128-bit (`int4`) loads, hardware-intrinsic FP8 conversion, 64-bit packed stores
  - Two-pass kernel: `amax_reduce` (atomicMax over the tensor) then `quantize_cast` (saturating round to E4M3)
  - Bit-exact against the PyTorch reference (uint16-level equality gate, 16/16 tests pass)
  - NCU on RTX 5080 (`sm_120a`): amax 92% / cast 91% / dequant 85-92% DRAM-busy at the roofline
  - Demo: `examples/quantization/quant_dequant_fp8.py`
* New **INT8** CUDA kernels covering per-tensor, per-channel (axis-configurable), and per-group quant/dequant
  - Drops the prior torch-math fallback; the INT8 path is now full-CUDA
  - Demo: `examples/quantization/quant_dequant_int8.py`
* Unified entry points: `quantize_tensor(x, dtype=..., mode=...)` and `dequantize_tensor(qt)` dispatch to the right kernel by dtype + mode
* Source layout: production kernels under `cublade/kernels/cuda/quantization/`; shared `fp8_pack.cuh` / `int8_pack.cuh` headers expose `DTypeTraits<T>` + `Pack16<T>` for use by upcoming RMSNorm-fused-quant work

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

This creates a `.venv`, installs PyTorch with matching CUDA, and installs cublade as metadata only. CUDA kernels JIT-compile on first use.

### Manual Install

If you prefer manual control:

```bash
# Create and activate venv
uv venv && source .venv/bin/activate

# Install PyTorch for your CUDA version (check yours with: nvcc --version)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130  # CUDA 13.0
# Other options: cu118, cu121, cu124, cu128

# Install cublade (sub-second; no kernel compilation here)
uv pip install -e .
```

## How CUDA Kernels Are Compiled

`uv pip install -e .` does not invoke nvcc. Each CUDA kernel JIT-compiles on first import via `torch.utils.cpp_extension.load`, with `.so` outputs cached at `~/.cache/torch_extensions/`. First call to a kernel takes roughly 5-30 seconds; subsequent calls (across processes, even after reboot) are instant. To force a rebuild, delete the matching subdirectory under `~/.cache/torch_extensions/`.

## Usage

Activate the environment:

```bash
source .venv/bin/activate
```

Example usage scripts are in the `examples/` folder. Quick FP8 round-trip check:

```bash
uv run python examples/quantization/quant_dequant_fp8.py
```
