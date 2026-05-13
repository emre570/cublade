"""First-import warm-up. Compiles both cublade CUDA kernels via torch's JIT
loader so the user's first kernel call doesn't pay the nvcc cost.

Invoked automatically from cublade/__init__.py on first import. Can also be
run standalone:
    python -m cublade._post_install

Skips quietly when CUDA is unavailable so cublade remains importable for
non-CUDA contexts (docs builds, IDE indexing, CI lints). Real kernel calls
will raise descriptively from torch's extension loader if invoked without
CUDA; this script doesn't pre-empt that error path.

Idempotency: relies on torch.utils.cpp_extension.load's own cache at
~/.cache/torch_extensions/. Calling _module() when the .so already exists is
a fast dlopen (~50ms), not a recompile. We time the call and only print a
summary line when an actual compile happened (threshold 1s).
"""
import sys
import time


def main() -> int:
    try:
        import torch
    except ImportError:
        print("[cublade] PyTorch not importable; skipping kernel warm-up.", file=sys.stderr)
        return 0

    if not torch.cuda.is_available():
        print("[cublade] No CUDA device visible; skipping kernel warm-up.", file=sys.stderr)
        return 0

    from cublade.bindings.cuda.dequantize_fp8 import _module as _dequant_fp8_module
    from cublade.bindings.cuda.quantize_fp8 import _module as _quant_fp8_module
    from cublade.bindings.cuda.dequantize_int8 import _module as _dequant_int8_module
    from cublade.bindings.cuda.quantize_int8 import _module as _quant_int8_module

    t0 = time.perf_counter()
    _quant_fp8_module()
    _dequant_fp8_module()
    _quant_int8_module()
    _dequant_int8_module()
    elapsed = time.perf_counter() - t0

    if elapsed > 1.0:
        print(
            f"[cublade] kernels compiled in {elapsed:.0f}s on {torch.cuda.get_device_name(0)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
