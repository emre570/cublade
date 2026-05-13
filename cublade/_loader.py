"""Lazy JIT loader for cublade CUDA extensions.

Each binding wrapper calls `load_cuda_module(name, sources, ...)` on first
use. The compiled .so is cached by torch under `~/.cache/torch_extensions/`
and memoized in-process by `name`. Failed compiles are remembered so we do
not retry them per call.

Add a new kernel:
    1. Drop a `.cu` file under `cublade/kernels/cuda/<group>/`.
    2. In a wrapper module under `cublade/bindings/cuda/`, call
       `load_cuda_module(name="cublade_<thing>", sources=[CSRC / "<group>" / "thing.cu"])`.
    3. Done. No central registry to update.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch

CSRC: Path = Path(__file__).parent / "kernels" / "cuda"

_loaded: dict[str, object] = {}
_failed: dict[str, str] = {}


def _default_arch() -> str:
    """Resolve `-arch=sm_XX` from the active CUDA device.

    SM120 (Blackwell consumer, e.g. RTX 5080) defaults to `sm_120a` because
    every cublade SM120 kernel needs the architecture-specific PTX features
    (mma.kind::f8f6f4, mma.kind::mxf4nvf4, ...). For SM86/SM89/SM90 we use
    the plain numeric suffix; pass `arch="sm_90a"` explicitly when an
    SM90 kernel needs Hopper-specific intrinsics.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; cannot resolve default arch")
    major, minor = torch.cuda.get_device_capability(0)
    sm = major * 10 + minor
    if sm == 120:
        return "sm_120a"
    return f"sm_{sm}"


def load_cuda_module(
    name: str,
    sources: Sequence[Path | str],
    *,
    arch: str | None = None,
    extra_cflags: Sequence[str] = (),
    extra_includes: Sequence[Path | str] = (),
    extra_ldflags: Sequence[str] = (),
    use_fast_math: bool = True,
    verbose: bool = False,
) -> object:
    """JIT-compile and cache a CUDA extension.

    Args:
        name: torch extension module name (also the JIT cache key).
        sources: list of `.cu` files (and `.cpp` if any) to compile.
        arch: `-arch=` flag, e.g. "sm_120a". Defaults to the device's
            compute capability resolved by `_default_arch`.
        extra_cflags: additional `-D...` / `-O...` flags appended to nvcc.
        extra_includes: additional `-I` paths.
        extra_ldflags: additional linker flags.
        use_fast_math: pass --use_fast_math (default True). Disable for
            kernels that need IEEE-precise division / FMA semantics to match
            a CPU reference bit-exactly (e.g. INT8 quant under
            round-half-to-even).
        verbose: stream nvcc output.

    Returns:
        The loaded module object.

    Raises:
        RuntimeError: If a previous load attempt for `name` already failed.
    """
    cached = _loaded.get(name)
    if cached is not None:
        return cached

    prior_err = _failed.get(name)
    if prior_err is not None:
        raise RuntimeError(
            f"cublade kernel '{name}' failed to compile earlier: {prior_err}"
        )

    from torch.utils.cpp_extension import load

    arch = arch or _default_arch()
    cuda_cflags = [
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        *(["--use_fast_math"] if use_fast_math else []),
        "--expt-relaxed-constexpr",
        *extra_cflags,
    ]
    include_dirs = [str(CSRC), *(str(p) for p in extra_includes)]

    try:
        mod = load(
            name=name,
            sources=[str(s) for s in sources],
            extra_cuda_cflags=cuda_cflags,
            extra_include_paths=include_dirs,
            extra_ldflags=list(extra_ldflags) if extra_ldflags else None,
            verbose=verbose,
        )
    except Exception as exc:
        _failed[name] = str(exc)
        raise

    _loaded[name] = mod
    return mod


def is_loaded(name: str) -> bool:
    return name in _loaded


def load_error(name: str) -> str | None:
    return _failed.get(name)
