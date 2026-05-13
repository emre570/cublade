import os as _os

from .benchmark import *
from .bindings import *
from .kernels import *
from .quantization import *


def _maybe_compile() -> None:
    """First-import warm-up: compile both CUDA kernels via torch JIT.

    On cache hit (.so files already exist at ~/.cache/torch_extensions/),
    this is sub-100ms and silent. On cold install, prints a one-line summary
    after the ~30s compile completes.

    Skip with CUBLADE_SKIP_WARMUP=1. The CUBLADE_POST_INSTALL_ACTIVE guard
    is vestigial from the setup.py era; kept so `python -m cublade._post_install`
    invocations don't recurse if the script ever re-imports cublade.
    """
    if _os.environ.get("CUBLADE_SKIP_WARMUP") == "1":
        return
    if _os.environ.get("CUBLADE_POST_INSTALL_ACTIVE") == "1":
        return
    try:
        from cublade._post_install import main
        main()
    except Exception as e:
        import sys as _sys
        print(f"[cublade] warm-up skipped: {e}", file=_sys.stderr)


_maybe_compile()
