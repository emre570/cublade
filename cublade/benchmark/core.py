# cublade/benchmarks/core.py
import time
import torch

def benchmark(
    kernel_fn,
    ref_fn,
    *args,
    n_flop: int,
    warmup: int = 1,
    atol: float = 1e-5,
    rtol: float = 1e-3,
):
    """
    Compare a custom kernel against a reference implementation.

    Reports               | Kernel | Reference
    ----------------------|--------|----------
    Time (ms)             |   ✔    |    ✔
    GFLOPS                |   ✔    |    ✔
    Max abs error         |   ✔    |    – (n/a)
    allclose (PASS/FAIL)  |   ✔    |    – (REF)

    Other details are returned as a dict for programmatic use.
    """
    # Warm-up
    for _ in range(warmup):
        kernel_fn(*args)
        ref_fn(*args)

    # Reference timing
    t0 = time.perf_counter()
    out_ref = ref_fn(*args)
    t1 = time.perf_counter()
    ms_ref   = (t1 - t0) * 1e3
    gflops_ref = n_flop / (ms_ref * 1e6)

    # Kernel timing
    t2 = time.perf_counter()
    out_k = kernel_fn(*args)
    t3 = time.perf_counter()
    ms_k   = (t3 - t2) * 1e3
    gflops_k = n_flop / (ms_k * 1e6)

    # Accuracy
    max_err = (out_k - out_ref).abs().max().item()
    passed  = torch.allclose(out_k, out_ref, rtol=rtol, atol=atol)

    # Pretty table
    header = f"{'Function':<20} {'Time (ms)':>10} {'GFLOPS':>10} {'max_err':>12} {'Status':>8}"
    line   = "-" * len(header)
    print(header)
    print(line)
    print(f"{ref_fn.__name__:<20} {ms_ref:10.2f} {gflops_ref:10.1f} {'-':>12} {'REF':>8}")
    print(f"{kernel_fn.__name__:<20} {ms_k:10.2f} {gflops_k:10.1f} {max_err:12.2e} "
          f"{'PASS' if passed else 'FAIL':>8}")

    return {
        "kernel": {
            "time_ms":  ms_k,
            "gflops":   gflops_k,
            "max_err":  max_err,
            "passed":   passed,
            "output":   out_k,
        },
        "reference": {
            "time_ms":  ms_ref,
            "gflops":   gflops_ref,
            "output":   out_ref,
        },
    }
