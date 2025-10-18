"""Compare Triton matmul with torch.matmul across multiple dtypes and report metrics."""

import math
from typing import Iterable, Tuple

import torch
import triton
from rich.console import Console
from rich.table import Table
from triton.testing import do_bench

from cublade.benchmark.triton_configs_checkers import is_cuda
from cublade.bindings.triton import matmul

DEVICE = triton.runtime.driver.active.get_active_torch_device()
CONSOLE = Console()
TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

def fp8_supported() -> bool:
    """Return True when torch exposes FP8 and the active device can run it."""
    if not TORCH_HAS_FP8 or not is_cuda():
        return False
    try:
        capability = torch.cuda.get_device_capability(torch.device(DEVICE))
    except RuntimeError:
        return False
    # Hopper+ (sm90) is required for accelerated FP8 kernels.
    major, _ = capability
    return major >= 9


def compute_tflops(ms: float, m: int, n: int, k: int) -> float:
    if math.isclose(ms, 0.0):
        return float("nan")
    flops = 2 * m * n * k
    return flops / (ms * 1e-3) / 1e12


def max_abs_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    """Return the maximum absolute difference between Triton and reference outputs."""
    return (output.to(torch.float32) - reference.to(torch.float32)).abs().max().item()


def generate_inputs(dtype: torch.dtype, m: int, n: int, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Triton inputs and reference tensors for the requested dtype."""
    if dtype == torch.int8:
        # INT8 path uses symmetric ranges; reference math happens in float32.
        a = torch.randint(-128, 128, (m, k), device=DEVICE, dtype=dtype)
        b = torch.randint(-128, 128, (k, n), device=DEVICE, dtype=dtype)
        ref_a = a.to(torch.float32)
        ref_b = b.to(torch.float32)
    elif dtype == torch.float8_e5m2:
        # FP8 tensors are produced via a float16 staging buffer for reproducibility.
        base_a = torch.randn((m, k), device=DEVICE, dtype=torch.float16)
        base_b = torch.randn((k, n), device=DEVICE, dtype=torch.float16)
        a = base_a.to(torch.float8_e5m2)
        b = base_b.to(torch.float8_e5m2)
        ref_a = a.to(torch.float16)
        ref_b = b.to(torch.float16)
    else:
        # Default path covers float32/float16/bfloat16 flavors.
        a = torch.randn((m, k), device=DEVICE, dtype=dtype)
        b = torch.randn((k, n), device=DEVICE, dtype=dtype)
        ref_a = a
        ref_b = b
    return a, b, ref_a, ref_b


def benchmark_case(
    dtype: torch.dtype,
    shape: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Tuple[float, float, float], Tuple[float, float, float]]:
    """Run Triton and Torch matmul for a single dtype/shape combination.

    Returns
    -------
    shape : tuple
        Dimensions (M, N, K).
    triton_metrics : tuple
        (latency_ms, tflops, max_abs_error) for Triton.
    torch_metrics : tuple
        (latency_ms, tflops, placeholder_error) for Torch (error fixed to 0).
    """
    m, n, k = shape
    a, b, ref_a, ref_b = generate_inputs(dtype, m, n, k)

    # Benchmark Triton kernel and keep the latency in milliseconds.
    triton_ms = do_bench(lambda: matmul(a, b), warmup=20, rep=100)
    triton_output = matmul(a, b)

    error = float("nan")
    torch_ms = float("nan")
    torch_tflops = float("nan")
    if dtype != torch.int8:
        torch_ms = do_bench(lambda: torch.matmul(ref_a, ref_b), warmup=20, rep=100)
        torch_output = torch.matmul(ref_a, ref_b)
        error = max_abs_error(triton_output, torch_output)
        torch_tflops = compute_tflops(torch_ms, m, n, k)
    else:
        torch_output = ref_a @ ref_b
        torch_output = torch_output.to(torch.int32)
        error = max_abs_error(triton_output, torch_output)

    triton_tflops = compute_tflops(triton_ms, m, n, k)
    return (m, n, k), (triton_ms, triton_tflops, error), (torch_ms, torch_tflops, 0.0)


def run(dtype: torch.dtype, shapes: Iterable[Tuple[int, int, int]]):
    """Build and print a Rich table summarizing the benchmark for `dtype`."""
    table = Table(title=f"Triton matmul vs Torch ({dtype})", header_style="bold")
    for column in ("M", "N", "K", "provider", "ms", "TFLOPS", "error_ref"):
        table.add_column(column, justify="right")

    for shape in shapes:
        try:
            (m, n, k), triton_metrics, torch_metrics = benchmark_case(dtype, shape)
            tri_ms, tri_tflops, tri_error = triton_metrics
            torch_ms, torch_tflops, torch_error = torch_metrics

            # Triton row with real error metric.
            table.add_row(
                str(m),
                str(n),
                str(k),
                "Triton",
                f"{tri_ms:.3f}",
                f"{tri_tflops:.2f}",
                f"{tri_error:.3e}",
            )

            # Torch row is omitted for int8 where reference kernel is missing.
            if not math.isnan(torch_ms):
                table.add_row(
                    str(m),
                    str(n),
                    str(k),
                    "Torch",
                    f"{torch_ms:.3f}",
                    f"{torch_tflops:.2f}",
                    f"{torch_error:.3e}",
                )
        except RuntimeError as exc:  # pragma: no cover - exposes launch issues without crashing
            m, n, k = shape
            table.add_row(
                str(m),
                str(n),
                str(k),
                "error",
                "error",
                "error",
                str(exc),
            )

    CONSOLE.print(table)


def main():
    torch.manual_seed(0)

    shapes = [(2048, 2048, 2048)]  # single large shape keeps the output compact
    dtype_suite = [torch.float32, torch.float16]
    if is_cuda() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        dtype_suite.append(torch.bfloat16)
    elif torch.device(DEVICE).type == "cpu" and hasattr(torch, "bfloat16"):
        dtype_suite.append(torch.bfloat16)

    for dtype in dtype_suite:
        run(dtype, shapes)

    if fp8_supported():
        CONSOLE.print("[bold green]FP8 is supported, starting benchmark.[/bold green]")
        run(torch.float8_e5m2, shapes)
    else:
        CONSOLE.print("[bold red]FP8 is not supported, skipping FP8 benchmark.[/bold red]")

    run(torch.int8, shapes)


if __name__ == "__main__":
    main()
