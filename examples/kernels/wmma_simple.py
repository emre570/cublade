"""Compare WMMA CUDA kernel with torch.matmul across supported precisions."""

import math
from typing import Tuple

import torch
from rich.console import Console
from rich.table import Table
from triton.testing import do_bench

from cublade.bindings.cuda import wmma_gemm

DEVICE = "cuda"
CONSOLE = Console()


def compute_tflops(ms: float, m: int, n: int, k: int) -> float:
    if math.isclose(ms, 0.0):
        return float("nan")
    flops = 2 * m * n * k
    return flops / (ms * 1e-3) / 1e12


def max_abs_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return (output.float() - reference.float()).abs().max().item()


def generate_inputs(dtype: torch.dtype, m: int, n: int, k: int):
    """Create inputs for WMMA and reference tensors."""
    if dtype == torch.int8:
        a = torch.randint(-128, 127, (m, k), device=DEVICE, dtype=dtype)
        b = torch.randint(-128, 127, (k, n), device=DEVICE, dtype=dtype)
        # Use float64 for exact integer arithmetic (float32 loses precision at large K)
        ref_a = a.double()
        ref_b = b.double()
    elif dtype == torch.uint8:
        a = torch.randint(0, 255, (m, k), device=DEVICE, dtype=dtype)
        b = torch.randint(0, 255, (k, n), device=DEVICE, dtype=dtype)
        # Use float64 for exact integer arithmetic (float32 loses precision at large K)
        ref_a = a.double()
        ref_b = b.double()
    else:
        a = torch.randn((m, k), device=DEVICE, dtype=dtype)
        b = torch.randn((k, n), device=DEVICE, dtype=dtype)
        ref_a = a
        ref_b = b
    return a, b, ref_a, ref_b


def benchmark_case(
    dtype: torch.dtype,
    acc_dtype: torch.dtype | None,
    shape: Tuple[int, int, int],
) -> Tuple[float, float, float, float, float]:
    """Benchmark WMMA vs Torch for a single dtype/shape.
    
    Returns: (wmma_ms, wmma_tflops, torch_ms, torch_tflops, error)
    """
    m, n, k = shape
    a, b, ref_a, ref_b = generate_inputs(dtype, m, n, k)

    # Benchmark WMMA
    wmma_ms = do_bench(lambda: wmma_gemm(a, b, acc_dtype), warmup=20, rep=100)
    wmma_output = wmma_gemm(a, b, acc_dtype)

    # Benchmark Torch reference
    if dtype in (torch.int8, torch.uint8):
        # torch.matmul doesn't support int8 directly
        torch_ms = float("nan")
        torch_tflops = float("nan")
        ref_output = (ref_a @ ref_b).to(torch.int32)
    else:
        torch_ms = do_bench(lambda: torch.matmul(ref_a, ref_b), warmup=20, rep=100)
        ref_output = torch.matmul(ref_a, ref_b)
        torch_tflops = compute_tflops(torch_ms, m, n, k)

    wmma_tflops = compute_tflops(wmma_ms, m, n, k)
    error = max_abs_error(wmma_output, ref_output)

    return wmma_ms, wmma_tflops, torch_ms, torch_tflops, error


def run(dtype: torch.dtype, acc_dtype: torch.dtype | None, shapes: list):
    """Build and print a Rich table for the benchmark."""
    acc_str = str(acc_dtype).replace("torch.", "") if acc_dtype else "default"
    dtype_str = str(dtype).replace("torch.", "")
    
    table = Table(
        title=f"WMMA GEMM vs Torch ({dtype_str} -> {acc_str})",
        header_style="bold"
    )
    for col in ("M", "N", "K", "provider", "ms", "TFLOPS", "error"):
        table.add_column(col, justify="right")

    for shape in shapes:
        m, n, k = shape
        try:
            wmma_ms, wmma_tflops, torch_ms, torch_tflops, error = benchmark_case(
                dtype, acc_dtype, shape
            )

            # WMMA row
            table.add_row(
                str(m), str(n), str(k),
                "[cyan]WMMA[/cyan]",
                f"{wmma_ms:.3f}",
                f"{wmma_tflops:.2f}",
                f"{error:.3e}",
            )

            # Torch row (skip for int8/uint8)
            if not math.isnan(torch_ms):
                table.add_row(
                    str(m), str(n), str(k),
                    "[yellow]Torch[/yellow]",
                    f"{torch_ms:.3f}",
                    f"{torch_tflops:.2f}",
                    "0.000e+00",
                )

        except RuntimeError as exc:
            table.add_row(
                str(m), str(n), str(k),
                "[red]error[/red]",
                "-", "-", str(exc)[:50],
            )

    CONSOLE.print(table)
    CONSOLE.print()


def main():
    torch.manual_seed(42)

    # WMMA requires dimensions to be multiples of 16
    shapes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (6144, 6144, 6144),
        (8192, 8192, 8192),
        (10240, 10240, 10240),
        (12288, 12288, 12288),
        (14336, 14336, 14336),
        (16384, 16384, 16384),
        (18432, 18432, 18432),
        (20480, 20480, 20480),
        (22528, 22528, 22528),
    ]

    CONSOLE.print("[bold]WMMA GEMM Benchmark[/bold]")
    CONSOLE.print(f"Device: {torch.cuda.get_device_name()}")
    CONSOLE.print(f"Shapes must be multiples of 16 (WMMA constraint)")
    CONSOLE.print()

    # float16 -> float32 (default)
    run(torch.float16, None, shapes)

    # float16 -> float16
    run(torch.float16, torch.float16, shapes)

    # bfloat16 -> float32
    if torch.cuda.is_bf16_supported():
        run(torch.bfloat16, None, shapes)
    else:
        CONSOLE.print("[yellow]bfloat16 not supported on this device[/yellow]\n")

    # int8 -> int32
    run(torch.int8, None, shapes)

    # uint8 -> int32
    run(torch.uint8, None, shapes)


if __name__ == "__main__":
    main()

