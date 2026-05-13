"""Per-tensor FP8 (E4M3) quant + dequant round-trip demo.

Runs the full pipeline through cublade's CUDA kernels across a few shapes and
reports the round-trip error budget so you can sanity check the FP8 path on
your GPU. Defaults to BF16 inputs/outputs; the last row exercises the FP16
legacy path for parity.
"""

import torch
from rich import print as pr
from rich.console import Console
from rich.table import Table

from cublade.quantization import dequantize_tensor, quantize_tensor

CONSOLE = Console()


def round_trip(x: torch.Tensor):
    qt = quantize_tensor(x, dtype=torch.float8_e4m3fn, mode="tensor")
    y = dequantize_tensor(qt)
    return qt.data, qt.scale, y


def report_row(name: str, x: torch.Tensor, q, scale, y) -> tuple[str, ...]:
    diff = (y.float() - x.float()).abs()
    rel_max = (diff / x.float().abs().max().clamp_min(1e-12)).max().item()
    rel_mean = (diff / x.float().abs().max().clamp_min(1e-12)).mean().item()
    return (
        name,
        f"{tuple(x.shape)}",
        str(x.dtype).replace("torch.", ""),
        f"{scale.item():.4e}",
        f"{rel_max:.4f}",
        f"{rel_mean:.4f}",
    )


def main():
    if not torch.cuda.is_available():
        pr("[red]CUDA unavailable; this demo needs a GPU[/red]")
        return

    torch.manual_seed(0)
    device = "cuda"
    bf16 = torch.bfloat16

    cases = [
        ("toy [1..8]",       torch.arange(1, 9, dtype=bf16, device=device)),
        ("1M random",        torch.randn(1_000_000, dtype=bf16, device=device) * 3.0),
        ("16M random",       torch.randn(16_000_000, dtype=bf16, device=device) * 3.0),
        ("single outlier",   torch.full((10_000,), 0.01, dtype=bf16, device=device)),
        ("2049 (tail path)", torch.randn(2049, dtype=bf16, device=device)),
        # Legacy parity: keep one FP16 case so regressions on the half path
        # are still visible in this demo.
        ("fp16 parity",      torch.randn(1_000_000, dtype=torch.float16, device=device) * 3.0),
    ]
    cases[3][1][5000] = 99.5

    table = Table(title="cublade FP8 per-tensor round-trip", header_style="bold")
    for col in ("case", "shape", "dtype", "scale", "rel_max", "rel_mean"):
        table.add_column(col, justify="right")

    for name, x in cases:
        q, scale, y = round_trip(x.contiguous())
        table.add_row(*report_row(name, x, q, scale, y))

    CONSOLE.print(f"GPU: [cyan]{torch.cuda.get_device_name(0)}[/cyan]")
    CONSOLE.print(table)

    pr("[bold]Toy round-trip values:[/bold]")
    x = cases[0][1]
    q, scale, y = round_trip(x)
    pr({
        "x":     x.float().tolist(),
        "q hex": [f"0x{b:02x}" for b in q.view(torch.uint8).tolist()],
        "y":     y.float().tolist(),
        "scale": scale.item(),
    })


if __name__ == "__main__":
    main()
