"""Per-tensor / per-channel / per-group INT8 quant + dequant round-trip demo.

Runs the full pipeline through cublade's CUDA kernels and reports the
round-trip error budget so you can sanity check the INT8 path on your GPU.
Defaults to BF16 inputs/outputs.
"""

import torch
from rich import print as pr
from rich.console import Console
from rich.table import Table

from cublade.quantization import dequantize_tensor, quantize_tensor

CONSOLE = Console()


def round_trip(x: torch.Tensor, *, mode: str, **kwargs):
    qt = quantize_tensor(x, dtype=torch.int8, mode=mode, **kwargs)
    y = dequantize_tensor(qt)
    return qt, y


def report_row(name: str, x: torch.Tensor, y: torch.Tensor, scale_summary: str) -> tuple[str, ...]:
    diff = (y.float() - x.float()).abs()
    denom = x.float().abs().max().clamp_min(1e-12)
    rel_max = (diff / denom).max().item()
    rel_mean = (diff / denom).mean().item()
    return (
        name,
        f"{tuple(x.shape)}",
        str(x.dtype).replace("torch.", ""),
        scale_summary,
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

    # -----------------------------------------------------------------------
    # Per-tensor
    # -----------------------------------------------------------------------
    pt_cases = [
        ("toy [-1..1]/255",  torch.linspace(-1.0, 1.0, 256, dtype=bf16, device=device)),
        ("1M random N(0,3)", torch.randn(1_000_000, dtype=bf16, device=device) * 3.0),
        ("16M random",       torch.randn(16_000_000, dtype=bf16, device=device) * 3.0),
        ("single outlier",   torch.full((10_000,), 0.01, dtype=bf16, device=device)),
        ("2049 (tail path)", torch.randn(2049, dtype=bf16, device=device)),
        ("fp16 parity",      torch.randn(1_000_000, dtype=torch.float16, device=device) * 3.0),
    ]
    pt_cases[3][1][5000] = 99.5

    pt_table = Table(title="cublade INT8 per-tensor round-trip", header_style="bold")
    for col in ("case", "shape", "dtype", "scale", "rel_max", "rel_mean"):
        pt_table.add_column(col, justify="right")
    for name, x in pt_cases:
        qt, y = round_trip(x.contiguous(), mode="tensor")
        pt_table.add_row(*report_row(name, x, y, f"{qt.scale.item():.4e}"))

    # -----------------------------------------------------------------------
    # Per-channel (ch_axis=0)
    # -----------------------------------------------------------------------
    pc_cases = [
        ("8x1024 weight",    torch.randn(8, 1024, dtype=bf16, device=device)),
        ("1024x1024 weight", torch.randn(1024, 1024, dtype=bf16, device=device)),
        ("4096x4096 weight", torch.randn(4096, 4096, dtype=bf16, device=device)),
    ]
    pc_table = Table(title="cublade INT8 per-channel (ch_axis=0)", header_style="bold")
    for col in ("case", "shape", "dtype", "scale_range", "rel_max", "rel_mean"):
        pc_table.add_column(col, justify="right")
    for name, x in pc_cases:
        qt, y = round_trip(x.contiguous(), mode="channel", ch_axis=0)
        s_lo, s_hi = qt.scale.min().item(), qt.scale.max().item()
        pc_table.add_row(*report_row(name, x, y, f"{s_lo:.2e}..{s_hi:.2e}"))

    # -----------------------------------------------------------------------
    # Per-group
    # -----------------------------------------------------------------------
    pg_cases = [
        ("1024-vec G=32",       torch.randn(1024, dtype=bf16, device=device),       {"group_size": 32,  "ch_axis": 0}),
        ("1024x1024 G=128 ax-1", torch.randn(1024, 1024, dtype=bf16, device=device), {"group_size": 128, "ch_axis": -1}),
        ("4096x4096 G=32 ax-1",  torch.randn(4096, 4096, dtype=bf16, device=device), {"group_size": 32,  "ch_axis": -1}),
    ]
    pg_table = Table(title="cublade INT8 per-group", header_style="bold")
    for col in ("case", "shape", "dtype", "scale_range", "rel_max", "rel_mean"):
        pg_table.add_column(col, justify="right")
    for name, x, kwargs in pg_cases:
        qt, y = round_trip(x.contiguous(), mode="group", **kwargs)
        s_lo, s_hi = qt.scale.min().item(), qt.scale.max().item()
        pg_table.add_row(*report_row(name, x, y, f"{s_lo:.2e}..{s_hi:.2e}"))

    CONSOLE.print(f"GPU: [cyan]{torch.cuda.get_device_name(0)}[/cyan]\n")
    CONSOLE.print(pt_table)
    CONSOLE.print(pg_table)
    CONSOLE.print(pc_table)

    pr("\n[bold]Toy per-tensor round-trip values:[/bold]")
    x = pt_cases[0][1]
    qt, y = round_trip(x.contiguous(), mode="tensor")
    pr({
        "x[:10]":     x[:10].float().tolist(),
        "q[:10]":     qt.data[:10].tolist(),
        "y[:10]":     y[:10].float().tolist(),
        "scale":      qt.scale.item(),
    })


if __name__ == "__main__":
    main()
