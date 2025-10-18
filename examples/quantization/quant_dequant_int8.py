import torch
from rich import print as pr

from cublade.quantization import quantize_tensor, dequantize_tensor

def demo():
    torch.manual_seed(0)

    # Random toy tensor we will quantize
    rand_org = torch.randn(4, 4)
    group_size = 4

    # Per-tensor vs per-channel quantization (axis=1 for channels)
    # Also Per-group grouping along the feature axis (axis=1)
    qt_tensor = quantize_tensor(rand_org, scheme="symmetric", dtype=torch.int8, mode="tensor")
    qt_channel = quantize_tensor(rand_org, scheme="symmetric", dtype=torch.int8, mode="channel", ch_axis=1)
    qt_group = quantize_tensor(rand_org, scheme="symmetric", dtype=torch.int8, mode="group", group_size=group_size, ch_axis=1)

    # cuBlade's QuantizedTensor keeps the quantized payload and the parameters (scale, zero-point,
    # mode, axis, group size) bundled together. This makes dequantization much easier,
    # because we just hand the object back to dequantize_tensor instead of juggling
    # multiple tensors and kwargs.
    pr("[bold cyan]QuantizedTensor metadata[/bold cyan]:", qt_tensor)

    dt_tensor = dequantize_tensor(qt_tensor)
    dt_channel = dequantize_tensor(qt_channel)
    dt_group = dequantize_tensor(qt_group)

    err_t = (dt_tensor - rand_org).pow(2).mean()
    err_c = (dt_channel - rand_org).pow(2).mean()
    err_g = (dt_group - rand_org).pow(2).mean()

    pr("[bold green]Original[/bold green]:\n", f"{rand_org}")
    pr("[bold red]Per-tensor q[/bold red]:\n", f"{qt_tensor.data}")
    pr("[bold blue]Per-tensor dequant[/bold blue]:\n", f"{dt_tensor}")
    pr("[bold red]Per-tensor MSE[/bold red]: ", f"{err_t:.6f}")
    
    pr("[bold]========================================[/bold]")
    
    pr("[bold red]Per-channel q[/bold red]:\n", f"{qt_channel.data}")
    pr("[bold blue]Per-channel dequant[/bold blue]:\n", f"{dt_channel}")
    pr("[bold red]Per-channel MSE[/bold red]: ", f"{err_c:.6f}")
    pr("[bold]========================================[/bold]")

    pr(f"[bold red]Per-group (g={group_size}) q[/bold red]:\n", f"{qt_group.data}")
    pr("[bold blue]Per-group dequant[/bold blue]:\n", f"{dt_group}")
    pr("[bold red]Per-group MSE[/bold red]: ", f"{err_g:.6f}")

if __name__ == '__main__':
    demo()
