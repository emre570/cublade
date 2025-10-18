"""
W8A16 Quantization - Dummy Model Example

Simple demonstration of W8A16 quantization on a small neural network.
Good starting point to understand the API before using it on LLMs.

Usage:
    uv run python examples/quantization/w8a16_toy_model.py
"""

import torch
import torch.nn as nn
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from cublade.quantization import quantize_model

# ============================================================================
# 1. Define a Simple Model
# ============================================================================

class DummyModel(nn.Module):
    """Simple 3-layer MLP for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ============================================================================
# 2. Create and Quantize Model
# ============================================================================

def main():
    console = Console()
    
    rprint("\n[bold cyan]cuBlade W8A16 Quantization - Dummy Model Demo[/bold cyan]\n")
    
    # Create original model
    torch.manual_seed(42)
    model_fp = DummyModel().eval()
    
    # Create a copy for quantization
    model_w8 = DummyModel().eval()
    model_w8.load_state_dict(model_fp.state_dict())
    
    # Quantize: Convert all Linear layers to W8A16
    rprint("[yellow]Quantizing model...[/yellow]")
    model_w8 = quantize_model(
        model_w8,
        quant_type="w8a16",
        inplace=True  # Modify in place
    )
    
    rprint("[green]✓ Quantization complete![/green]\n")
    
    # ========================================================================
    # 3. Test Accuracy
    # ========================================================================
    
    # Generate random test input
    x = torch.randn(4, 128)
    
    with torch.no_grad():
        out_fp = model_fp(x)
        out_w8 = model_w8(x)
    
    # Calculate error metrics
    mse = (out_fp - out_w8).pow(2).mean().item()
    max_err = (out_fp - out_w8).abs().max().item()
    
    # Display results in a table
    table = Table(title="Accuracy Comparison", show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", justify="right", style="magenta")
    table.add_column("Status", justify="center", style="green")
    
    mse_status = "✓ Good" if mse < 0.01 else "⚠ High"
    max_status = "✓ Good" if max_err < 0.5 else "⚠ High"
    
    table.add_row("MSE", f"{mse:.6f}", mse_status)
    table.add_row("Max Error", f"{max_err:.6f}", max_status)
    
    console.print(table)
    
    # ========================================================================
    # 4. Inspect Quantized Layer
    # ========================================================================
    
    rprint("\n[bold cyan]Inspecting Quantized Layer:[/bold cyan]")
    rprint(f"Original layer type: [blue]{type(model_fp.fc1).__name__}[/blue]")
    rprint(f"Quantized layer type: [green]{type(model_w8.fc1).__name__}[/green]")
    rprint(f"Weight dtype: [yellow]{model_w8.fc1.int8_weights.dtype}[/yellow]")
    rprint(f"Scales dtype: [yellow]{model_w8.fc1.scales.dtype}[/yellow]")
    
    # Calculate memory savings
    def model_size(m):
        return sum(p.numel() * p.element_size() for p in m.parameters())
    
    size_fp = model_size(model_fp)
    size_w8 = model_size(model_w8)
    savings = (size_fp - size_w8) / size_fp * 100
    
    rprint(f"\n[bold green]Memory savings: {savings:.1f}%[/bold green]")
    rprint(f"[dim]({size_fp/1024:.1f} KB → {size_w8/1024:.1f} KB)[/dim]")

if __name__ == '__main__':
    main()

