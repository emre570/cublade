import torch
import triton

import nvtx

from cublade.bindings.triton import matmul
from cublade.benchmark.triton_configs_checkers import is_cuda

DEVICE = triton.runtime.driver.active.get_active_torch_device()

torch.manual_seed(0)

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

# Check values of tensors between Torch and Triton
def check_torch_triton():
    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5

    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        
    if TORCH_HAS_FP8 and is_cuda():
        torch.manual_seed(0)
        a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
        b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
        a = a.to(torch.float8_e5m2)
        # pre-transpose b for efficiency
        # cuBlade will have it's own transpose kernel soon.
        b = b.T
        b = b.to(torch.float8_e5m2)
        with nvtx.annotate("BENCH_TARGET"):
            triton_output = matmul(a, b)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")
            
# Benchmark and plot the results
def benchmark_and_plot(plot_save_path: str):
    ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'
    configs = []
    for fp8_inputs in [False, True]:
        if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
                x_vals=[128 * i for i in range(2, 65)],  # Different possible values for `x_name`
                line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
                # Possible values for `line_arg`
                # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
                line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
                line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",  # Label name for the y-axis
                plot_name="matmul-performance-" +
                ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
                args={"fp8_inputs": fp8_inputs},
            ))


    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider, fp8_inputs):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        if TORCH_HAS_FP8 and fp8_inputs:
            a = a.to(torch.float8_e5m2)
            b = b.T
            b = b.to(torch.float8_e5m2)
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)


    benchmark.run(show_plots=True, print_data=True, save_path=plot_save_path)
    
if __name__ == '__main__':
    save_path = '/workspace/cublade/tests/matmul/triton'
    check_torch_triton()
    #benchmark_and_plot(save_path)