import torch
import triton

from cublade.bindings.triton import gemv
#from cublade.benchmark import benchmark

# Set CUDA device
device = triton.runtime.driver.active.get_active_torch_device()

# Set seed for same numbers
torch.manual_seed(0)

# Set matrix size
size_m, size_n = 1024, 2048

# Fill matrix with random numbers
A = torch.rand((size_m, size_n), device=device)
x = torch.rand(size_n, device=device)

# You must create callable functions if you want use cublade's benchmark script.
def run_torch(A, x):
    return A @ x

def run_triton(A, x):
    return gemv(A, x)

# You must also calculate flops for cublade's benchmark.
#flops = 2 * size_m * size_n

# cublade's benchmark gives ms and gflops performances, as well as compares whether the values obtained are the same
# Not stable, needs work
#benchmark(run_triton, run_torch, A, x, n_flop=flops, warmup=150)

# You can also use Triton's Benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['m'],
        x_vals=[2**i for i in range(10, 20)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GFLOPS',
        plot_name='gemv-gflops-vs-m',
        args={'n': 2048}
    ))
def tr_benchmark(m, n, provider):
    A = torch.rand((m, n), device=device)
    x = torch.rand(n, device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, *_ = triton.testing.do_bench(lambda: A @ x, quantiles=quantiles)
    else:
        ms, *_ = triton.testing.do_bench(lambda: gemv(A, x), quantiles=quantiles)

    flops = 2 * m * n
    gflops = lambda ms: flops / (ms * 1e-3) * 1e-9
    return gflops(ms), gflops(ms), gflops(ms)

# Change save_path for your system
tr_benchmark.run(print_data=True, save_path="/workspace/cublade/tests/triton/")