#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void avg_pool_1d_kernel(
    const T* __restrict__ input, int input_len,
    T* output, int out_len,
    int kernel_size, int stride
){
    extern __shared__ float smem[];
    int tid = threadIdx.x, bid = blockIdx.x;
    int block_offset = bid * blockDim.x * stride;
    int global_i = block_offset + tid * stride;

    // Load data to smem
    for (int i = tid; i < blockDim.x * stride + kernel_size; i += blockDim.x){
        int input_idx = block_offset + i;
        smem[i] = (input_idx < input_len) ? input[input_idx] : 0.0f;
    }
    __syncthreads();

    if (global_i + kernel_size <= input_len){
        float sum = 0.0f;
        int local_start = tid * stride;

        for (int k = 0; k < kernel_size; ++k){
            sum += smem[local_start + k];
        }

        output[global_i / stride] = sum / kernel_size;
    }
}

torch::Tensor avg_pool_1d(
    const torch::Tensor& input, int kernel_size, int stride
){
    int input_len = input.size(0);

    auto output_tensor = torch::empty(
        {(input_len - kernel_size) / stride + 1}, 
        input.options()
    );
    int out_len = output_tensor.size(0);

    dim3 block_dim(1024);
    dim3 grid_dim((out_len + block_dim.x - 1) / block_dim.x);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avg_pool_1d", [&] {
        avg_pool_1d_kernel<scalar_t><<<grid_dim, block_dim, (block_dim.x * stride + kernel_size) * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(), input_len,
            output_tensor.data_ptr<scalar_t>(), out_len,
            kernel_size, stride
        );
    });

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool_1d", &avg_pool_1d, "Average Pooling 1D");
}