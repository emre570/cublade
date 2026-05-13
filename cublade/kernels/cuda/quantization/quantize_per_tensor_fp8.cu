// Per-tensor symmetric FP8 (E4M3) quantizer. Two-pass:
//   amax_reduce: amax = max(|x|) over the whole tensor (atomicMax).
//   quantize_cast: writes q = round(x * 448 / amax) and scale = amax / 448.
//
// Each thread loads VEC=8 elements via one int4 (128-bit). The input dtype is
// half or __nv_bfloat16, dispatched at the pybind entry. The FP32 multiply
// path is required to stay bit-exact with the PyTorch reference at
// cublade/quantization/quantizers.py:quantize_per_tensor_fp8.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "fp8_pack.cuh"

using cublade::quantization::BLOCK;
using cublade::quantization::VEC;
using cublade::quantization::Pack16;
using cublade::quantization::DTypeTraits;
using cublade::quantization::fp8_grid;

__device__ float atomicMaxFloat(float* addr, float val) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Reduction routes through float for both half and bfloat16 - __hmax2 is
// half2-only, and at 92% DRAM busy the extra fmaxf path is in the noise.
template <typename T>
__global__ void amax_reduce_fp8(
    const T* __restrict__ x,
    float* __restrict__ amax_buf,
    int64_t n)
{
    using Traits = DTypeTraits<T>;

    const int64_t base = (int64_t)blockIdx.x * blockDim.x * VEC
                       + (int64_t)threadIdx.x * VEC;
    float my_val = 0.0f;

    if (base + VEC <= n) {
        Pack16<T> pk;
        pk.raw = __ldg(reinterpret_cast<const int4*>(x + base));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f = Traits::to_float2(pk.h2[k]);
            my_val = fmaxf(my_val, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    } else {
        int64_t end = (base + VEC < n) ? (base + VEC) : n;
        for (int64_t k = base; k < end; ++k) {
            my_val = fmaxf(my_val, fabsf(Traits::to_float(x[k])));
        }
    }

    float v = 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = __shfl_xor_sync(0xffffffff, my_val, offset);
        my_val = fmaxf(my_val, v);
    }

    __shared__ float warp_max[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_max[warp_id] = my_val;
    __syncthreads();

    if (warp_id == 0) {
        my_val = (lane < 8) ? warp_max[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            v = __shfl_xor_sync(0xffffffff, my_val, offset);
            my_val = fmaxf(my_val, v);
        }
        if (lane == 0) atomicMaxFloat(amax_buf, my_val);
    }
}

template <typename T>
__global__ void quantize_cast_fp8(
    const T* __restrict__ x,
    const float* __restrict__ amax_buf,
    __nv_fp8_storage_t* __restrict__ q,
    float* __restrict__ scale_out,
    int64_t n,
    float eps)
{
    using Traits = DTypeTraits<T>;

    float scale = 448.0f / fmaxf(*amax_buf, eps);

    if (blockIdx.x == 0 && threadIdx.x == 0)
        *scale_out = 1.0f / scale;

    const int64_t base = (int64_t)blockIdx.x * blockDim.x * VEC
                       + (int64_t)threadIdx.x * VEC;

    if (base + VEC <= n) {
        Pack16<T> pk;
        pk.raw = __ldg(reinterpret_cast<const int4*>(x + base));

        __nv_fp8x2_storage_t p[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f2 = Traits::to_float2(pk.h2[k]);
            f2.x *= scale;
            f2.y *= scale;
            p[k] = __nv_cvt_float2_to_fp8x2(f2, __NV_SATFINITE, __NV_E4M3);
        }

        uint64_t packed = (uint64_t)p[0]
                        | ((uint64_t)p[1] << 16)
                        | ((uint64_t)p[2] << 32)
                        | ((uint64_t)p[3] << 48);
        *reinterpret_cast<uint64_t*>(q + base) = packed;
    } else {
        int64_t end = (base + VEC < n) ? (base + VEC) : n;
        for (int64_t k = base; k < end; ++k) {
            float v = Traits::to_float(x[k]) * scale;
            q[k] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// Explicit instantiations - both versions compile into the same cubin.
template __global__ void amax_reduce_fp8<half>(const half*, float*, int64_t);
template __global__ void amax_reduce_fp8<__nv_bfloat16>(const __nv_bfloat16*, float*, int64_t);
template __global__ void quantize_cast_fp8<half>(const half*, const float*, __nv_fp8_storage_t*, float*, int64_t, float);
template __global__ void quantize_cast_fp8<__nv_bfloat16>(const __nv_bfloat16*, const float*, __nv_fp8_storage_t*, float*, int64_t, float);

template <typename T>
static void launch_quantize(at::Tensor& x,
                            at::Tensor& q,
                            at::Tensor& amax,
                            at::Tensor& scale,
                            int64_t n,
                            float eps) {
    const int grid = fp8_grid(n);
    amax_reduce_fp8<T><<<grid, BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        n);
    quantize_cast_fp8<T><<<grid, BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        reinterpret_cast<__nv_fp8_storage_t*>(q.data_ptr()),
        scale.data_ptr<float>(),
        n,
        eps);
}

// Returns (q[float8_e4m3fn, same shape as x], scale[float32, ()]).
std::tuple<at::Tensor, at::Tensor>
quantize_per_tensor_fp8_cuda(at::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x must be torch.float16 or torch.bfloat16");

    int64_t n = x.numel();

    auto q     = torch::empty(x.sizes(), x.options().dtype(at::kFloat8_e4m3fn));
    auto scale = torch::zeros({}, x.options().dtype(torch::kFloat32));
    auto amax  = torch::zeros({}, x.options().dtype(torch::kFloat32));

    if (x.scalar_type() == at::kHalf) {
        launch_quantize<half>(x, q, amax, scale, n, static_cast<float>(eps));
    } else {
        launch_quantize<__nv_bfloat16>(x, q, amax, scale, n, static_cast<float>(eps));
    }

    return std::make_tuple(q, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_per_tensor_fp8", &quantize_per_tensor_fp8_cuda,
          "Per-tensor FP8 e4m3 quantization (CUDA)",
          py::arg("x"), py::arg("eps") = 1e-12);
}
