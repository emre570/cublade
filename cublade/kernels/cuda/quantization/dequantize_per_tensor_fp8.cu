// Per-tensor FP8 (E4M3) dequantizer. Single-pass elementwise:
//   y = (FP8 -> float) * scale, narrowed to half or bfloat16.
//
// Each thread loads VEC=8 FP8 bytes via one uint64 (8 B) and stores 8
// elements via one int4 (16 B). FP32 multiply path matches the CPU
// reference bit-exactly under round-to-nearest-even.

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

template <typename T>
__global__ void dequantize_fp8(
    const __nv_fp8_storage_t* __restrict__ q,
    const float* __restrict__ scale,
    T* __restrict__ y,
    int64_t n)
{
    using Traits = DTypeTraits<T>;

    const float s = __ldg(scale);

    const int64_t base = (int64_t)blockIdx.x * blockDim.x * VEC
                       + (int64_t)threadIdx.x * VEC;

    if (base + VEC <= n) {
        const uint64_t packed =
            __ldg(reinterpret_cast<const uint64_t*>(q + base));

        uint16_t p[4];
        p[0] = (uint16_t)( packed        & 0xFFFFu);
        p[1] = (uint16_t)((packed >> 16) & 0xFFFFu);
        p[2] = (uint16_t)((packed >> 32) & 0xFFFFu);
        p[3] = (uint16_t)((packed >> 48) & 0xFFFFu);

        Pack16<T> pk;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            typename Traits::vec2 v2 = Traits::fp8x2_to_vec2(p[k]);
            float2 f2 = Traits::to_float2(v2);
            f2.x *= s;
            f2.y *= s;
            pk.h2[k] = Traits::from_float2(f2);
        }

        *reinterpret_cast<int4*>(y + base) = pk.raw;
    } else {
        int64_t end = (base + VEC < n) ? (base + VEC) : n;
        for (int64_t k = base; k < end; ++k) {
            __nv_fp8_e4m3 fp8; fp8.__x = q[k];
            float v = static_cast<float>(fp8) * s;
            y[k] = Traits::from_float(v);
        }
    }
}

template __global__ void dequantize_fp8<half>(const __nv_fp8_storage_t*, const float*, half*, int64_t);
template __global__ void dequantize_fp8<__nv_bfloat16>(const __nv_fp8_storage_t*, const float*, __nv_bfloat16*, int64_t);

template <typename T>
static void launch_dequantize(at::Tensor& q, at::Tensor& scale, at::Tensor& y, int64_t n) {
    const int grid = fp8_grid(n);
    dequantize_fp8<T><<<grid, BLOCK>>>(
        reinterpret_cast<__nv_fp8_storage_t*>(q.data_ptr()),
        scale.data_ptr<float>(),
        reinterpret_cast<T*>(y.data_ptr()),
        n);
}

// scale is the per-tensor scalar produced by quantize_per_tensor_fp8
// (= amax / 448, broadcast across the whole tensor). out_dtype picks the
// element type: at::kHalf or at::kBFloat16.
at::Tensor
dequantize_per_tensor_fp8_cuda(at::Tensor q,
                               at::Tensor scale,
                               c10::ScalarType out_dtype) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn,
                "q must be torch.float8_e4m3fn");
    TORCH_CHECK(scale.is_cuda(), "scale must be on CUDA");
    TORCH_CHECK(scale.scalar_type() == at::kFloat,
                "scale must be torch.float32");
    TORCH_CHECK(scale.numel() == 1, "scale must be a 0-d or 1-element tensor");
    TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16,
                "out_dtype must be torch.float16 or torch.bfloat16");

    int64_t n = q.numel();
    auto y = torch::empty(q.sizes(), q.options().dtype(out_dtype));

    if (out_dtype == at::kHalf) {
        launch_dequantize<half>(q, scale, y, n);
    } else {
        launch_dequantize<__nv_bfloat16>(q, scale, y, n);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_per_tensor_fp8", &dequantize_per_tensor_fp8_cuda,
          "Per-tensor FP8 e4m3 dequantization (CUDA)",
          py::arg("q"), py::arg("scale"), py::arg("out_dtype"));
}
