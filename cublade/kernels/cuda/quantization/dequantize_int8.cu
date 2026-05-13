// Symmetric INT8 dequantizers (per-tensor, per-channel, per-group).
// Single-pass elementwise:
//   y = T((float)q * scale)
//
// Per-tensor + per-channel load 8 int8s via uint64 (8 B) and store via int4
// (16 B Pack16<T>). Per-group does scalar loads + stride loop because group
// sizes 16-256 are too small for the vec path to pay back the indexing.
// FP32 multiply matches the torch reference bit-exactly under
// round-to-nearest-even narrowing.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "int8_pack.cuh"

using cublade::quantization::BLOCK;
using cublade::quantization::VEC;
using cublade::quantization::Pack16;
using cublade::quantization::DTypeTraits;
using cublade::quantization::fp8_grid;

// =============================================================================
// Per-tensor
// =============================================================================

template <typename T>
__global__ void dequantize_tensor_int8(
    const int8_t* __restrict__ q,
    const float* __restrict__ scale_p,
    T* __restrict__ y,
    int64_t n)
{
    using Traits = DTypeTraits<T>;
    constexpr int lanes = 8;

    const float s = __ldg(scale_p);

    const int64_t base = (int64_t)blockIdx.x * blockDim.x * lanes
                       + (int64_t)threadIdx.x * lanes;

    if (base + lanes <= n) {
        const uint64_t packed = __ldg(reinterpret_cast<const uint64_t*>(q + base));

        Pack16<T> pk;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int8_t a = static_cast<int8_t>((packed >> (16*k))     & 0xFFu);
            int8_t b = static_cast<int8_t>((packed >> (16*k + 8)) & 0xFFu);
            float2 f2;
            f2.x = static_cast<float>(a) * s;
            f2.y = static_cast<float>(b) * s;
            pk.h2[k] = Traits::from_float2(f2);
        }
        *reinterpret_cast<int4*>(y + base) = pk.raw;
    } else if (base < n) {
        int64_t end = (base + lanes < n) ? (base + lanes) : n;
        for (int64_t k = base; k < end; ++k) {
            y[k] = Traits::from_float(static_cast<float>(q[k]) * s);
        }
    }
}

// =============================================================================
// Per-channel. q viewed as (C, F) row-major contiguous, scale shape (C,).
// =============================================================================

template <typename T>
__global__ void dequantize_channel_int8(
    const int8_t* __restrict__ q,
    const float* __restrict__ scale,
    T* __restrict__ y,
    int64_t F)
{
    using Traits = DTypeTraits<T>;
    constexpr int lanes = 8;

    const int64_t c = blockIdx.x;
    const float s = __ldg(scale + c);

    const int64_t row_base = c * F;
    const int64_t base = row_base
                       + (int64_t)blockIdx.y * blockDim.x * lanes
                       + (int64_t)threadIdx.x * lanes;
    const int64_t row_end = row_base + F;

    if (base + lanes <= row_end) {
        const uint64_t packed = __ldg(reinterpret_cast<const uint64_t*>(q + base));

        Pack16<T> pk;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int8_t a = static_cast<int8_t>((packed >> (16*k))     & 0xFFu);
            int8_t b = static_cast<int8_t>((packed >> (16*k + 8)) & 0xFFu);
            float2 f2;
            f2.x = static_cast<float>(a) * s;
            f2.y = static_cast<float>(b) * s;
            pk.h2[k] = Traits::from_float2(f2);
        }
        *reinterpret_cast<int4*>(y + base) = pk.raw;
    } else if (base < row_end) {
        int64_t end = (base + lanes < row_end) ? (base + lanes) : row_end;
        for (int64_t k = base; k < end; ++k) {
            y[k] = Traits::from_float(static_cast<float>(q[k]) * s);
        }
    }
}

// =============================================================================
// Per-group. q viewed as (num_groups, G) contiguous, scale shape (num_groups,).
// Scalar load + stride loop; one block per group.
// =============================================================================

constexpr int GROUP_BLOCK = 128;

template <typename T>
__global__ void dequantize_group_int8(
    const int8_t* __restrict__ q,
    const float* __restrict__ scale,
    T* __restrict__ y,
    int group_size)
{
    using Traits = DTypeTraits<T>;

    const int64_t g = blockIdx.x;
    const int64_t base = g * (int64_t)group_size;
    const float s = __ldg(scale + g);

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        y[base + i] = Traits::from_float(static_cast<float>(q[base + i]) * s);
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================

#define INSTANTIATE_FOR(T) \
    template __global__ void dequantize_tensor_int8<T>(const int8_t*, const float*, T*, int64_t); \
    template __global__ void dequantize_channel_int8<T>(const int8_t*, const float*, T*, int64_t); \
    template __global__ void dequantize_group_int8<T>(const int8_t*, const float*, T*, int);

INSTANTIATE_FOR(half)
INSTANTIATE_FOR(__nv_bfloat16)

#undef INSTANTIATE_FOR

// =============================================================================
// Launchers
// =============================================================================

template <typename T>
static void launch_dequantize_tensor(at::Tensor& q, at::Tensor& scale,
                                     at::Tensor& y, int64_t n) {
    const int grid = fp8_grid(n);  // both fp8 and int8 dequant use VEC=8 (lanes).
    dequantize_tensor_int8<T><<<grid, BLOCK>>>(
        q.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<T*>(y.data_ptr()),
        n);
}

template <typename T>
static void launch_dequantize_channel(at::Tensor& q, at::Tensor& scale,
                                      at::Tensor& y, int64_t C, int64_t F) {
    constexpr int lanes = 8;
    const int blocks_per_row = static_cast<int>(
        (F + (int64_t)BLOCK * lanes - 1) / ((int64_t)BLOCK * lanes));
    const dim3 grid(C, blocks_per_row);
    dequantize_channel_int8<T><<<grid, BLOCK>>>(
        q.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<T*>(y.data_ptr()),
        F);
}

template <typename T>
static void launch_dequantize_group(at::Tensor& q, at::Tensor& scale,
                                    at::Tensor& y, int64_t num_groups, int group_size) {
    dequantize_group_int8<T><<<num_groups, GROUP_BLOCK>>>(
        q.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        reinterpret_cast<T*>(y.data_ptr()),
        group_size);
}

// =============================================================================
// Public entry points
// =============================================================================

static void check_q_scale(const at::Tensor& q, const at::Tensor& scale,
                          c10::ScalarType out_dtype) {
    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(q.scalar_type() == at::kChar, "q must be torch.int8");
    TORCH_CHECK(scale.is_cuda(), "scale must be on CUDA");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be torch.float32");
    TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kBFloat16,
                "out_dtype must be torch.float16 or torch.bfloat16");
}

#define DISPATCH_OUT(out_dtype, FN) \
    do { \
        if ((out_dtype) == at::kHalf) FN(half); \
        else                          FN(__nv_bfloat16); \
    } while (0)

at::Tensor
dequantize_per_tensor_int8_cuda(at::Tensor q, at::Tensor scale,
                                c10::ScalarType out_dtype) {
    check_q_scale(q, scale, out_dtype);
    TORCH_CHECK(scale.numel() == 1, "scale must be a scalar for per-tensor mode");

    int64_t n = q.numel();
    auto y = torch::empty(q.sizes(), q.options().dtype(out_dtype));

    #define DO_LAUNCH(T) launch_dequantize_tensor<T>(q, scale, y, n)
    DISPATCH_OUT(out_dtype, DO_LAUNCH);
    #undef DO_LAUNCH

    return y;
}

at::Tensor
dequantize_per_channel_int8_cuda(at::Tensor q, at::Tensor scale,
                                 c10::ScalarType out_dtype) {
    check_q_scale(q, scale, out_dtype);
    TORCH_CHECK(q.dim() >= 1, "q must have at least 1 dim");
    const int64_t C = q.size(0);
    const int64_t F = q.numel() / C;
    TORCH_CHECK(scale.numel() == C, "scale must have C elements (size(0) of q)");
    // 16-byte alignment for the int4 store path requires F % 8 == 0 for
    // half/bf16 output. Per-channel weight tensors typically satisfy this.
    TORCH_CHECK(F % 8 == 0,
                "per-channel dequant requires inner stride F to be a multiple of 8; "
                "got F=", F);

    auto y = torch::empty(q.sizes(), q.options().dtype(out_dtype));

    #define DO_LAUNCH(T) launch_dequantize_channel<T>(q, scale, y, C, F)
    DISPATCH_OUT(out_dtype, DO_LAUNCH);
    #undef DO_LAUNCH

    return y;
}

at::Tensor
dequantize_per_group_int8_cuda(at::Tensor q_grouped, at::Tensor scale,
                               int64_t group_size, c10::ScalarType out_dtype) {
    check_q_scale(q_grouped, scale, out_dtype);
    TORCH_CHECK(q_grouped.dim() == 2,
                "q_grouped must be 2D (num_groups, group_size)");
    TORCH_CHECK(q_grouped.size(1) == group_size,
                "q_grouped.size(1) must equal group_size");
    TORCH_CHECK(group_size > 0 && group_size <= 4096,
                "group_size must be in (0, 4096]");

    const int64_t num_groups = q_grouped.size(0);
    TORCH_CHECK(scale.numel() == num_groups,
                "scale must have num_groups elements");

    auto y = torch::empty(q_grouped.sizes(), q_grouped.options().dtype(out_dtype));

    #define DO_LAUNCH(T) \
        launch_dequantize_group<T>(q_grouped, scale, y, num_groups, \
                                   static_cast<int>(group_size))
    DISPATCH_OUT(out_dtype, DO_LAUNCH);
    #undef DO_LAUNCH

    return y;
}

#undef DISPATCH_OUT

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_per_tensor_int8", &dequantize_per_tensor_int8_cuda,
          "Per-tensor symmetric INT8 dequantization (CUDA)",
          py::arg("q"), py::arg("scale"), py::arg("out_dtype"));
    m.def("dequantize_per_channel_int8", &dequantize_per_channel_int8_cuda,
          "Per-channel symmetric INT8 dequantization (CUDA)",
          py::arg("q"), py::arg("scale"), py::arg("out_dtype"));
    m.def("dequantize_per_group_int8", &dequantize_per_group_int8_cuda,
          "Per-group symmetric INT8 dequantization (CUDA)",
          py::arg("q_grouped"), py::arg("scale"), py::arg("group_size"), py::arg("out_dtype"));
}
