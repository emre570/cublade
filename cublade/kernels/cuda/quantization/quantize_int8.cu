// Symmetric INT8 quantization kernels (per-tensor, per-channel, per-group).
// Each mode is two-pass:
//   amax_reduce : find |x|_max over the reduction domain (tensor / row / group).
//   quantize_cast: scale = 127 / max(amax, eps); writes
//                  q = round_half_even_clamp_int8(x * scale)
//                  scale_out = amax / 127 (this is what dequant multiplies by).
//
// Input dtype T in {half, __nv_bfloat16, float}. Half/bf16 use 16-byte int4
// loads (8 lanes/thread); fp32 uses int4 loads of 4 floats (4 lanes/thread).
// Group kernels use scalar loads + a stride loop because group sizes (16-256)
// don't benefit meaningfully from int4 vectorization.

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#include "int8_pack.cuh"

using cublade::quantization::BLOCK;
using cublade::quantization::INT8_QMAX;
using cublade::quantization::DTypeTraits;
using cublade::quantization::PackInT;
using cublade::quantization::LanesPerInt4;
using cublade::quantization::round_half_even_clamp_int8;
using cublade::quantization::int8_grid;

// fp32 atomic max via int CAS. -0.0 and +0.0 fold to the same int bits so a
// positive amax always wins against a freshly-zeroed buffer.
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

// =============================================================================
// Per-tensor
// =============================================================================

template <typename T>
__global__ void amax_reduce_tensor_int8(
    const T* __restrict__ x,
    float* __restrict__ amax_buf,
    int64_t n)
{
    using Traits = DTypeTraits<T>;
    constexpr int lanes = LanesPerInt4<T>::value;

    const int64_t base = (int64_t)blockIdx.x * blockDim.x * lanes
                       + (int64_t)threadIdx.x * lanes;
    float my_val = 0.0f;

    if (base + lanes <= n) {
        PackInT<T> pk;
        pk.raw = __ldg(reinterpret_cast<const int4*>(x + base));
        if constexpr (std::is_same_v<T, float>) {
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                my_val = fmaxf(my_val, fabsf(pk.f[k]));
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = Traits::to_float2(pk.h2[k]);
                my_val = fmaxf(my_val, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    } else if (base < n) {
        int64_t end = (base + lanes < n) ? (base + lanes) : n;
        for (int64_t k = base; k < end; ++k)
            my_val = fmaxf(my_val, fabsf(Traits::to_float(x[k])));
    }

    // Warp reduce.
    float v = 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = __shfl_xor_sync(0xffffffff, my_val, offset);
        my_val = fmaxf(my_val, v);
    }

    __shared__ float warp_max[8];  // BLOCK=256 -> 8 warps.
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
__global__ void quantize_cast_tensor_int8(
    const T* __restrict__ x,
    const float* __restrict__ amax_buf,
    int8_t* __restrict__ q,
    float* __restrict__ scale_out,
    int64_t n,
    float eps)
{
    using Traits = DTypeTraits<T>;
    constexpr int lanes = LanesPerInt4<T>::value;

    // Route division through fp64 (hardware on Blackwell SM120) so the
    // narrowed-to-fp32 result is the same IEEE-rounded value PyTorch's CPU
    // `/` produces. Single-precision division on SM120 is emulated via
    // FRCP + Newton-Raphson and drifts ~1 ULP, which flips round-half-to-even
    // ties versus the torch reference for elements at integer half points.
    const float amax_clamped = fmaxf(*amax_buf, eps);
    const float scale = static_cast<float>(
        static_cast<double>(INT8_QMAX) / static_cast<double>(amax_clamped));

    if (blockIdx.x == 0 && threadIdx.x == 0)
        *scale_out = static_cast<float>(
            1.0 / static_cast<double>(scale));

    const int64_t base = (int64_t)blockIdx.x * blockDim.x * lanes
                       + (int64_t)threadIdx.x * lanes;

    if (base + lanes <= n) {
        PackInT<T> pk;
        pk.raw = __ldg(reinterpret_cast<const int4*>(x + base));

        if constexpr (std::is_same_v<T, float>) {
            int8_t out[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                out[k] = round_half_even_clamp_int8(pk.f[k] * scale);
            uint32_t packed = ((uint32_t)(uint8_t)out[0])
                            | ((uint32_t)(uint8_t)out[1] << 8)
                            | ((uint32_t)(uint8_t)out[2] << 16)
                            | ((uint32_t)(uint8_t)out[3] << 24);
            *reinterpret_cast<uint32_t*>(q + base) = packed;
        } else {
            int8_t out[8];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f2 = Traits::to_float2(pk.h2[k]);
                out[2*k]     = round_half_even_clamp_int8(f2.x * scale);
                out[2*k + 1] = round_half_even_clamp_int8(f2.y * scale);
            }
            uint64_t packed = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                packed |= ((uint64_t)(uint8_t)out[k]) << (8 * k);
            *reinterpret_cast<uint64_t*>(q + base) = packed;
        }
    } else if (base < n) {
        int64_t end = (base + lanes < n) ? (base + lanes) : n;
        for (int64_t k = base; k < end; ++k) {
            float v = Traits::to_float(x[k]) * scale;
            q[k] = round_half_even_clamp_int8(v);
        }
    }
}

// =============================================================================
// Per-channel (ch_axis=0). Input viewed as (C, F) row-major contiguous.
// amax_buf is shape (C,), zero-initialised by the caller.
// =============================================================================

template <typename T>
__global__ void amax_reduce_channel_int8(
    const T* __restrict__ x,
    float* __restrict__ amax_buf,
    int64_t F)
{
    using Traits = DTypeTraits<T>;
    constexpr int lanes = LanesPerInt4<T>::value;

    const int64_t c = blockIdx.x;
    const int64_t row_base = c * F;
    const int64_t base = row_base
                       + (int64_t)blockIdx.y * blockDim.x * lanes
                       + (int64_t)threadIdx.x * lanes;
    const int64_t row_end = row_base + F;

    float my_val = 0.0f;
    if (base + lanes <= row_end) {
        PackInT<T> pk;
        pk.raw = __ldg(reinterpret_cast<const int4*>(x + base));
        if constexpr (std::is_same_v<T, float>) {
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                my_val = fmaxf(my_val, fabsf(pk.f[k]));
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = Traits::to_float2(pk.h2[k]);
                my_val = fmaxf(my_val, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
    } else if (base < row_end) {
        int64_t end = (base + lanes < row_end) ? (base + lanes) : row_end;
        for (int64_t k = base; k < end; ++k)
            my_val = fmaxf(my_val, fabsf(Traits::to_float(x[k])));
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
        if (lane == 0) atomicMaxFloat(amax_buf + c, my_val);
    }
}

template <typename T>
__global__ void quantize_cast_channel_int8(
    const T* __restrict__ x,
    const float* __restrict__ amax_buf,
    int8_t* __restrict__ q,
    float* __restrict__ scale_out,
    int64_t F,
    float eps)
{
    using Traits = DTypeTraits<T>;
    constexpr int lanes = LanesPerInt4<T>::value;

    const int64_t c = blockIdx.x;
    const float amax = __ldg(amax_buf + c);
    // fp64-divide-then-narrow gives bit-exact parity with CPU IEEE division.
    // See the comment in quantize_cast_tensor_int8.
    const float amax_clamped = fmaxf(amax, eps);
    const float scale = static_cast<float>(
        static_cast<double>(INT8_QMAX) / static_cast<double>(amax_clamped));

    if (blockIdx.y == 0 && threadIdx.x == 0)
        scale_out[c] = static_cast<float>(
            1.0 / static_cast<double>(scale));

    const int64_t row_base = c * F;
    const int64_t base = row_base
                       + (int64_t)blockIdx.y * blockDim.x * lanes
                       + (int64_t)threadIdx.x * lanes;
    const int64_t row_end = row_base + F;

    if (base + lanes <= row_end) {
        PackInT<T> pk;
        pk.raw = __ldg(reinterpret_cast<const int4*>(x + base));

        if constexpr (std::is_same_v<T, float>) {
            int8_t out[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                out[k] = round_half_even_clamp_int8(pk.f[k] * scale);
            uint32_t packed = ((uint32_t)(uint8_t)out[0])
                            | ((uint32_t)(uint8_t)out[1] << 8)
                            | ((uint32_t)(uint8_t)out[2] << 16)
                            | ((uint32_t)(uint8_t)out[3] << 24);
            *reinterpret_cast<uint32_t*>(q + base) = packed;
        } else {
            int8_t out[8];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f2 = Traits::to_float2(pk.h2[k]);
                out[2*k]     = round_half_even_clamp_int8(f2.x * scale);
                out[2*k + 1] = round_half_even_clamp_int8(f2.y * scale);
            }
            uint64_t packed = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                packed |= ((uint64_t)(uint8_t)out[k]) << (8 * k);
            *reinterpret_cast<uint64_t*>(q + base) = packed;
        }
    } else if (base < row_end) {
        int64_t end = (base + lanes < row_end) ? (base + lanes) : row_end;
        for (int64_t k = base; k < end; ++k) {
            float v = Traits::to_float(x[k]) * scale;
            q[k] = round_half_even_clamp_int8(v);
        }
    }
}

// =============================================================================
// Per-group. Input viewed as contiguous (num_groups, G). One block per group.
// Scalar loads + stride loop - group sizes 16-256 are too small for int4
// vectorisation to pay back the indexing complexity.
// =============================================================================

constexpr int GROUP_BLOCK = 128;

template <typename T>
__global__ void amax_reduce_group_int8(
    const T* __restrict__ x,
    float* __restrict__ amax_buf,
    int group_size)
{
    using Traits = DTypeTraits<T>;

    const int64_t g = blockIdx.x;
    const int64_t base = g * (int64_t)group_size;

    float my_val = 0.0f;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        my_val = fmaxf(my_val, fabsf(Traits::to_float(x[base + i])));
    }

    float v = 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = __shfl_xor_sync(0xffffffff, my_val, offset);
        my_val = fmaxf(my_val, v);
    }

    __shared__ float warp_max[GROUP_BLOCK / 32];  // 4 warps.
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) warp_max[warp_id] = my_val;
    __syncthreads();

    if (warp_id == 0) {
        my_val = (lane < GROUP_BLOCK / 32) ? warp_max[lane] : 0.0f;
        for (int offset = (GROUP_BLOCK / 32) / 2; offset > 0; offset >>= 1) {
            v = __shfl_xor_sync(0xffffffff, my_val, offset);
            my_val = fmaxf(my_val, v);
        }
        if (lane == 0) amax_buf[g] = my_val;  // one writer per group, no atomic.
    }
}

template <typename T>
__global__ void quantize_cast_group_int8(
    const T* __restrict__ x,
    const float* __restrict__ amax_buf,
    int8_t* __restrict__ q,
    float* __restrict__ scale_out,
    int group_size,
    float eps)
{
    using Traits = DTypeTraits<T>;

    const int64_t g = blockIdx.x;
    const int64_t base = g * (int64_t)group_size;
    const float amax = __ldg(amax_buf + g);
    // fp64-divide-then-narrow for bit-exact parity. See quantize_cast_tensor_int8.
    const float amax_clamped = fmaxf(amax, eps);
    const float scale = static_cast<float>(
        static_cast<double>(INT8_QMAX) / static_cast<double>(amax_clamped));

    if (threadIdx.x == 0) scale_out[g] = static_cast<float>(
        1.0 / static_cast<double>(scale));

    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = Traits::to_float(x[base + i]) * scale;
        q[base + i] = round_half_even_clamp_int8(v);
    }
}

// =============================================================================
// Explicit instantiations
// =============================================================================

#define INSTANTIATE_FOR(T) \
    template __global__ void amax_reduce_tensor_int8<T>(const T*, float*, int64_t); \
    template __global__ void quantize_cast_tensor_int8<T>(const T*, const float*, int8_t*, float*, int64_t, float); \
    template __global__ void amax_reduce_channel_int8<T>(const T*, float*, int64_t); \
    template __global__ void quantize_cast_channel_int8<T>(const T*, const float*, int8_t*, float*, int64_t, float); \
    template __global__ void amax_reduce_group_int8<T>(const T*, float*, int); \
    template __global__ void quantize_cast_group_int8<T>(const T*, const float*, int8_t*, float*, int, float);

INSTANTIATE_FOR(half)
INSTANTIATE_FOR(__nv_bfloat16)
INSTANTIATE_FOR(float)

#undef INSTANTIATE_FOR

// =============================================================================
// Launchers
// =============================================================================

template <typename T>
static void launch_quantize_tensor(at::Tensor& x, at::Tensor& q,
                                   at::Tensor& amax, at::Tensor& scale,
                                   int64_t n, float eps) {
    const int grid = int8_grid<T>(n);
    amax_reduce_tensor_int8<T><<<grid, BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        n);
    quantize_cast_tensor_int8<T><<<grid, BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        q.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        n, eps);
}

template <typename T>
static void launch_quantize_channel(at::Tensor& x, at::Tensor& q,
                                    at::Tensor& amax, at::Tensor& scale,
                                    int64_t C, int64_t F, float eps) {
    constexpr int lanes = LanesPerInt4<T>::value;
    const int blocks_per_row = static_cast<int>(
        (F + (int64_t)BLOCK * lanes - 1) / ((int64_t)BLOCK * lanes));
    const dim3 grid(C, blocks_per_row);

    amax_reduce_channel_int8<T><<<grid, BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        F);
    quantize_cast_channel_int8<T><<<grid, BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        q.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        F, eps);
}

template <typename T>
static void launch_quantize_group(at::Tensor& x, at::Tensor& q,
                                  at::Tensor& amax, at::Tensor& scale,
                                  int64_t num_groups, int group_size, float eps) {
    amax_reduce_group_int8<T><<<num_groups, GROUP_BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        group_size);
    quantize_cast_group_int8<T><<<num_groups, GROUP_BLOCK>>>(
        reinterpret_cast<T*>(x.data_ptr()),
        amax.data_ptr<float>(),
        q.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        group_size, eps);
}

// =============================================================================
// Public entry points
// =============================================================================

static void check_input(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf
                || x.scalar_type() == at::kBFloat16
                || x.scalar_type() == at::kFloat,
                "x must be torch.float16, torch.bfloat16, or torch.float32");
}

#define DISPATCH_INPUT(x, FN) \
    do { \
        if ((x).scalar_type() == at::kHalf)        FN(half); \
        else if ((x).scalar_type() == at::kBFloat16) FN(__nv_bfloat16); \
        else                                          FN(float); \
    } while (0)

// Returns (q[int8, same shape as x], scale[float32, 0-d]).
std::tuple<at::Tensor, at::Tensor>
quantize_per_tensor_int8_cuda(at::Tensor x, double eps) {
    check_input(x);
    int64_t n = x.numel();

    auto q     = torch::empty(x.sizes(), x.options().dtype(at::kChar));
    auto scale = torch::zeros({}, x.options().dtype(torch::kFloat32));
    auto amax  = torch::zeros({}, x.options().dtype(torch::kFloat32));

    #define DO_LAUNCH(T) launch_quantize_tensor<T>(x, q, amax, scale, n, static_cast<float>(eps))
    DISPATCH_INPUT(x, DO_LAUNCH);
    #undef DO_LAUNCH

    return std::make_tuple(q, scale);
}

// Per-channel quant along axis 0. Returns (q[int8, same shape], scale[float32, (C,)]).
std::tuple<at::Tensor, at::Tensor>
quantize_per_channel_int8_cuda(at::Tensor x, double eps) {
    check_input(x);
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim for per-channel quant");
    const int64_t C = x.size(0);
    const int64_t F = x.numel() / C;
    // 16-byte int4 loads/stores need each row to start on a 16-byte boundary
    // relative to the dtype: F * element_size must be a multiple of 16.
    // For half/bf16 that means F % 8 == 0; for fp32 that means F % 4 == 0.
    TORCH_CHECK(F * x.element_size() % 16 == 0,
                "per-channel quant requires inner stride F * element_size to be a "
                "multiple of 16; got F=", F, " element_size=", x.element_size());

    auto q     = torch::empty(x.sizes(), x.options().dtype(at::kChar));
    auto scale = torch::zeros({C}, x.options().dtype(torch::kFloat32));
    auto amax  = torch::zeros({C}, x.options().dtype(torch::kFloat32));

    #define DO_LAUNCH(T) launch_quantize_channel<T>(x, q, amax, scale, C, F, static_cast<float>(eps))
    DISPATCH_INPUT(x, DO_LAUNCH);
    #undef DO_LAUNCH

    return std::make_tuple(q, scale);
}

// Per-group quant. `x_grouped` must be contiguous (num_groups, group_size).
// Caller does the movedim+view; this kernel just sees a 2D contiguous tensor.
// Returns (q[int8, same shape as x_grouped], scale[float32, (num_groups,)]).
std::tuple<at::Tensor, at::Tensor>
quantize_per_group_int8_cuda(at::Tensor x_grouped, int64_t group_size, double eps) {
    check_input(x_grouped);
    TORCH_CHECK(x_grouped.dim() == 2, "x_grouped must be 2D (num_groups, group_size)");
    TORCH_CHECK(x_grouped.size(1) == group_size,
                "x_grouped.size(1) must equal group_size");
    TORCH_CHECK(group_size > 0 && group_size <= 4096,
                "group_size must be in (0, 4096]");

    const int64_t num_groups = x_grouped.size(0);

    auto q     = torch::empty(x_grouped.sizes(), x_grouped.options().dtype(at::kChar));
    auto scale = torch::zeros({num_groups}, x_grouped.options().dtype(torch::kFloat32));
    auto amax  = torch::zeros({num_groups}, x_grouped.options().dtype(torch::kFloat32));

    #define DO_LAUNCH(T) \
        launch_quantize_group<T>(x_grouped, q, amax, scale, num_groups, \
                                 static_cast<int>(group_size), static_cast<float>(eps))
    DISPATCH_INPUT(x_grouped, DO_LAUNCH);
    #undef DO_LAUNCH

    return std::make_tuple(q, scale);
}

#undef DISPATCH_INPUT

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_per_tensor_int8", &quantize_per_tensor_int8_cuda,
          "Per-tensor symmetric INT8 quantization (CUDA)",
          py::arg("x"), py::arg("eps") = 1e-12);
    m.def("quantize_per_channel_int8", &quantize_per_channel_int8_cuda,
          "Per-channel symmetric INT8 quantization along axis 0 (CUDA)",
          py::arg("x"), py::arg("eps") = 1e-12);
    m.def("quantize_per_group_int8", &quantize_per_group_int8_cuda,
          "Per-group symmetric INT8 quantization, (num_groups, group_size) input (CUDA)",
          py::arg("x_grouped"), py::arg("group_size"), py::arg("eps") = 1e-12);
}
