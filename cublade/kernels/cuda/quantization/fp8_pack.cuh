// Shared utilities for the per-tensor FP8 quant/dequant kernels.
//
// Each thread owns VEC=8 elements via one int4 (128-bit) load. Both half and
// __nv_bfloat16 are 2 bytes, so the same int4 packs 8 of either. The Pack16<T>
// union aliases the same 16 bytes as int4 / 4x vec2 / 8x T so we can load via
// int4 and access as vec2 without an explicit reinterpret_cast.
//
// DTypeTraits<T> specialises the dtype-specific conversion primitives so the
// kernels themselves stay one template parameter wide.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace cublade {
namespace quantization {

constexpr int BLOCK = 256;
constexpr int VEC   = 8;

template <typename T>
struct DTypeTraits;

template <>
struct DTypeTraits<half> {
    using vec2 = __half2;

    __device__ __forceinline__ static float2 to_float2(vec2 v) {
        return __half22float2(v);
    }
    __device__ __forceinline__ static vec2 from_float2(float2 f) {
        return __float22half2_rn(f);
    }
    // FP8x2 -> half2 via the direct hardware intrinsic (E4M3 fits losslessly
    // in half - the conversion is exact).
    __device__ __forceinline__ static vec2 fp8x2_to_vec2(uint16_t pair) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
        return half2(hr);
    }
    __device__ __forceinline__ static float to_float(half v) {
        return __half2float(v);
    }
    __device__ __forceinline__ static half from_float(float v) {
        return __float2half_rn(v);
    }
};

template <>
struct DTypeTraits<__nv_bfloat16> {
    using vec2 = __nv_bfloat162;

    __device__ __forceinline__ static float2 to_float2(vec2 v) {
        return __bfloat1622float2(v);
    }
    __device__ __forceinline__ static vec2 from_float2(float2 f) {
        return __float22bfloat162_rn(f);
    }
    // FP8 -> bf16 has no direct intrinsic on SM120. Go via halfraw2 (the
    // FP8->half conversion is exact), then narrow to bf16 through float. The
    // extra hop costs a few SFU instructions per pack; with the kernel at
    // 85-92% DRAM busy that's noise.
    __device__ __forceinline__ static vec2 fp8x2_to_vec2(uint16_t pair) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3);
        float2 f = __half22float2(half2(hr));
        return __float22bfloat162_rn(f);
    }
    __device__ __forceinline__ static float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    __device__ __forceinline__ static __nv_bfloat16 from_float(float v) {
        return __float2bfloat16_rn(v);
    }
};

template <typename T>
union Pack16 {
    int4                              raw;
    typename DTypeTraits<T>::vec2     h2[4];
    T                                 h[8];
};

// Grid math: each thread owns VEC elements.
__host__ __device__ __forceinline__ int fp8_grid(int64_t n) {
    const int64_t work = (int64_t)BLOCK * VEC;
    return static_cast<int>((n + work - 1) / work);
}

}  // namespace quantization
}  // namespace cublade
