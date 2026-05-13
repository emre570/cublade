// Shared utilities for the INT8 quant/dequant kernels (per-tensor, per-channel,
// per-group). Layers on top of fp8_pack.cuh so the half/__nv_bfloat16 paths
// reuse DTypeTraits + Pack16 + BLOCK/VEC verbatim. INT8 only adds:
//   - a float specialisation for DTypeTraits (the FP8 header doesn't need fp32
//     inputs; INT8 accepts fp32 because nn.Linear.weight is fp32 by default);
//   - an INT8 storage pack so we can write 16 int8 lanes via one int4 store;
//   - the symmetric-INT8 round/clamp helper. PyTorch's torch.round uses
//     round-half-to-even; __float2int_rn matches that exactly, so the kernel
//     stays bit-exact with the reference at the int8 level.
#pragma once

#include "fp8_pack.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace cublade {
namespace quantization {

// Symmetric INT8: representable range is [-127, 127]. -128 is never produced
// because the scale is set so that amax maps to 127 exactly, keeping the
// quantization symmetric around zero.
constexpr float INT8_QMAX = 127.0f;

// Float specialisation. Reduction kernels only need to_float / fabsf; the cast
// kernel only needs to_float on the input side. The half2 vec2 path is not
// applicable to fp32, so we treat one fp32 as one "lane" - the kernels handle
// that by reading 4 floats (one int4) per thread instead of 8 halves.
template <>
struct DTypeTraits<float> {
    using vec2 = float2;

    __device__ __forceinline__ static float2 to_float2(vec2 v) { return v; }
    __device__ __forceinline__ static vec2 from_float2(float2 f) { return f; }
    __device__ __forceinline__ static float to_float(float v) { return v; }
    __device__ __forceinline__ static float from_float(float v) { return v; }
};

// Pack of 16 int8s aliased with one int4 (16 bytes). Used by per-tensor and
// per-channel cast kernels so 16 lanes write via one 128-bit store. Distinct
// from Pack16<T> in fp8_pack.cuh, which is keyed on the *input* dtype.
union Pack16Int8 {
    int4    raw;
    int8_t  q[16];
};

// Round-half-to-even and saturate to symmetric INT8 range [-127, 127].
// __float2int_rn rounds-to-nearest-even (matches torch.round); we then clamp
// to int8_t range. -128 cannot be produced for symmetric INT8 because the
// caller scales so that |x_max| maps to +/-127.
__device__ __forceinline__ int8_t round_half_even_clamp_int8(float v) {
    int r = __float2int_rn(v);
    if (r >  127) r =  127;
    if (r < -127) r = -127;
    return static_cast<int8_t>(r);
}

// Number of input elements (lanes) handled by one thread when reading via
// one int4. For half/bf16 this is 8 (two bytes/lane). For fp32 this is 4.
template <typename T>
struct LanesPerInt4;

template <> struct LanesPerInt4<half>             { static constexpr int value = 8; };
template <> struct LanesPerInt4<__nv_bfloat16>    { static constexpr int value = 8; };
template <> struct LanesPerInt4<float>            { static constexpr int value = 4; };

// Per-thread input pack templated on input dtype. For half/bf16 it aliases
// the 4-lane half2 vec; for fp32 it aliases 4 floats.
template <typename T>
union PackInT;

template <>
union PackInT<half> {
    int4   raw;
    __half2 h2[4];
    half   h[8];
};

template <>
union PackInT<__nv_bfloat16> {
    int4              raw;
    __nv_bfloat162    h2[4];
    __nv_bfloat16     h[8];
};

template <>
union PackInT<float> {
    int4    raw;
    float   f[4];
};

// Grid math: each thread owns LanesPerInt4<T>::value elements. The "fp8_grid"
// helper in fp8_pack.cuh is hardcoded for VEC=8; mirror it here so the fp32
// path (VEC=4) gets the right block count.
template <typename T>
__host__ __device__ __forceinline__ int int8_grid(int64_t n) {
    constexpr int lanes = LanesPerInt4<T>::value;
    const int64_t work = (int64_t)BLOCK * lanes;
    return static_cast<int>((n + work - 1) / work);
}

}  // namespace quantization
}  // namespace cublade
