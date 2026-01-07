#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <torch/extension.h>

using namespace nvcuda;

// Type-specific zero value (needed because __CUDA_NO_HALF_CONVERSIONS__ is defined)
template <typename T> __device__ __forceinline__ T zero_val();
template <> __device__ __forceinline__ float zero_val<float>() { return 0.0f; }
template <> __device__ __forceinline__ half zero_val<half>() { return __float2half(0.0f); }
template <> __device__ __forceinline__ int zero_val<int>() { return 0; }

// Vectorized loading parameters (compile-time)
// int4 = 16 bytes: holds 8 halfs/bf16 or 16 int8s
template <typename T>
struct VecTraits {
    static constexpr int elems_per_vec = sizeof(int4) / sizeof(T);
    static constexpr int a_vecs_per_row = 16 / elems_per_vec;  // 16 cols per k-tile
    static constexpr int b_vecs_per_row = 128 / elems_per_vec; // 128 cols per block
    static constexpr int a_total_vecs = 128 * a_vecs_per_row;
    static constexpr int b_total_vecs = 16 * b_vecs_per_row;
};

// Template kernel for WMMA GEMM with async pipeline
// T: input type (half, __nv_bfloat16, signed char, unsigned char)
// AccT: accumulator type (float, half, int)
template <typename T, typename AccT>
__global__ void wmma_gemm_async(const T* A, const T* B, AccT* C, int M, int N, int K) {
    __shared__ T As[4096];
    __shared__ T Bs[4096];

    constexpr int elems_per_vec = VecTraits<T>::elems_per_vec;
    constexpr int a_vecs_per_row = VecTraits<T>::a_vecs_per_row;
    constexpr int b_vecs_per_row = VecTraits<T>::b_vecs_per_row;
    constexpr int a_total_vecs = VecTraits<T>::a_total_vecs;
    constexpr int b_total_vecs = VecTraits<T>::b_total_vecs;

    int warpId = threadIdx.x / 32;
    int warp_row = warpId / 2;
    int warp_col = warpId % 2;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, AccT> acc_frag[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(acc_frag[i][j], zero_val<AccT>());
        }
    }

    int lda_vec = K / elems_per_vec;
    int ldb_vec = N / elems_per_vec;
    int compute_stage = 0;

    // Initial load for stage 0
    {
        T* dst_base = As;
        const T* src_base = A + (blockIdx.y * 128 * K);

        int4* dst = reinterpret_cast<int4*>(dst_base);
        const int4* src = reinterpret_cast<const int4*>(src_base);

        for (int i = threadIdx.x; i < a_total_vecs; i += blockDim.x) {
            int row = i / a_vecs_per_row;
            int col = i % a_vecs_per_row;
            __pipeline_memcpy_async(&dst[i], &src[row * lda_vec + col], sizeof(int4));
        }
    }

    {
        T* dst_base = Bs;
        const T* src_base = B + (blockIdx.x * 128);

        int4* dst = reinterpret_cast<int4*>(dst_base);
        const int4* src = reinterpret_cast<const int4*>(src_base);

        for (int i = threadIdx.x; i < b_total_vecs; i += blockDim.x) {
            int row = i / b_vecs_per_row;
            int col = i % b_vecs_per_row;
            __pipeline_memcpy_async(&dst[i], &src[row * ldb_vec + col], sizeof(int4));
        }
    }

    __pipeline_commit();

    for (int k_step = 0; k_step < K; k_step += 16) {
        __pipeline_wait_prior(0);
        __syncthreads();

        T* A_warp = As + (compute_stage * 2048) + (warp_row * 64 * 16);
        T* B_warp = Bs + (compute_stage * 2048) + (warp_col * 64);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            wmma::load_matrix_sync(a_frag[i], A_warp + i * 16 * 16, 16);
        }

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::load_matrix_sync(b_frag[j], B_warp + j * 16, 128);
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
            }
        }

        if (k_step + 16 < K) {
            int next_stage = compute_stage ^ 1;

            {
                T* dst_base = As + (next_stage * 2048);
                const T* src_base = A + (blockIdx.y * 128 * K) + (k_step + 16);

                int4* dst = reinterpret_cast<int4*>(dst_base);
                const int4* src = reinterpret_cast<const int4*>(src_base);

                for (int i = threadIdx.x; i < a_total_vecs; i += blockDim.x) {
                    int row = i / a_vecs_per_row;
                    int col = i % a_vecs_per_row;
                    __pipeline_memcpy_async(&dst[i], &src[row * lda_vec + col], sizeof(int4));
                }
            }

            {
                T* dst_base = Bs + (next_stage * 2048);
                const T* src_base = B + ((k_step + 16) * N) + (blockIdx.x * 128);

                int4* dst = reinterpret_cast<int4*>(dst_base);
                const int4* src = reinterpret_cast<const int4*>(src_base);

                for (int i = threadIdx.x; i < b_total_vecs; i += blockDim.x) {
                    int row = i / b_vecs_per_row;
                    int col = i % b_vecs_per_row;
                    __pipeline_memcpy_async(&dst[i], &src[row * ldb_vec + col], sizeof(int4));
                }
            }

            __pipeline_commit();
        }

        compute_stage ^= 1;
    }

    int C_row_base = (blockIdx.y * 128) + (warp_row * 64);
    int C_col_base = (blockIdx.x * 128) + (warp_col * 64);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = C_row_base + i * 16;
            int col = C_col_base + j * 16;

            if (row < M && col < N) {
                AccT* dst = C + row * N + col;
                wmma::store_matrix_sync(dst, acc_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

// Explicit instantiations for all supported precision combinations (16x16x16)
// float16 -> float32/float16
template __global__ void wmma_gemm_async<half, float>(const half*, const half*, float*, int, int, int);
template __global__ void wmma_gemm_async<half, half>(const half*, const half*, half*, int, int, int);
// bfloat16 -> float32
template __global__ void wmma_gemm_async<__nv_bfloat16, float>(const __nv_bfloat16*, const __nv_bfloat16*, float*, int, int, int);
// int8 -> int32
template __global__ void wmma_gemm_async<signed char, int>(const signed char*, const signed char*, int*, int, int, int);
template __global__ void wmma_gemm_async<unsigned char, int>(const unsigned char*, const unsigned char*, int*, int, int, int);

// Launch configuration: 128 threads (4 warps), each block handles 128x128 output tile
constexpr int BLOCK_SIZE = 128;
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;

torch::Tensor wmma_gemm(const torch::Tensor& A, const torch::Tensor& B,
                        c10::optional<torch::ScalarType> acc_dtype) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D matrices");
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions incompatible for GEMM");
    TORCH_CHECK(A.dtype() == B.dtype(), "A and B must have the same dtype");

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    // WMMA 16x16x16 tile constraints
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                "Dimensions must be multiples of 16 for WMMA");

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(BLOCK_SIZE);

    auto input_dtype = A.scalar_type();

    if (input_dtype == torch::kFloat16) {
        auto out_dtype = acc_dtype.value_or(torch::kFloat32);
        auto C = torch::empty({M, N}, A.options().dtype(out_dtype));

        if (out_dtype == torch::kFloat32) {
            wmma_gemm_async<half, float><<<grid, block>>>(
                reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
                C.data_ptr<float>(), M, N, K);
        } else if (out_dtype == torch::kFloat16) {
            wmma_gemm_async<half, half><<<grid, block>>>(
                reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
                reinterpret_cast<half*>(C.data_ptr<at::Half>()), M, N, K);
        } else {
            TORCH_CHECK(false, "float16 inputs support float32 or float16 accumulator");
        }
        return C;

    } else if (input_dtype == torch::kBFloat16) {
        // bfloat16 only supports float32 accumulator
        TORCH_CHECK(!acc_dtype.has_value() || acc_dtype.value() == torch::kFloat32,
                    "bfloat16 inputs only support float32 accumulator");
        auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
        wmma_gemm_async<__nv_bfloat16, float><<<grid, block>>>(
            reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
            C.data_ptr<float>(), M, N, K);
        return C;

    } else if (input_dtype == torch::kInt8) {
        // signed int8 -> int32 accumulator
        TORCH_CHECK(!acc_dtype.has_value() || acc_dtype.value() == torch::kInt32,
                    "int8 inputs only support int32 accumulator");
        auto C = torch::empty({M, N}, A.options().dtype(torch::kInt32));
        wmma_gemm_async<signed char, int><<<grid, block>>>(
            A.data_ptr<int8_t>(),
            B.data_ptr<int8_t>(),
            C.data_ptr<int32_t>(), M, N, K);
        return C;

    } else if (input_dtype == torch::kUInt8) {
        // unsigned int8 -> int32 accumulator
        TORCH_CHECK(!acc_dtype.has_value() || acc_dtype.value() == torch::kInt32,
                    "uint8 inputs only support int32 accumulator");
        auto C = torch::empty({M, N}, A.options().dtype(torch::kInt32));
        wmma_gemm_async<unsigned char, int><<<grid, block>>>(
            A.data_ptr<uint8_t>(),
            B.data_ptr<uint8_t>(),
            C.data_ptr<int32_t>(), M, N, K);
        return C;

    } else {
        TORCH_CHECK(false, "Unsupported dtype. Expected: float16, bfloat16, int8, or uint8");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wmma_gemm", &wmma_gemm, "WMMA GEMM with multi-precision support",
          py::arg("A"), py::arg("B"), py::arg("acc_dtype") = py::none());
}
