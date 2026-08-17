#pragma once
// rc_gemm_i8.cuh -- int8 x int8 -> int32 tensor-core GEMM, the one shape the RC solvers need.
//
// WHY THIS EXISTS (2026-08-05): linking cuBLASLt statically costs ~530 MB of the shipped
// binary -- it carries 2,785 precompiled kernels across 11 architectures plus ~293 MB of
// dispatch tables and CASK descriptors so it can serve every dtype/epilogue/layout combo.
// The miner asks it for exactly ONE of those: row-major C(m,n) = A(m,k).B(k,n), int8 in,
// int32 accumulate, alpha=1, beta in {0,1}. (gemm8_nt/gemm8_tn in rc_gpu_episode.cu have no
// callers -- see the MEASURED DEAD note there, 2026-07-20.) This file is that one kernel;
// dropping the cuBLASLt link takes the binary from ~598 MB to ~80 MB.
//
// BYTE-EXACTNESS IS FREE HERE, and that is the whole reason this swap is safe. int8*int8
// accumulated in int32 is EXACT integer arithmetic, and integer addition is associative --
// so any correct GEMM produces bit-identical results regardless of tile shape, k-order,
// split-k, or reduction scheme. There is no FP reassociation hazard. The caller's
// GuardInt32Bound() already proves the accumulator cannot overflow (worst case
// 2304 * contraction < 2^31). Zero-padding partial tiles is likewise exact: a zero operand
// contributes nothing to the sum.
//
// LAYOUT, and the reason we can beat cuBLASLt's own kernel choice here. INT8 IMMA wants BOTH
// operands k-contiguous ("TN"). Ours are not: A(m,k) row-major is k-contiguous, but B(k,n)
// row-major is n-contiguous. cuBLASLt resolves that mismatch with an operand transform and
// lands on a worse kernel -- which is what the 2026-07-20 note in rc_gpu_episode.cu measured
// from the other direction (folding the transpose into TRANSA/TRANSB cost 10%, 1.998 ->
// 1.804 ep/s). We sidestep it entirely: B is transposed ONCE on the way into shared memory,
// per BK-tile, so the inner loop reads both operands k-contiguous and no global transform
// pass is ever materialised.
//
// TILE CONFIG. Tile shape must be a compile-time template parameter -- it determines register
// allocation, shared-memory layout and unrolling -- which is precisely why cuBLASLt ships
// thousands of cubins rather than one. M/N/K/strides/beta are runtime arguments, so a single
// instantiation serves ANY shape, including a consensus change to matmul_n.

#include <cuda_runtime.h>
#include <cstdint>

namespace rcgemm {

using i8  = int8_t;
using i32 = int32_t;
using u32 = uint32_t;

// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 -- sm_80+ (all of MATADOR_CUDA_ARCH).
// Thread fragment layout, lane 0..31, groupID = lane>>2, tig = lane&3:
//   A(16x32): reg r holds 4 int8 at row = groupID + (r&1 ? 8 : 0),
//                                  col = tig*4 + (r&2 ? 16 : 0) + 0..3
//   B(32x8) : reg r holds 4 int8 at col = groupID,
//                                  row = tig*4 + (r ? 16 : 0) + 0..3
//   C(16x8) : reg r at row = groupID + (r&2 ? 8 : 0), col = tig*2 + (r&1)
__device__ __forceinline__ void mma_m16n8k32(i32 (&d)[4], const i32 (&a)[4],
                                             const i32 (&b)[2], const i32 (&c)[4])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#else
    (void)a; (void)b;
    d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

// BM x BN output tile per block, BK-deep steps. 256 threads = 8 warps in a 2x4 grid, so each
// warp owns a 64x32 slab = 4x4 mma tiles = 64 int32 accumulators per thread.
//
// SMEM PADDING (SMEM_PAD=4) is load-bearing, not cosmetic. The B transpose writes 16
// consecutive n to 16 different smem rows; at an unpadded 64-byte stride those all land in
// the same bank group. Padding to 68 bytes makes the row stride 17 words, and 17 is coprime
// with 32, so the 16 writes hit 16 distinct banks.
template <int BM, int BN, int BK>
__global__ __launch_bounds__(256) void k_gemm_i8_nn(
    const i8* __restrict__ A, const i8* __restrict__ B, i32* __restrict__ C,
    int m, int n, int k, int lda, int ldb, int ldc, i32 beta)
{
    constexpr int SMEM_PAD = 4;
    constexpr int AS_LD    = BK + SMEM_PAD;
    constexpr int BS_LD    = BK + SMEM_PAD;
    constexpr int THREADS  = 256;
    constexpr int WARPS_N  = 4;              // 2 warps in M, 4 in N
    constexpr int WM       = BM / 2;         // 64
    constexpr int WN       = BN / WARPS_N;   // 32
    constexpr int MT       = WM / 16;        // 4 m-fragments per warp
    constexpr int NT       = WN / 8;         // 4 n-fragments per warp

    __shared__ i8 As[BM * AS_LD];
    __shared__ i8 Bs[BN * BS_LD];

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int grp    = lane >> 2;            // groupID 0..7
    const int tig    = lane & 3;             // thread-in-group 0..3
    const int warp_m = warp >> 2;            // 0..1
    const int warp_n = warp & 3;             // 0..3

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;

    i32 acc[MT][NT][4] = {};

    for (int k0 = 0; k0 < k; k0 += BK) {
        // A tile -> As[BM][BK]. Consecutive tid walk consecutive k, so global reads coalesce.
        for (int idx = tid; idx < BM * BK; idx += THREADS) {
            const int r = idx / BK, c = idx % BK;
            const int gr = m0 + r, gc = k0 + c;
            As[r * AS_LD + c] =
                (gr < m && gc < k) ? A[(size_t)gr * lda + gc] : (i8)0;
        }
        // B tile -> Bs[BN][BK], transposed on the way in. Read coalesced along n; the
        // scattered smem write is what SMEM_PAD defuses.
        for (int idx = tid; idx < BK * BN; idx += THREADS) {
            const int kk = idx / BN, nn = idx % BN;
            const int gk = k0 + kk, gn = n0 + nn;
            Bs[nn * BS_LD + kk] =
                (gk < k && gn < n) ? B[(size_t)gk * ldb + gn] : (i8)0;
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < BK; ks += 32) {
            i32 af[MT][4], bf[NT][2];
            #pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int row = warp_m * WM + i * 16 + grp;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int rr = row + ((r & 1) ? 8 : 0);
                    const int cc = ks + tig * 4 + ((r & 2) ? 16 : 0);
                    af[i][r] = *reinterpret_cast<const i32*>(&As[rr * AS_LD + cc]);
                }
            }
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                const int col = warp_n * WN + j * 8 + grp;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int kk = ks + tig * 4 + (r ? 16 : 0);
                    bf[j][r] = *reinterpret_cast<const i32*>(&Bs[col * BS_LD + kk]);
                }
            }
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_m16n8k32(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue. beta=1 means D = A.B + C with C preloaded (the fused residual down-proj).
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int row = m0 + warp_m * WM + i * 16 + grp + ((r & 2) ? 8 : 0);
                const int col = n0 + warp_n * WN + j * 8 + tig * 2 + (r & 1);
                if (row < m && col < n) {
                    i32* p = &C[(size_t)row * ldc + col];
                    *p = beta ? (*p + acc[i][j][r]) : acc[i][j][r];
                }
            }
        }
    }
}

// row-major C(m,n) = A(m,k).B(k,n) [+ C when beta], int8 -> int32.
// Shapes are runtime arguments: one instantiation serves any m/n/k, so a consensus change to
// matmul_n needs no rebuild.
inline void gemm_i8_nn(const i8* A, const i8* B, i32* C,
                       int m, int n, int k, int lda, int ldb, int ldc,
                       i32 beta, cudaStream_t stream)
{
    constexpr int BM = 128, BN = 128, BK = 64;
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    k_gemm_i8_nn<BM, BN, BK><<<grid, 256, 0, stream>>>(
        A, B, C, m, n, k, lda, ldb, ldc, beta);
}

}  // namespace rcgemm
