#pragma once
// rc_gemm_i8_cutlass.cuh -- the int8 GEMM the RC solvers need, built from CUTLASS.
//
// WHY CUTLASS AND NOT OUR OWN KERNEL. cuBLASLt's int8 tensor-core kernels are generated from
// CUTLASS; NVIDIA publishes the source. So this is the same kernel family cuBLASLt would have
// picked, compiled from the same origin -- except we instantiate a handful of configurations
// instead of linking all 2,785 precompiled variants plus ~293 MB of dispatch tables. Same
// math, same performance class, ~530 MB less binary.
//
// (The hand-written kernel in rc_gemm_i8.cuh is kept as the correctness oracle and as the
// fallback for shapes CUTLASS's 16-byte alignment rejects. It is NOT the fast path -- measured
// 5-6x slower than cuBLASLt.)
//
// THE TILE CONFIG IS THE WHOLE GAME, and it cannot be a runtime flag -- it decides register
// allocation, shared-memory layout and unrolling, so each config is a separate compiled
// kernel. That is precisely why cuBLASLt ships thousands of them and autotunes at runtime.
// We do the same thing at a sane scale: instantiate CFG_COUNT configs, time them once per
// distinct shape, cache the winner. A blind single config left 16-36% on the table against
// cuBLASLt (measured 2026-08-05), which is what motivated this sweep.
//
// LAYOUT. INT8 IMMA requires both operands k-contiguous ("TN"). A(m,k) row-major already is;
// B(k,n) row-major is not. We materialise B^T once per call, which agrees with the 2026-07-20
// finding in rc_gpu_episode.cu that materialising a transpose beats folding it into the
// descriptors.
//
// MEASURED DEAD (2026-08-05): consuming B in its native RowMajor layout, letting the mainloop's
// global->shared iterator absorb the transpose, DOES NOT COMPILE -- CUTLASS has no int8
// TensorOp instantiation for LayoutB=RowMajor (DefaultMmaCore rejects it; the error surfaces as
// an incomplete NumericArrayConverter deep in the warp-level Mma). This is the hardware
// instruction's constraint, not a CUTLASS gap: mma.m16n8k32 reads B from a k-major fragment.
// cuBLASLt has the same problem and solves it the same way, internally -- which is a large part
// of why its NN kernel only reaches ~64% of int8 peak. Do not retry this without a newer
// CUTLASS whose CuTe path advertises a k-major-B int8 collective.
//
// The transpose is therefore mandatory, and it is now DONE as an optimisation target: the
// vectorised kernel moves 134 MB in 0.087 ms = ~1.54 TB/s, which is 5090 memory bandwidth. It
// cannot be made faster, only unnecessary -- and the only route to unnecessary is generating
// the sigma-derived weights pre-transposed, which the operand generator's (rows, cols) block
// structure currently forbids (swapping the dims would change the VALUES, not just the layout).
//
// The transpose cost is INCLUDED in every timing. Do not quote a GEMM-only number against
// cuBLASLt -- that would be comparing a partial operation.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <sys/stat.h>

#include "rc_gemm_i8.cuh"          // the hand kernel: correctness oracle + ragged-shape fallback

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <cutlass/epilogue/thread/linear_combination.h>

namespace rcgemm {

using i8  = int8_t;
using i32 = int32_t;

// Tiled int8 transpose: B(k,n) row-major -> Bt(n,k) row-major (== B column-major, ldb=k).
// 32x33 shared tile: the +1 makes the column stride coprime with the bank count, so the
// strided side is conflict-free. BYTE-GRANULAR on both sides -- measured 0.31 ms for 134 MB
// (~430 GB/s on a card that does ~1.5 TB/s), i.e. bound by access width, not bandwidth.
// Kept as the fallback for shapes the vectorised kernel's 16-byte tiling cannot cover.
template <int TILE = 32>
__global__ void k_transpose_i8_tiled(const i8* __restrict__ src, i8* __restrict__ dst,
                                     int rows, int cols)
{
    __shared__ i8 tile[TILE][TILE + 1];
    const int x = blockIdx.x * TILE + threadIdx.x;
    const int y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = src[(size_t)y * cols + x];
    __syncthreads();
    const int tx = blockIdx.y * TILE + threadIdx.x;
    const int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < rows && ty < cols) dst[(size_t)ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
}

// Vectorised 64x64 transpose: 16-byte loads on the way in, 16-byte stores on the way out, with
// the byte-level shuffle happening in shared memory where it is free. The smem row stride is
// padded to 80 bytes so the int4 stores stay 16-byte aligned (a 68-byte stride would be
// cheaper on banks but misaligns every vector store, which costs far more than it saves).
// Requires rows and cols to be multiples of 64; callers fall back to the byte kernel otherwise.
__global__ __launch_bounds__(256) void k_transpose_i8_vec(const i8* __restrict__ src,
                                                          i8* __restrict__ dst,
                                                          int rows, int cols)
{
    constexpr int T = 64, LD = T + 16;
    __shared__ i8 tile[T * LD];

    const int y0 = blockIdx.y * T;      // along rows (k)
    const int x0 = blockIdx.x * T;      // along cols (n)
    const int tid = threadIdx.x;

    // In: 64 rows x 4 int4 each = 256 vectors, one per thread. Consecutive threads cover
    // consecutive 16-byte chunks of a row, so the global side is fully coalesced.
    {
        const int r = tid >> 2, cgrp = tid & 3;
        const int4 v = *reinterpret_cast<const int4*>(&src[(size_t)(y0 + r) * cols + x0 + cgrp * 16]);
        *reinterpret_cast<int4*>(&tile[r * LD + cgrp * 16]) = v;
    }
    __syncthreads();

    // Out: thread owns one output row (a fixed n) and gathers 16 consecutive k from the tile's
    // strided direction, packing them into one int4 store.
    {
        const int c = tid >> 2, rgrp = tid & 3;
        i8 packed[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) packed[i] = tile[(rgrp * 16 + i) * LD + c];
        *reinterpret_cast<int4*>(&dst[(size_t)(x0 + c) * rows + y0 + rgrp * 16]) =
            *reinterpret_cast<const int4*>(packed);
    }
}

// A: RowMajor (m,k), B: ColumnMajor (k,n) via the transposed buffer -- both k-contiguous.
// C: RowMajor (m,n), int32 accumulate, alpha=1, beta in {0,1}. LinearCombination over int32
// with those coefficients is a plain add: exact, no rounding, no clamp.
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int STAGES, int SWZ = 1>
using GemmCfg = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,                              // IMMA m16n8k32: Ampere through Blackwell
    cutlass::gemm::GemmShape<TBM, TBN, TBK>,
    cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SWZ>,
    STAGES,
    16, 16>;                                          // 128-bit aligned A and B

// g_validate is on while the tuner is probing candidates (it must be able to reject a config
// that cannot run this shape) and off on the steady path, where the shape has already been
// validated and can_implement is pure host overhead -- 136 GEMM calls per episode makes that
// worth removing.
inline bool& gemm_validate() { static bool v = true; return v; }

template <typename Gemm>
static cudaError_t run_cfg(const i8* A, const i8* Bt, i32* C,
                           int m, int n, int k, i32 beta, cudaStream_t stream)
{
    Gemm op;
    typename Gemm::Arguments args({m, n, k}, {A, k}, {Bt, k}, {C, n}, {C, n}, {1, beta});
    if (gemm_validate() && op.can_implement(args) != cutlass::Status::kSuccess)
        return cudaErrorInvalidValue;
    return op(args, nullptr, stream) == cutlass::Status::kSuccess
               ? cudaSuccess : cudaErrorInvalidValue;
}

// ---- split-k -------------------------------------------------------------------------------
// The attention SV GEMM is m=512, n=128, k=786432 at profile-1 dims. Output is 65k elements but
// the contraction is 786k deep, so a 128x256 tile yields ONE threadblock: one SM grinding the
// whole reduction while 169 sit idle. Measured as ~9% of total episode time -- the entire gap
// against cuBLASLt, which picks a narrow split-k kernel here.
//
// SplitKSerial partitions k across slices and accumulates them under a semaphore. Byte-exact:
// the partial sums are int32 and integer addition is associative, so slice count cannot change
// the result -- same argument as everything else in this file.
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int STAGES>
using GemmCfgSplitK = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TBM, TBN, TBK>,
    cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    STAGES,
    16, 16,
    true>;                                        // SplitKSerial

// Serial split-k needs a semaphore workspace. It is small (one int per output tile) and the
// shape set is tiny and recurring, so one lazily-grown buffer covers every call.
inline void* splitk_workspace(size_t need)
{
    static void*  buf = nullptr;
    static size_t cap = 0;
    if (need > cap) {
        if (buf) cudaFree(buf);
        if (cudaMalloc(&buf, need) != cudaSuccess) { buf = nullptr; cap = 0; return nullptr; }
        cap = need;
    }
    if (buf) cudaMemset(buf, 0, need);            // semaphore must start zeroed
    return buf;
}

template <typename Gemm>
static cudaError_t run_cfg_splitk(const i8* A, const i8* Bt, i32* C,
                                  int m, int n, int k, i32 beta, int slices,
                                  cudaStream_t stream)
{
    Gemm op;
    typename Gemm::Arguments args({m, n, k}, {A, k}, {Bt, k}, {C, n}, {C, n}, {1, beta}, slices);
    if (op.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    void* ws = splitk_workspace(Gemm::get_workspace_size(args));
    if (op.initialize(args, ws, stream) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    return op(stream) == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorInvalidValue;
}

// ---- stream-k --------------------------------------------------------------------------------
// WAVE QUANTIZATION, measured: FFN H.W_down (m=16384 n=4096 k=16384) tiles at 128x256 into
// 128 x 16 = 2048 threadblocks against 170 SMs = 12.05 waves. The fractional wave runs ~5%
// occupied, wasting 1 - 2048/(13*170) = 7.3% -- which is the -7.8% that survived every tile,
// stage and swizzle sweep. It is not kernel quality; it is the grid not dividing by the machine.
//
// Stream-K assigns each SM an equal share of MAC work rather than whole tiles, so there is no
// tail: partial tiles are reduced through a workspace. Byte-exact for the same reason split-k is
// -- the partials are int32 and integer addition is associative.
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int STAGES>
using GemmCfgStreamK = cutlass::gemm::device::GemmUniversal<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TBM, TBN, TBK>,
    cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
    STAGES,
    16, 16>;

template <typename Gemm>
static cudaError_t run_cfg_streamk(const i8* A, const i8* Bt, i32* C,
                                   int m, int n, int k, i32 beta, cudaStream_t stream)
{
    Gemm op;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,                       // batch / split factor -- stream-k does the balancing itself
        {1, beta},
        A, Bt, C, C,
        int64_t(0), int64_t(0), int64_t(0), int64_t(0),
        int64_t(k), int64_t(k), int64_t(n), int64_t(n));
    if (op.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    void* ws = splitk_workspace(Gemm::get_workspace_size(args));
    if (op.initialize(args, ws, stream) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
    return op(stream) == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorInvalidValue;
}

using CfgFn = cudaError_t (*)(const i8*, const i8*, i32*, int, int, int, i32, cudaStream_t);

// The sweep. Tile shape trades reuse against occupancy and register pressure; stage count
// trades shared memory against how deep the global->shared pipeline runs. Which one wins is
// shape-dependent and cannot be reasoned out from first principles -- that is why it is timed.
// The SHIPPED config set: every entry won at least one of the four production shapes across the
// 2026-08-05 sweeps, plus a small spread so a consensus matmul_n change has somewhere to land.
// Pruned from 51 candidates -- the tuner times all of these on first use of each shape, and at
// FFN dims 51 x 7 reps x 3.5 ms was seconds of GPU time on the first episode.
//
// Dropped, with reasons (see research/matmul-v4/FINDINGS-cutlass-gemm-2026-08-05.md):
//   TBK=128         sm_120 caps smem near 99 KB/block; 128x256x64 s4 at 98 KB is already there
//   NoBetaScaling   the beta=1 deficit is the C load, not the multiply -- measured no better
//   TBK=32          warp-K must be >= 2x the m16n8k32 instruction K (CUTLASS static-asserts)
//   swz2/swz4/swz16 never beat swz8 or swz1 on any production shape
inline CfgFn cfg_at(int i)
{
    switch (i) {
        // --- plain, default swizzle
        case 0: return &run_cfg<GemmCfg<256, 128, 64, 64, 64, 64, 4>>;      // FFN X.W_up winner
        case 1: return &run_cfg<GemmCfg<128, 256, 64, 64, 64, 64, 3>>;      // general large-shape
        case 2: return &run_cfg<GemmCfg<256, 128, 64, 64, 64, 64, 3>>;
        case 3: return &run_cfg<GemmCfg< 64,  64, 64, 32, 32, 64, 6>>;      // small/skinny output
        // --- swz8: the L2-reuse ordering that took FFN down from -15.8% to -7.1%
        case 4: return &run_cfg<GemmCfg<128, 256, 64, 64, 64, 64, 4, 8>>;   // FFN H.W_down winner
        case 5: return &run_cfg<GemmCfg<256, 128, 64, 64, 64, 64, 4, 8>>;
        case 6: return &run_cfg<GemmCfg<128, 128, 64, 64, 64, 64, 4, 8>>;   // attn Q.Kt winner
        case 7: return &run_cfg<GemmCfg< 64, 256, 64, 32, 64, 64, 4, 8>>;   // attn Q.Kt winner
        case 8: return &run_cfg<GemmCfg<128, 256, 64, 64, 64, 64, 3, 8>>;
        // --- stream-k: no wave tail, and it is what attn S.V wants
        case  9: return &run_cfg_streamk<GemmCfgStreamK<128, 256, 64, 64, 64, 64, 4>>;
        case 10: return &run_cfg_streamk<GemmCfgStreamK<128, 128, 64, 64, 64, 64, 4>>;
        case 11: return &run_cfg_streamk<GemmCfgStreamK< 64, 128, 64, 32, 64, 64, 4>>;  // attn S.V
        default: return nullptr;
    }
}
inline constexpr int CFG_COUNT = 12;

inline const char* cfg_name(int i)
{
    static const char* n[CFG_COUNT] = {
        "256x128x64 s4", "128x256x64 s3", "256x128x64 s3", "64x64x64 s6",
        "128x256x64 s4 swz8", "256x128x64 s4 swz8", "128x128x64 s4 swz8",
        "64x256x64 s4 swz8", "128x256x64 s3 swz8",
        "streamk 128x256x64 s4", "streamk 128x128x64 s4", "streamk 64x128x64 s4"};
    return (i >= 0 && i < CFG_COUNT) ? n[i] : "?";
}

// Two separate alignment constraints, and both bite:
//   k % 16 -- the 128-bit operand loads (lda and ldb are both k).
//   n %  4 -- the epilogue's 4-wide int32 store (ldc = n).
// Production shapes (k = 4096/16384, n = 4096/16384) satisfy both. Anything else is not a
// failure, it is the hand kernel's job -- which is exactly why that kernel is kept.
inline bool cutlass_eligible(int n, int k) { return (k % 16) == 0 && (n % 4) == 0; }

inline void transpose_b(const i8* B, i8* Bt, int k, int n, cudaStream_t stream)
{
    if ((k % 64) == 0 && (n % 64) == 0) {
        dim3 tg(n / 64, k / 64);
        k_transpose_i8_vec<<<tg, 256, 0, stream>>>(B, Bt, k, n);
        return;
    }
    constexpr int TILE = 32;
    dim3 tb(TILE, TILE);
    dim3 tg((n + TILE - 1) / TILE, (k + TILE - 1) / TILE);
    k_transpose_i8_tiled<TILE><<<tg, tb, 0, stream>>>(B, Bt, k, n);
}

// ---- split-k candidates ---------------------------------------------------------------------
// Small tiles, because the shapes that need split-k have tiny outputs. Slice count is a runtime
// argument, so one instantiation covers every depth.
struct SplitKFn {
    cudaError_t (*fn)(const i8*, const i8*, i32*, int, int, int, i32, int, cudaStream_t);
    const char* name;
};
inline SplitKFn splitk_cfg_at(int i)
{
    switch (i) {
        case 0: return {&run_cfg_splitk<GemmCfgSplitK< 64,  64, 64, 32, 32, 64, 4>>, "sk 64x64x64"};
        case 1: return {&run_cfg_splitk<GemmCfgSplitK< 64, 128, 64, 32, 64, 64, 4>>, "sk 64x128x64"};
        case 2: return {&run_cfg_splitk<GemmCfgSplitK<128,  64, 64, 64, 32, 64, 4>>, "sk 128x64x64"};
        case 3: return {&run_cfg_splitk<GemmCfgSplitK<128, 128, 64, 64, 64, 64, 3>>, "sk 128x128x64"};
        default: return {nullptr, "?"};
    }
}
inline constexpr int SPLITK_CFG_COUNT = 4;
// Slice counts to try. The 5090 has 170 SMs; the point is to fill them, so the useful range is
// bounded below by "enough tiles to occupy the machine" and above by reduction overhead.
inline constexpr int kSplitKSlices[] = {4, 16, 64, 192};
inline constexpr int SPLITK_SLICE_COUNT = 4;

// ---- autotune -------------------------------------------------------------------------------
// Replaces a hand-written shape rule that was measured WRONG: it served the two FFN shapes
// (-2.7% / -4.3% vs cuBLASLt) but sent the attention SV GEMM to a 256-wide tile it could not
// fill, costing ~9% of the whole episode. The episode's shape set is tiny (4 distinct shapes)
// and every one recurs 4-64 times per episode, so a one-time timed sweep per shape amortises to
// nothing -- the same reasoning BTX_RC_GEMM_TUNE already uses for cuBLASLt.
//
// Every candidate is bit-identical, so this only ever picks a SPEED, never a result.
struct TuneKey {
    int m, n, k, beta;
    bool operator<(const TuneKey& o) const {
        if (m != o.m) return m < o.m;
        if (n != o.n) return n < o.n;
        if (k != o.k) return k < o.k;
        return beta < o.beta;
    }
};
struct TunePick { int cfg; int splitk_cfg; int slices; };   // splitk_cfg < 0 => plain cfg

inline bool GemmTuneEnabledCutlass()
{
    static const bool on = [] {
        const char* v = getenv("BTX_RC_GEMM_TUNE");
        return v == nullptr || (v[0] != '0');     // default ON
    }();
    return on;
}

// ---- disk-persisted tune cache (2026-08-10) ---------------------------------------------------
// The cuBLASLt path had this (-34% cold, a50e317); the CUTLASS switch silently compiled it out
// and every ONE-SHOT process (matador-verify share audits, rc_attest replays) went back to
// re-timing the candidate set per shape -- seconds of GPU per process. A TunePick is 3 ints, so
// persistence is trivial. Keyed on GPU name + the candidate-set sizes (a config-set change
// invalidates the file); loaded picks are bounds-checked, and a pick that fails to launch falls
// back to the hand kernel at the call site -- correctness never rests on the file (every
// candidate is bit-identical anyway). Same env surface as before: BTX_RC_TUNE_CACHE (default
// ON, =0 disables the disk IO), BTX_RC_TUNE_CACHE_PATH (default
// ~/.cache/matador-miner/rc_gemm_tune_cutlass.bin). Atomic tmp+rename on save.
struct CutTuneHeader {
    uint32_t magic, fver;
    int32_t  cfg_count, splitk_count;
    char     dev[64];
};
inline bool TuneCacheEnabledCutlass()
{
    static const bool on = [] {
        const char* v = getenv("BTX_RC_TUNE_CACHE");
        return v == nullptr || (v[0] != '0');
    }();
    return on;
}
inline std::string TuneCachePathCutlass()
{
    if (const char* p = getenv("BTX_RC_TUNE_CACHE_PATH")) return std::string(p);
    const char* home = getenv("HOME");
    if (home == nullptr || home[0] == 0) return std::string();
    return std::string(home) + "/.cache/matador-miner/rc_gemm_tune_cutlass.bin";
}
inline CutTuneHeader TuneCacheHdrCutlass()
{
    CutTuneHeader h;
    std::memset(&h, 0, sizeof(h));                // padding participates in the memcmp match
    h.magic = 0x32544352u;                        // "RCT2"
    h.fver = 1;
    h.cfg_count = CFG_COUNT;
    h.splitk_count = SPLITK_CFG_COUNT;
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        std::strncpy(h.dev, prop.name, sizeof(h.dev) - 1);
    return h;
}
// Both run under the tuner's mutex.
inline void TuneCacheLoadCutlass(std::map<TuneKey, TunePick>& cache)
{
    const std::string path = TuneCachePathCutlass();
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) return;
    const CutTuneHeader want = TuneCacheHdrCutlass();
    CutTuneHeader got{};
    uint32_t count = 0;
    bool ok = fread(&got, sizeof(got), 1, f) == 1 &&
              std::memcmp(&got, &want, sizeof(got)) == 0 &&
              fread(&count, sizeof(count), 1, f) == 1 && count <= 64;
    uint32_t loaded = 0;
    for (uint32_t i = 0; ok && i < count; ++i) {
        TuneKey k{}; TunePick p{};
        if (fread(&k, sizeof(k), 1, f) != 1 || fread(&p, sizeof(p), 1, f) != 1) break;
        const bool sane = (p.cfg >= 0 && p.cfg < CFG_COUNT && p.splitk_cfg < 0) ||
                          (p.cfg < 0 && p.splitk_cfg >= 0 && p.splitk_cfg < SPLITK_CFG_COUNT &&
                           p.slices > 0 && p.slices <= 4096);
        if (!sane) continue;
        cache[k] = p;
        ++loaded;
    }
    fclose(f);
    if (loaded) printf("[rc-tune-cache] loaded %u tuned pick(s) from %s\n", loaded, path.c_str());
}
inline void TuneCacheSaveCutlass(const std::map<TuneKey, TunePick>& cache)
{
    const std::string path = TuneCachePathCutlass();
    if (path.empty()) return;
    const size_t slash = path.rfind('/');
    if (slash != std::string::npos) {             // best-effort mkdir -p, two levels
        const std::string dir = path.substr(0, slash);
        const size_t s2 = dir.rfind('/');
        if (s2 != std::string::npos) ::mkdir(dir.substr(0, s2).c_str(), 0755);
        ::mkdir(dir.c_str(), 0755);
    }
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) return;
    const CutTuneHeader h = TuneCacheHdrCutlass();
    const uint32_t count = (uint32_t)cache.size();
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1 && fwrite(&count, sizeof(count), 1, f) == 1;
    for (auto it = cache.begin(); ok && it != cache.end(); ++it)
        ok = fwrite(&it->first, sizeof(it->first), 1, f) == 1 &&
             fwrite(&it->second, sizeof(it->second), 1, f) == 1;
    fclose(f);
    if (ok) ::rename(tmp.c_str(), path.c_str()); else ::remove(tmp.c_str());
}

// Times candidates into `dst`, which the caller guarantees is safe to clobber.
// Times every candidate in INTERLEAVED PASSES and keeps each one's global minimum.
//
// Why not a single pass: with locked clocks cuBLASLt's episode time varies 2.4 ms run to run
// while ours varied 24 ms with IDENTICAL tuner picks -- so the spread was the tuner resolving a
// few-percent difference through its own noise and occasionally crowning the wrong config. One
// run that picked well came in at 935.0 ms against cuBLASLt's 947.5, so the good configs are
// reachable; they just were not being found reliably.
//
// Interleaving matters more than raw rep count: a transient (another process, a power excursion)
// lands on whichever candidate happens to be timed during it. Spreading each candidate's samples
// across passes makes interference hit all of them roughly equally instead of libelling one.
inline TunePick tune_shape(const i8* A, const i8* Bt, i32* dst,
                           int m, int n, int k, i32 beta, cudaStream_t stream)
{
    struct Cand { int cfg; int sk; int slices; float t; };
    Cand cand[CFG_COUNT + SPLITK_CFG_COUNT * SPLITK_SLICE_COUNT];
    int ncand = 0;

    for (int i = 0; i < CFG_COUNT; ++i)
        if (cfg_at(i)) cand[ncand++] = {i, -1, 0, 3e38f};

    // Split-k exists to create parallelism a shape lacks. If the plain tiling already yields
    // several times the SM count it cannot win, and timing it is wasted startup.
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const long tiles = (long)((m + 127) / 128) * ((n + 127) / 128);
    if (sms <= 0 || tiles < 4L * sms) {
        for (int i = 0; i < SPLITK_CFG_COUNT; ++i) {
            if (!splitk_cfg_at(i).fn) continue;
            for (int si = 0; si < SPLITK_SLICE_COUNT; ++si) {
                const int sl = kSplitKSlices[si];
                if (sl > k / 64) continue;
                cand[ncand++] = {-1, i, sl, 3e38f};
            }
        }
    }
    if (ncand == 0) return TunePick{1, -1, 0};

    auto launch = [&](const Cand& c) -> cudaError_t {
        if (c.cfg >= 0) return cfg_at(c.cfg)(A, Bt, dst, m, n, k, beta, stream);
        return splitk_cfg_at(c.sk).fn(A, Bt, dst, m, n, k, beta, c.slices, stream);
    };

    // Warm every candidate before any is measured, so clock ramp is not attributed to whichever
    // one happens to be timed first.
    for (int c = 0; c < ncand; ++c)
        if (launch(cand[c]) != cudaSuccess) { cudaGetLastError(); cand[c].t = -1.0f; }
    cudaStreamSynchronize(stream);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    constexpr int kPasses = 3, kRepsPerPass = 3;
    for (int pass = 0; pass < kPasses; ++pass) {
        for (int c = 0; c < ncand; ++c) {
            if (cand[c].t < 0.0f) continue;               // failed to launch at warm-up
            for (int r = 0; r < kRepsPerPass; ++r) {
                cudaEventRecord(e0, stream);
                if (launch(cand[c]) != cudaSuccess) { cudaGetLastError(); cand[c].t = -1.0f; break; }
                cudaEventRecord(e1, stream);
                if (cudaEventSynchronize(e1) != cudaSuccess) { cudaGetLastError(); cand[c].t = -1.0f; break; }
                float ms = 3e38f;
                cudaEventElapsedTime(&ms, e0, e1);
                if (ms < cand[c].t) cand[c].t = ms;
            }
        }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);

    int best = -1;
    for (int c = 0; c < ncand; ++c)
        if (cand[c].t > 0.0f && (best < 0 || cand[c].t < cand[best].t)) best = c;
    if (best < 0) return TunePick{1, -1, 0};
    return TunePick{cand[best].cfg, cand[best].sk, cand[best].slices};
}

// Tuned dispatch over an ALREADY-k-contiguous B (i.e. B^T materialised, or an operand that is
// natively (n,k) row-major). Shares one tune cache with the transpose path: both feed the
// identical bytes to the identical candidate set, so a pick made through either is valid for
// both. cfg < 0 selects the tuned/default config; otherwise the named one (the probe's sweep).
inline cudaError_t gemm_i8_bt_cutlass(const i8* A, const i8* Bt, i32* C,
                                      int m, int n, int k, i32 beta,
                                      cudaStream_t stream, int cfg = -1)
{
    if (cfg >= 0) {                       // explicit config (the probe's sweep)
        gemm_validate() = true;
        CfgFn fn = cfg_at(cfg);
        if (!fn) return cudaErrorInvalidValue;
        return fn(A, Bt, C, m, n, k, beta, stream);
    }

    // Tuned path. One timed sweep per distinct shape, cached for the process lifetime and
    // persisted to disk so one-shot validator processes skip the sweep entirely.
    static std::map<TuneKey, TunePick> cache;
    static std::mutex mtx;
    static std::once_flag disk_once;
    const TuneKey key{m, n, k, beta};
    TunePick pick{1, -1, 0};
    {
        std::lock_guard<std::mutex> lk(mtx);
        if (TuneCacheEnabledCutlass())
            std::call_once(disk_once, [&] { TuneCacheLoadCutlass(cache); });
        auto it = cache.find(key);
        if (it != cache.end()) {
            pick = it->second;
        } else if (GemmTuneEnabledCutlass()) {
            // Candidates must NOT be timed into the real C: the beta=1 down-projection has its
            // residual preloaded there, and repeated runs would corrupt it. Tune into a throwaway
            // instead -- one allocation per shape, freed immediately, never on the steady path.
            i32* dst = C;
            i32* tmp = nullptr;
            if (beta != 0 && cudaMalloc(&tmp, (size_t)m * n * sizeof(i32)) == cudaSuccess) dst = tmp;
            else if (beta != 0) { cudaGetLastError(); dst = nullptr; }
            if (dst != nullptr) {
                gemm_validate() = true;
                pick = tune_shape(A, Bt, dst, m, n, k, beta, stream);
                gemm_validate() = false;
                if (pick.cfg >= 0)
                    printf("[rc-gemm-tune] m=%d n=%d k=%d beta=%d -> %s\n",
                           m, n, k, beta, cfg_name(pick.cfg));
                else
                    printf("[rc-gemm-tune] m=%d n=%d k=%d beta=%d -> %s slices=%d\n",
                           m, n, k, beta, splitk_cfg_at(pick.splitk_cfg).name, pick.slices);
            }
            if (tmp) cudaFree(tmp);
            cache[key] = pick;
            if (TuneCacheEnabledCutlass()) TuneCacheSaveCutlass(cache);
        } else {
            cache[key] = pick;
        }
    }

    gemm_validate() = false;
    if (pick.cfg >= 0) {
        CfgFn fn = cfg_at(pick.cfg);
        if (!fn) return cudaErrorInvalidValue;
        return fn(A, Bt, C, m, n, k, beta, stream);
    }
    const SplitKFn sk = splitk_cfg_at(pick.splitk_cfg);
    if (!sk.fn) return cudaErrorInvalidValue;
    return sk.fn(A, Bt, C, m, n, k, beta, pick.slices, stream);
}

// scratch must hold at least (size_t)k*n bytes -- the transposed B.
// cfg < 0 selects the default config; otherwise the named one (used by the sweep).
inline cudaError_t gemm_i8_nn_cutlass(const i8* A, const i8* B, i32* C,
                                      int m, int n, int k, i32 beta,
                                      i8* scratch, cudaStream_t stream, int cfg = 0)
{
    transpose_b(B, scratch, k, n, stream);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    return gemm_i8_bt_cutlass(A, scratch, C, m, n, k, beta, stream, cfg);
}

// B^T supplied directly (an operand already materialised k-contiguous, e.g. attention K for
// C = Q.K^T where the expanded K is (n_ctx, d_head) row-major = exactly (n, k) at ld = k).
// Returns cudaErrorInvalidValue when the shape is not CUTLASS-eligible; the caller keeps its
// transpose fallback. Byte-exact: identical operand bytes reach the identical kernels.
inline cudaError_t gemm_i8_nn_bt_auto(const i8* A, const i8* Bt, i32* C,
                                      int m, int n, int k, i32 beta, cudaStream_t stream)
{
    if (!cutlass_eligible(n, k)) return cudaErrorInvalidValue;
    const cudaError_t e = gemm_i8_bt_cutlass(A, Bt, C, m, n, k, beta, stream, -1);
    if (e != cudaSuccess) cudaGetLastError();
    return e;
}

// THE ENTRY POINT the solvers call. Same contract as gemm8() in rc_gpu_episode.cu:
// row-major C(m,n) = A(m,k).B(k,n) [+ C when beta], int8 -> int32, alpha = 1.
//
// Dispatch is on shape alone, and both paths are bit-identical to each other and to the CPU
// reference -- int8 into int32 is exact integer arithmetic, so which kernel ran can never
// change the digest. That is the property that makes falling back safe.
//
// scratch may be null; then the CUTLASS path is skipped regardless of shape.
inline cudaError_t gemm_i8_nn_auto(const i8* A, const i8* B, i32* C,
                                   int m, int n, int k, i32 beta,
                                   i8* scratch, cudaStream_t stream)
{
    if (scratch != nullptr && cutlass_eligible(n, k)) {
        const cudaError_t e = gemm_i8_nn_cutlass(A, B, C, m, n, k, beta, scratch, stream, -1);
        if (e == cudaSuccess) return e;
        cudaGetLastError();          // clear and fall through -- correctness never depends on it
    }
    gemm_i8_nn(A, B, C, m, n, k, k, n, n, beta, stream);
    return cudaGetLastError();
}

}  // namespace rcgemm
