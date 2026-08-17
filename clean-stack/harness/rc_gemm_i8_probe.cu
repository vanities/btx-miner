// rc_gemm_i8_probe.cu -- correctness + speed gate for the in-house int8 GEMM that replaces
// the static cuBLASLt link (see cuda/rc_gemm_i8.cuh for why).
//
//   rc_gemm_i8_probe              -- correctness across the shapes the solvers actually use
//   rc_gemm_i8_probe --bench      -- ours vs cuBLASLt at the profile-1 activation shapes
//
// CORRECTNESS BAR IS BIT-EQUALITY, not a tolerance. int8*int8 -> int32 is exact integer
// arithmetic, so a correct kernel matches the CPU reference on every element; anything else
// is a bug, not rounding. That is what makes this swap safe to make at all.

#include "../core/cuda/rc_gemm_i8.cuh"
#ifdef MATADOR_HAVE_CUTLASS
#include "../core/cuda/rc_gemm_i8_cutlass.cuh"
#endif

#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using rcgemm::i8;
using rcgemm::i32;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) {                       \
    printf("!! %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));      \
    exit(2); } } while (0)

// Row-major C(m,n) = A(m,k).B(k,n) [+ C], the same contract as gemm8() in rc_gpu_episode.cu.
static void cpu_ref(const std::vector<i8>& A, const std::vector<i8>& B, std::vector<i32>& C,
                    int m, int n, int k, i32 beta)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            i32 acc = 0;
            for (int p = 0; p < k; ++p) acc += (i32)A[(size_t)i * k + p] * (i32)B[(size_t)p * n + j];
            C[(size_t)i * n + j] = beta ? C[(size_t)i * n + j] + acc : acc;
        }
}

// Operand magnitudes match the solver's: the episode feeds bounded int8 (|resid| <= 48 per
// the GuardInt32Bound note), but we deliberately probe the full int8 range to catch sign and
// overflow-adjacent bugs the narrow production range would hide.
static void fill(std::vector<i8>& v, unsigned seed, int lo = -128, int hi = 127)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(lo, hi);
    for (auto& x : v) x = (i8)d(rng);
}

static bool one_case(int m, int n, int k, i32 beta, unsigned seed, bool quiet = false)
{
    std::vector<i8>  hA((size_t)m * k), hB((size_t)k * n);
    std::vector<i32> hC((size_t)m * n), ref;
    fill(hA, seed); fill(hB, seed + 1);
    { std::mt19937 rng(seed + 2); std::uniform_int_distribution<int> d(-1000, 1000);
      for (auto& x : hC) x = beta ? d(rng) : 0; }
    ref = hC;
    cpu_ref(hA, hB, ref, m, n, k, beta);
    const std::vector<i32> hCin = hC;   // check() overwrites hC with the device result

    i8 *dA, *dB; i32* dC;
    CK(cudaMalloc(&dA, hA.size()));
    CK(cudaMalloc(&dB, hB.size()));
    CK(cudaMalloc(&dC, hC.size() * sizeof(i32)));
    CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dC, hC.data(), hC.size() * sizeof(i32), cudaMemcpyHostToDevice));

    auto check = [&](const char* who) -> size_t {
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(i32), cudaMemcpyDeviceToHost));
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < hC.size(); ++i)
            if (hC[i] != ref[i]) { if (!bad) first = i; ++bad; }
        if (!quiet) {
            if (bad) printf("  FAIL %-8s m=%-6d n=%-6d k=%-6d beta=%d : %zu/%zu mismatched, "
                            "first at %zu (got %d want %d)\n",
                            who, m, n, k, beta, bad, hC.size(), first, hC[first], ref[first]);
            else     printf("  ok   %-8s m=%-6d n=%-6d k=%-6d beta=%d : %zu elements bit-identical\n",
                            who, m, n, k, beta, hC.size());
        }
        return bad;
    };

    rcgemm::gemm_i8_nn(dA, dB, dC, m, n, k, k, n, n, beta, 0);
    size_t bad = check("hand");

#ifdef MATADOR_HAVE_CUTLASS
    if (rcgemm::cutlass_eligible(n, k)) {
        i8* scratch; CK(cudaMalloc(&scratch, (size_t)k * n));
        CK(cudaMemcpy(dC, hCin.data(), hC.size() * sizeof(i32), cudaMemcpyHostToDevice));
        cudaError_t e = rcgemm::gemm_i8_nn_cutlass(dA, dB, dC, m, n, k, beta, scratch, 0);
        if (e != cudaSuccess) { printf("  FAIL cutlass launch: %s\n", cudaGetErrorString(e)); ++bad; }
        else bad += check("cutlass");
        cudaFree(scratch);
    }
#endif

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad == 0;
}

static float time_ms(void (*fn)(void*), void* ctx, int iters)
{
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    fn(ctx); CK(cudaDeviceSynchronize());            // warm
    CK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) fn(ctx);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

struct BenchCtx { const i8 *A, *B; i32* C; int m, n, k; i32 beta;
                  cublasLtHandle_t lt; void* ws; size_t wsz; i8* scratch; };

static void run_ours(void* p)
{
    BenchCtx* c = (BenchCtx*)p;
    rcgemm::gemm_i8_nn(c->A, c->B, c->C, c->m, c->n, c->k, c->k, c->n, c->n, c->beta, 0);
}

#ifdef MATADOR_HAVE_CUTLASS
static int g_cfg = 0;   // which tile config the sweep is currently timing
// Includes the B transpose -- see the note in rc_gemm_i8_cutlass.cuh. Timing only the GEMM
// would flatter this arm against cuBLASLt, which does its operand transform internally.
static void run_cutlass(void* p)
{
    BenchCtx* c = (BenchCtx*)p;
    rcgemm::gemm_i8_nn_cutlass(c->A, c->B, c->C, c->m, c->n, c->k, c->beta, c->scratch, 0, g_cfg);
}
static int g_sk_cfg = 0, g_sk_slices = 8;
static void run_cutlass_sk(void* p)
{
    BenchCtx* c = (BenchCtx*)p;
    rcgemm::transpose_b(c->B, c->scratch, c->k, c->n, 0);
    rcgemm::splitk_cfg_at(g_sk_cfg).fn(c->A, c->scratch, c->C, c->m, c->n, c->k, c->beta,
                                       g_sk_slices, 0);
}
static void run_transpose_only(void* p)
{
    BenchCtx* c = (BenchCtx*)p;
    rcgemm::transpose_b(c->B, c->scratch, c->k, c->n, 0);
}
#endif

// Same descriptor mapping gemm8() uses: operands swapped so cuBLASLt's column-major view of
// (B,A) computes our row-major C = A.B.
//
// THE BASELINE MUST BE THE AUTOTUNED cuBLASLt, NOT THE DEFAULT ONE. Passing algo=nullptr uses
// NVIDIA's generic heuristic, but the episode runs BTX_RC_GEMM_TUNE (default ON), which times
// heuristic candidates per shape and keeps the fastest -- worth a measured +4.4% (2026-07-31).
// Benchmarking against the nullptr path understated the real target by that much and is why the
// isolated bench said -2.7%/-4.3% while the episode measured -9% on the same configs.
static bool g_lt_tuned = true;
static cublasLtMatmulAlgo_t g_lt_algo;
static bool g_lt_have_algo = false;

static void lt_call(BenchCtx* c, const cublasLtMatmulAlgo_t* algo)
{
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasOperation_t nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &nn, sizeof(nn));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof(nn));
    cublasLtMatrixLayout_t l0, l1, lc;
    cublasLtMatrixLayoutCreate(&l0, CUDA_R_8I,  c->n, c->k, c->n);
    cublasLtMatrixLayoutCreate(&l1, CUDA_R_8I,  c->k, c->m, c->k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, c->n, c->m, c->n);
    i32 alpha = 1, beta = c->beta;
    cublasLtMatmul(c->lt, op, &alpha, c->B, l0, c->A, l1, &beta, c->C, lc, c->C, lc,
                   algo, c->ws, c->wsz, 0);
    cublasLtMatrixLayoutDestroy(l0); cublasLtMatrixLayoutDestroy(l1);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
}

static void run_cublaslt(void* p)
{
    BenchCtx* c = (BenchCtx*)p;
    lt_call(c, (g_lt_tuned && g_lt_have_algo) ? &g_lt_algo : nullptr);
}

// Mirror of gemm8_ex's tune: pull heuristic candidates, time each, keep the fastest.
static void lt_tune(BenchCtx* c)
{
    g_lt_have_algo = false;
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasOperation_t nn = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &nn, sizeof(nn));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nn, sizeof(nn));
    cublasLtMatrixLayout_t l0, l1, lc;
    cublasLtMatrixLayoutCreate(&l0, CUDA_R_8I,  c->n, c->k, c->n);
    cublasLtMatrixLayoutCreate(&l1, CUDA_R_8I,  c->k, c->m, c->k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, c->n, c->m, c->n);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &c->wsz, sizeof(c->wsz));
    cublasLtMatmulHeuristicResult_t cand[32]; int ncand = 0;
    cublasLtMatmulAlgoGetHeuristic(c->lt, op, l0, l1, lc, lc, pref, 32, cand, &ncand);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(l0); cublasLtMatrixLayoutDestroy(l1);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);

    float best = 3e38f;
    for (int i = 0; i < ncand; ++i) {
        for (int w = 0; w < 2; ++w) lt_call(c, &cand[i].algo);
        CK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        float tmin = 3e38f;
        for (int r = 0; r < 5; ++r) {
            CK(cudaEventRecord(e0));
            lt_call(c, &cand[i].algo);
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < tmin) tmin = ms;
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        if (tmin < best) { best = tmin; g_lt_algo = cand[i].algo; g_lt_have_algo = true; }
    }
    printf("      [cuBLASLt autotune: %d candidates, best %.3f ms]\n", ncand, best);
}

static void bench(int m, int n, int k, i32 beta, const char* label)
{
    std::vector<i8> hA((size_t)m * k), hB((size_t)k * n);
    fill(hA, 7); fill(hB, 9);
    i8 *dA, *dB; i32* dC;
    CK(cudaMalloc(&dA, hA.size()));
    CK(cudaMalloc(&dB, hB.size()));
    CK(cudaMalloc(&dC, (size_t)m * n * sizeof(i32)));
    CK(cudaMemcpy(dA, hA.data(), hA.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
    CK(cudaMemset(dC, 0, (size_t)m * n * sizeof(i32)));

    BenchCtx c{dA, dB, dC, m, n, k, beta, nullptr, nullptr, 0, nullptr};
    cublasLtCreate(&c.lt);
    c.wsz = 32u << 20; CK(cudaMalloc(&c.ws, c.wsz));
    CK(cudaMalloc(&c.scratch, (size_t)k * n));

    const int iters = 20;
    const double mac = 2.0 * (double)m * n * k;   // 1 MAC = 1 mul + 1 add
    auto tops = [&](float ms) { return mac / (ms * 1e-3) / 1e12; };
    printf("  %-22s m=%-6d n=%-6d k=%-6d beta=%d\n", label, m, n, k, beta);
    g_lt_tuned = false;
    const float t_lt_def = time_ms(run_cublaslt, &c, iters);
    printf("      cuBLASLt(default) %8.3f ms  %7.1f TOPS\n", t_lt_def, tops(t_lt_def));
    lt_tune(&c);
    g_lt_tuned = true;
    const float t_lt = time_ms(run_cublaslt, &c, iters);
    printf("      cuBLASLt(TUNED)   %8.3f ms  %7.1f TOPS   <- the real target\n",
           t_lt, tops(t_lt));
#ifdef MATADOR_HAVE_CUTLASS
    // Sweep every compiled tile config -- this is our equivalent of cuBLASLt's autotune, and
    // the reason a single blind config is not a fair read on what CUTLASS can do here.
    int best = -1; float t_best = 1e30f;
    for (int i = 0; i < rcgemm::CFG_COUNT; ++i) {
        g_cfg = i;
        if (rcgemm::gemm_i8_nn_cutlass(c.A, c.B, c.C, c.m, c.n, c.k, c.beta, c.scratch, 0, i)
            != cudaSuccess) {
            cudaGetLastError();
            printf("      cutlass %-16s  unsupported for this shape\n", rcgemm::cfg_name(i));
            continue;
        }
        CK(cudaDeviceSynchronize());
        const float t = time_ms(run_cutlass, &c, iters);
        printf("      cutlass %-16s %8.3f ms  %7.1f TOPS   %+.1f%%\n",
               rcgemm::cfg_name(i), t, tops(t), 100.0 * (t_lt / t - 1.0));
        if (t < t_best) { t_best = t; best = i; }
    }
    if (best >= 0) {
        printf("      -> best cutlass: %s  %+.1f%% vs cuBLASLt\n",
               rcgemm::cfg_name(best), 100.0 * (t_lt / t_best - 1.0));
        // How much of the remaining gap is the B transpose? If the solvers can materialise
        // their sigma-derived weights already transposed, that cost goes away entirely -- so
        // it is worth knowing separately from the kernel's own deficit.
        const float t_tr = time_ms(run_transpose_only, &c, iters);
        printf("         of which transpose: %.3f ms (%.0f%% of the gap); "
               "GEMM alone would be %+.1f%%\n",
               t_tr, 100.0 * t_tr / (t_best - t_lt),
               100.0 * (t_lt / (t_best - t_tr) - 1.0));
    }
#endif
    const float t_ours = time_ms(run_ours, &c, iters);
    printf("      handwritten%8.3f ms  %7.1f TOPS   %+.1f%% vs cuBLASLt\n",
           t_ours, tops(t_ours), 100.0 * (t_lt / t_ours - 1.0));

    cudaFree(c.scratch); cudaFree(c.ws); cublasLtDestroy(c.lt);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv)
{
    const bool do_bench = argc > 1 && strcmp(argv[1], "--bench") == 0;

    if (do_bench) {
        // Profile 1 = the Epoch-A activation shape: b_seq=16384, d_model=4096, d_ff=16384.
        // These two FFN GEMMs are the ones that dominate episode time.
        // The four GEMMs the episode issues at profile-1 dims (d_head=128, n_q=512,
        // n_ctx=786432, d_model=4096, d_ff=16384, b_seq=16384). FFN runs 16x per round,
        // attention twice per round, so weight them accordingly when reading the totals.
        printf("[rc_gemm_i8] profile-1 episode shapes\n");
        bench(16384, 16384, 4096, 0, "FFN X.W_up      x64");
        bench(16384, 4096, 16384, 1, "FFN H.W_down+X  x64");
        bench(512, 786432, 128, 0, "attn Q.Kt        x4");
        bench(512, 128, 786432, 0, "attn S.V         x4");
        // Diagnostic: the SAME shape as FFN down but beta=0. FFN down is the only shape still
        // behind, and it is also the only one with beta=1 (its epilogue re-reads C, 268 MB at
        // these dims). If beta=0 closes the gap, the deficit is the epilogue, not the mainloop.
        bench(16384, 4096, 16384, 0, "DIAG down beta=0");
        return 0;
    }

    printf("[rc_gemm_i8] correctness -- bit-equality against the CPU reference\n");
    bool ok = true;
    // Shapes the solvers use, plus deliberately ragged ones: the tile is 128x128x64, so these
    // exercise partial tiles in every dimension. Zero-padding a partial tile is exact, but
    // only if the bounds logic is right -- that is what these catch.
    ok &= one_case(128, 128, 64, 0, 11);      // exactly one tile
    ok &= one_case(128, 128, 64, 1, 12);      // beta path
    ok &= one_case(256, 256, 128, 0, 13);
    ok &= one_case(1, 1, 32, 0, 14);          // degenerate
    ok &= one_case(17, 33, 65, 0, 15);        // ragged in all three dims
    ok &= one_case(129, 129, 65, 1, 16);      // one past a tile, beta on
    ok &= one_case(64, 4096, 4096, 0, 17);    // skinny m (attention-like)
    ok &= one_case(4096, 64, 4096, 0, 18);    // skinny n
    ok &= one_case(512, 512, 4096, 1, 19);    // deep k, beta on
    ok &= one_case(1024, 1024, 512, 0, 20);

    printf("[rc_gemm_i8] %s\n", ok ? "ALL PASS" : "FAILED");
    return ok ? 0 : 1;
}
