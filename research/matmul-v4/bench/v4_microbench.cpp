// MatMul v4 work-shape micro-benchmark.
//
// Faithfully reproduces the *shape* of the v4 per-nonce miner workload as
// implemented in the PR #89 reference (src/matmul/*.cpp):
//
//   1. operand expansion  A,B (n x n) + U (m x n) + V (n x m), one SHA-256 per
//      element (SampleBalancedS8FromOracle) -- rejection sample byte[0] to s8.
//   2. two exact s8xs8->s32 GEMMs  P = U*A (m x n), Q = B*V (n x m)   (§E.3)
//   3. Fq combine  Chat[a][c] = (sum_k P[a][k]*Q[k][c]) mod q,  q = 2^61-1
//
// Goal: measure where the time goes -- SHA operand-gen vs INT8 matmul vs the
// mod-q combine -- and extrapolate to n=4096. NOT byte-exact to BTX (no header
// plumbing); the *work split* is derivation-shape-invariant, which is the point.
//
// SHA-256 via CommonCrypto (hardware SHA on Apple silicon) so the SHA number is
// a FAST implementation -- the fair comparison to a real miner.

#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// --- consensus-exact element derivation (retry-0 path, ~98% of draws) --------
static constexpr uint8_t kReject = 251;
static constexpr int32_t kBal = 125;

// One SHA-256 of (seed[32] || index_le[4]); mirror SampleBalancedS8FromOracle.
// Loops on rejection exactly like the reference (byte >= 251 -> append retry_le).
static inline int8_t SampleS8(const uint8_t seed[32], uint32_t index) {
    for (uint32_t retry = 0; retry < 256; ++retry) {
        uint8_t buf[40];
        memcpy(buf, seed, 32);
        buf[32] = index & 0xff; buf[33] = (index >> 8) & 0xff;
        buf[34] = (index >> 16) & 0xff; buf[35] = (index >> 24) & 0xff;
        size_t len = 36;
        if (retry > 0) {
            buf[36] = retry & 0xff; buf[37] = (retry >> 8) & 0xff;
            buf[38] = (retry >> 16) & 0xff; buf[39] = (retry >> 24) & 0xff;
            len = 40;
        }
        uint8_t h[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(buf, (CC_LONG)len, h);
        if (h[0] < kReject) return (int8_t)((int32_t)h[0] - kBal);
    }
    return 0;
}

static void expand(std::vector<int8_t>& out, const uint8_t seed[32], size_t count) {
    out.resize(count);
    for (size_t i = 0; i < count; ++i) out[i] = SampleS8(seed, (uint32_t)i);
}

// --- Fq = 2^61-1 (mirror int8_field.cpp) ------------------------------------
static constexpr uint64_t kQ = ((uint64_t)1 << 61) - 1;
static inline uint64_t FqReduce(unsigned __int128 x) {
    uint64_t lo = (uint64_t)(x & kQ);
    uint64_t hi = (uint64_t)(x >> 61);
    uint64_t s = lo + hi;
    s = (s & kQ) + (s >> 61);
    if (s >= kQ) s -= kQ;
    return s;
}
static inline uint64_t FqAdd(uint64_t a, uint64_t b) { uint64_t s = a + b; if (s >= kQ) s -= kQ; return s; }
static inline uint64_t FqMul(uint64_t a, uint64_t b) { return FqReduce((unsigned __int128)a * b); }
static inline uint64_t FqFromS32(int32_t x) {
    if (x >= 0) return FqReduce((unsigned __int128)(uint64_t)x);
    uint64_t mag = (uint64_t)(-(int64_t)x);
    uint64_t r = FqReduce((unsigned __int128)mag);
    return r == 0 ? 0 : kQ - r;
}

int main(int argc, char** argv) {
    uint32_t n = (argc > 1) ? (uint32_t)atoi(argv[1]) : 512;
    const uint32_t b = 8, m = n / b;
    printf("=== MatMul v4 work-shape microbench  n=%u  m=n/8=%u ===\n", n, m);

    uint8_t seedA[32], seedB[32], seedU[32], seedV[32];
    for (int i = 0; i < 32; ++i) { seedA[i]=1+i; seedB[i]=100+i; seedU[i]=50+i; seedV[i]=200+i; }

    // ---- Stage 1: operand expansion (SHA-256 per element) ----
    std::vector<int8_t> A, B, U, V;
    auto t0 = clk::now();
    expand(A, seedA, (size_t)n * n);
    expand(B, seedB, (size_t)n * n);
    expand(U, seedU, (size_t)m * n);
    expand(V, seedV, (size_t)n * m);
    double t_expand = ms_since(t0);
    uint64_t sha_count = 2ull*n*n + 2ull*m*n;  // ~1.02x with rejections

    // ---- Stage 2: two exact s8->s32 GEMMs  P=U*A (m x n), Q=B*V (n x m) ----
    std::vector<int32_t> P((size_t)m * n, 0), Q((size_t)n * m, 0);
    t0 = clk::now();
    for (uint32_t a = 0; a < m; ++a) {
        const int8_t* urow = &U[(size_t)a * n];
        int32_t* prow = &P[(size_t)a * n];
        for (uint32_t i = 0; i < n; ++i) {
            int32_t u = urow[i]; if (!u) continue;
            const int8_t* arow = &A[(size_t)i * n];
            for (uint32_t k = 0; k < n; ++k) prow[k] += u * (int32_t)arow[k];
        }
    }
    for (uint32_t k = 0; k < n; ++k) {
        const int8_t* brow = &B[(size_t)k * n];
        int32_t* qrow = &Q[(size_t)k * m];
        for (uint32_t j = 0; j < n; ++j) {
            int32_t bb = brow[j]; if (!bb) continue;
            const int8_t* vrow = &V[(size_t)j * m];
            for (uint32_t c = 0; c < m; ++c) qrow[c] += bb * (int32_t)vrow[c];
        }
    }
    double t_gemm = ms_since(t0);
    uint64_t gemm_macs = 2ull*n*n*m;

    // ---- Stage 3: Fq combine  Chat = (P*Q) mod q  (m x m) ----
    std::vector<uint64_t> Chat((size_t)m * m, 0);
    t0 = clk::now();
    for (uint32_t a = 0; a < m; ++a) {
        const int32_t* prow = &P[(size_t)a * n];
        uint64_t* crow = &Chat[(size_t)a * m];
        for (uint32_t k = 0; k < n; ++k) {
            int32_t p = prow[k]; if (!p) continue;
            uint64_t pf = FqFromS32(p);
            const int32_t* qrow = &Q[(size_t)k * m];
            for (uint32_t c = 0; c < m; ++c)
                crow[c] = FqAdd(crow[c], FqMul(pf, FqFromS32(qrow[c])));
        }
    }
    double t_combine = ms_since(t0);
    uint64_t combine_macs = (uint64_t)n * m * m;

    // volatile sink so nothing is optimized away
    volatile uint64_t sink = Chat[0] ^ Chat[Chat.size()-1] ^ (uint64_t)P[0] ^ (uint64_t)Q.back();
    (void)sink;

    double total = t_expand + t_gemm + t_combine;
    printf("\n stage            time(ms)     %%total     work\n");
    printf(" ---------------------------------------------------------------\n");
    printf(" 1 operand-expand %9.2f  %7.2f%%   %llu SHA-256\n", t_expand, 100*t_expand/total, (unsigned long long)sha_count);
    printf(" 2 INT8 GEMMs     %9.2f  %7.2f%%   %llu MACs (s8->s32)\n", t_gemm, 100*t_gemm/total, (unsigned long long)gemm_macs);
    printf(" 3 Fq combine     %9.2f  %7.2f%%   %llu modmul (q=2^61-1)\n", t_combine, 100*t_combine/total, (unsigned long long)combine_macs);
    printf(" ---------------------------------------------------------------\n");
    printf(" TOTAL/nonce      %9.2f            %.1f nonce/s (1 core)\n", total, 1000.0/total);

    double sha_rate = sha_count / (t_expand/1000.0) / 1e9;
    double gemm_rate = gemm_macs / (t_gemm/1000.0) / 1e9;
    printf("\n measured this core:  SHA-256 = %.2f GH/s   scalar-INT8 = %.2f GMAC/s\n", sha_rate, gemm_rate);
    printf(" SHA share of matmul work = %.0fx the GEMM (this core)\n", t_expand / t_gemm);

    // ---- extrapolate to n=4096 (SHA ~ n^2, GEMM ~ n^2*m = n^3/8) ----
    if (n != 4096) {
        double s = 4096.0 / n;
        double e4 = t_expand * s * s;
        double g4 = t_gemm * s * s * s;      // MACs ~ n^3
        double c4 = t_combine * s * s * s;
        printf("\n extrapolated to n=4096 (this core, single-thread):\n");
        printf("   expand ~%.0f ms   gemm ~%.0f ms   combine ~%.0f ms   total ~%.0f ms  (%.1f nonce/s)\n",
               e4, g4, c4, e4+g4+c4, 1000.0/(e4+g4+c4));
        printf("   per-nonce SHA-256 count at n=4096 = %llu (~%.1f M)\n",
               (unsigned long long)(2ull*4096*4096 + 2ull*512*4096),
               (2.0*4096*4096 + 2.0*512*4096)/1e6);
    }
    return 0;
}
