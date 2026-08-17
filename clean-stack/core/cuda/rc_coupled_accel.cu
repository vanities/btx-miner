// rc_coupled_accel.cu -- GPU coupled-V3 digest backend for matador_core.
//
// Same construction as rc_episode_accel.cu: include the byte-exact episode research file as a
// library (RC_GPU_SOLVER_AS_LIB) and REUSE its primitives -- host/device SHA-256, ChaCha20, the
// M11 table, the two-pass MX rejection-sampling expansion (gpu_expand_mx_dequant), the cuBLASLt
// int8 IMMA entry points, and the pooled device allocator. The coupled puzzle shares every one
// of those primitives with the episode (same MatExpand MX family, same domains), so inheriting
// them by inclusion preserves byte-exactness by construction; only the coupled-specific pieces
// (bank pages, lobe GEMM schedule, permutation/butterfly-mix/exchange rounds, i64 Extract) are
// implemented here, each mirroring matmul_v4_rc_coupled.cpp's reference exactly.
//
// Gated by rc_gpu_coupled_probe: GPU digest == CPU oracle == frozen medium-V3 golden a4bb0cc4,
// plus a production-structure shape (exchange_rounds > 0, 8 barriers) GPU == CPU.
//
// Residency: bank pages are nonce-INDEPENDENT (keyed by the template's bank_root_seed), so when
// the whole dequantized bank fits VRAM it is derived ONCE per template and cached across nonces
// (thread-local, keyed by brs). When it does not fit (production 1536 x 64 MiB = 96 GiB vs a
// 32 GiB card), pages are re-derived on demand per use -- the streamed path the episode file's
// stream_penalty_bench quantified. Cross-nonce page batching is the next lever, not this file.

#define RC_GPU_SOLVER_AS_LIB 1
#include "rc_gpu_episode.cu"

#include <unordered_map>

// ============================================================ coupled V3 constants (mirror
// matmul_v4_rc_coupled.cpp verbatim -- independent COUP_*_V3 domain family)
static constexpr char kCoupBankTag[]="BTX_RC_COUP_BANK_V3", kCoupLobeTag[]="BTX_RC_COUP_LOBE_V3",
                      kCoupBarrierTag[]="BTX_RC_COUP_BARRIER_V3", kCoupPermTag[]="BTX_RC_COUP_PERM_V3",
                      kCoupExtractTag[]="BTX_RC_COUP_EXTRACT_V3",
                      kCoupFullBankTag[]="BTX_RC_COUP_FULL_BANK_V3",
                      kCoupMatXchgTag[]="BTX_RC_COUP_MAT_XCHG_V3",
                      kCoupMatXchgRoundsTag[]="BTX_RC_COUP_MAT_XCHG_ROUNDS_V3",
                      kCoupEpisodeTag[]="BTX_RC_COUP_EPISODE_V3";
static constexpr u32 kCoupMixPatterns = 2, kCoupExchangeRows = 128;

struct CoupP {
    u32 barriers, lobes, lobe_width, bank_pages, rows_per_lobe, pages_per_barrier_lobe,
        exchange_rounds;
};
static u32 coup_state_bytes(const CoupP& p) { return p.lobes*p.rows_per_lobe*p.lobe_width; }

static void cwle32(u8* p, u32 v){ for(int i=0;i<4;++i) p[i]=u8(v>>(8*i)); }
static void cwle64(u8* p, u64 v){ for(int i=0;i<8;++i) p[i]=u8(v>>(8*i)); }

// sha_tag_u32 == tagged_u32 from the included file (same byte stream); u32u32 is coupled-only.
static H256 coup_sha_tag_u32u32(const char* tag, const H256& a, u32 x, u32 y) {
    Sha256 s; s.write(tag, std::strlen(tag)); s.write(a.data(),32);
    u8 le[4]; cwle32(le,x); s.write(le,4); cwle32(le,y); s.write(le,4);
    H256 o; s.finalize(o.data()); return o;
}

// ---- counter-SHA XOF (ShaXof) -- host, mirrors the CPU reference exactly -------------------
struct CoupShaXof {
    H256 seed; u32 ctr=0, pos=32; u8 blk[32]{};
    explicit CoupShaXof(const H256& s): seed(s) {}
    void refill(){ Sha256 h; h.write(seed.data(),32); u8 le[4]; cwle32(le,ctr++); h.write(le,4);
                   h.finalize(blk); pos=0; }
    u32 next(){ if(pos+4>32) refill(); u32 v=u32(blk[pos])|(u32(blk[pos+1])<<8)|
               (u32(blk[pos+2])<<16)|(u32(blk[pos+3])<<24); pos+=4; return v; }
};

// full-bank page schedule + balanced permutation + exchange-round perms (all host: Fisher-Yates
// is a sequential swap chain, byte-frozen -- there is no parallel formulation that matches it).
static std::vector<u32> coup_select_pages(u32 barrier, u32 lobe, const H256& sigma, const CoupP& p) {
    const H256 perm_seed = tagged_u32(kCoupFullBankTag, sigma, p.bank_pages);
    CoupShaXof xof(perm_seed);
    std::vector<u32> perm(p.bank_pages); for(u32 i=0;i<p.bank_pages;++i) perm[i]=i;
    for (u32 i=p.bank_pages-1;i>0;--i){ u32 j=xof.next()%(i+1); std::swap(perm[i],perm[j]); }
    const u32 P = p.pages_per_barrier_lobe;
    const u64 base = (u64(barrier)*p.lobes + lobe)*u64(P);
    std::vector<u32> out(P);
    for (u32 k=0;k<P;++k) out[k]=perm[size_t((base+k)%p.bank_pages)];
    return out;
}
static std::vector<u32> coup_fy_perm(const H256& seed, u32 n) {
    CoupShaXof xof(seed);
    std::vector<u32> pi(n); for(u32 i=0;i<n;++i) pi[i]=i;
    for (u32 i=n-1;i>0;--i){ u32 j=xof.next()%(i+1); std::swap(pi[i],pi[j]); }
    return pi;
}

// ============================================================ coupled device kernels
// scatter: out[pi[i]] = in[i]  (apply_perm's tmp[pi[i]] = s[i])
static __global__ void k_coup_scatter64(const u64* __restrict__ in, u64* __restrict__ out,
                                 const u32* __restrict__ pi, u32 n)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[pi[i]] = in[i];
}
// butterfly mix, one STAGE per launch (stages are sequential; pairs within a stage are disjoint
// because i^mask / rotl(i^mask,3) are bijections). u64 two's-complement wrap by construction.
static __global__ void k_coup_mix_asc(u64* __restrict__ s, u32 mask, u32 stride, u32 n)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || (i & stride)) return;
    const u32 j = i | stride, pi = i ^ mask, pj = j ^ mask;
    const u64 a = s[pi], b = s[pj];
    s[pi] = a + b; s[pj] = a - b;
}
static __global__ void k_coup_mix_desc(u64* __restrict__ s, u32 mask, u32 stride, u32 n, u32 bits)
{
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || (i & stride)) return;
    const u32 j = i | stride;
    const u32 r = 3u % bits;
    const u32 xi = i ^ mask, xj = j ^ mask;
    const u32 pi = ((xi << r) | (xi >> (bits - r))) & (n - 1);
    const u32 pj = ((xj << r) | (xj >> (bits - r))) & (n - 1);
    const u64 a = s[pi], b = s[pj];
    s[pi] = a + b; s[pj] = b - a;
}
// exchange-round fold: XOR of all lanes (order-free), grid-stride + one atomicXor per block-lane
static __global__ void k_coup_fold(const u64* __restrict__ s, size_t n, unsigned long long* out)
{
    u64 acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        acc ^= s[i];
    if (acc) atomicXor(out, (unsigned long long)acc);
}
// exchange-round keystream: lane l's u64 = words (2l, 2l+1) of the sequential ShaXof word stream
// = LE u32 pair at byte offset 8*(l%4) of SHA256(seed || LE32(l/4)). One thread per XOF block.
static __global__ void k_coup_xor_ks(u64* __restrict__ s, const u8* __restrict__ seed, size_t n)
{
    const size_t blk = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t lane0 = blk * 4;
    if (lane0 >= n) return;
    u8 msg[36];
    #pragma unroll
    for (int k = 0; k < 32; ++k) msg[k] = seed[k];
    msg[32]=u8(blk); msg[33]=u8(blk>>8); msg[34]=u8(blk>>16); msg[35]=u8(blk>>24);
    u8 hash[32]; dsha256(msg, 36, hash);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const size_t l = lane0 + q;
        if (l >= n) return;
        const int off = 8 * q;
        const u64 lo = u64(hash[off])   | (u64(hash[off+1])<<8) |
                       (u64(hash[off+2])<<16) | (u64(hash[off+3])<<24);
        const u64 hi = u64(hash[off+4]) | (u64(hash[off+5])<<8) |
                       (u64(hash[off+6])<<16) | (u64(hash[off+7])<<24);
        s[l] ^= lo | (hi << 32);
    }
}
// non-affine Extract over the coupled state: tile t is extract_mx_tile_i64(prf, i=0, bj=t).
// Same ChaCha/M11/scale pipeline as k_extract_tiles but the raw lanes are TRUE i64 (the coupled
// mix ring exceeds int32 by construction), so the full dmix_from_i64 folding applies.
static __global__ void k_coup_extract64(const u8* __restrict__ prf, const i64* __restrict__ raw,
                                 i8* __restrict__ out, u32 n_tiles)
{
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tiles) return;
    const size_t base = (size_t)t * kBlockLen;

    u32 key[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
        key[k] = u32(prf[k*4]) | (u32(prf[k*4+1])<<8) | (u32(prf[k*4+2])<<16) | (u32(prf[k*4+3])<<24);

    i8 mu[kBlockLen];
    u32 filled = 0, remix = 0;
    while (filled < kBlockLen) {
        u8 ks[64];
        dchacha_block(key, t ^ kLaneMxBlock, u64(t), remix, ks);   // i=0: (i<<32)|bj == bj
        for (u32 b = 0; b < 64 && filled < kBlockLen; ++b) {
            for (int shift = 0; shift <= 4; shift += 4) {
                if (filled >= kBlockLen) break;
                const u8 nib = u8((ks[b] >> shift) & 0x0F);
                const u32 raw_u = dmix_from_i64(raw[base + filled]);
                const u8 mixed = u8((nib ^ u8((raw_u * 0x9E3779B9u) >> 28)) & 0x0F);
                if (dM11a[mixed]) mu[filled++] = dM11v[mixed];
            }
        }
        ++remix;
    }
    // Computed-schedule instantiation: the coupled leg's tile index can exceed the episode
    // W2C table's bj range, and this path is gated by its own golden (a4bb0cc4) -- keep it
    // byte-for-byte on the legacy schedule.
    const u8 e = u8((dsha256_scale_h0<false>(prf, 0, t) >> 24) & 0x3);
#pragma unroll
    for (u32 k = 0; k < kBlockLen; ++k) out[base + k] = i8(i32(mu[k]) * (i32{1} << e));
}

// ---- accumulating int8 GEMM (cuBLASLt, beta selectable) ------------------------------------
// row-major C(m,n) += A(m,k).B(k,n). Same descriptor mapping as gemm8 (B first, no transposes);
// beta=1 accumulates the 24-page schedule directly in int32 -- bit-safe because operands are
// extract outputs bounded |48|: per-page acc <= 48*48*W and the P-page sum <= 2304*W*P, guarded
// below against 2^31. Widening per page then summing in i64 would produce the SAME values.
static void coup_gemm8_acc(const i8* A, const i8* B, i32* C, int m, int n, int k, i32 beta)
{
#ifdef MATADOR_HAVE_CUTLASS
    if (RcGemmCutlassBackend()) {
        // Identical contract to gemm8(), so it takes the same backend. beta=1 accumulation is
        // exact int32 addition either way -- the page-schedule bound argued above is unchanged
        // by which kernel performs the multiply.
        const size_t mk = TuneScratchMark();
        i8* scratch = TuneScratchI8((size_t)k * n);
        const cudaError_t e =
            rcgemm::gemm_i8_nn_auto(A, B, C, m, n, k, beta, scratch, cudaStreamPerThread);
        TuneScratchRelease(mk);
        if (e != cudaSuccess) {
            printf("!! coup rc gemm (cutlass) FAILED: %s\n", cudaGetErrorString(e));
            exit(2);
        }
        return;
    }
#endif
#if MATADOR_USE_CUBLASLT
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasOperation_t nop = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &nop, sizeof(nop));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &nop, sizeof(nop));
    cublasLtMatrixLayout_t l0, l1, lc;
    cublasLtMatrixLayoutCreate(&l0, CUDA_R_8I,  n, k, n);
    cublasLtMatrixLayoutCreate(&l1, CUDA_R_8I,  k, m, k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, n, m, n);
    i32 alpha = 1;
    cublasStatus_t st = cublasLtMatmul(g_lt, op, &alpha, B, l0, A, l1, &beta, C, lc, C, lc,
                                       nullptr, g_ws, g_wsz, cudaStreamPerThread);
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("!! coup cublasLtMatmul FAILED status=%d\n", (int)st);
        exit(2);
    }
    cublasLtMatrixLayoutDestroy(l0); cublasLtMatrixLayoutDestroy(l1);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
#else
    printf("!! coup rc gemm: no backend compiled in\n");
    exit(2);
#endif
}

// ============================================================ template-keyed page residency
// Pages depend only on (brs, page) -- derive once per template and reuse across nonces when the
// whole dequantized bank fits VRAM (minus headroom). cudaMalloc'd OUTSIDE the g_dev pool: the
// pool is per-nonce scratch; the bank must survive across nonces.
struct CoupBankCache {
    bool valid = false, resident = false;
    H256 brs{};
    u32 npages = 0; size_t page_bytes = 0;
    std::vector<i8*> pages;
    void drop() {
        for (i8* p : pages) if (p) cudaFree(p);
        pages.clear(); valid = false; resident = false;
    }
};
static thread_local CoupBankCache g_coup_bank;

static H256 coup_page_seed(const H256& brs, u32 page) { return tagged_u32(kCoupBankTag, brs, page); }

static void coup_ensure_lt()
{
    static std::once_flag once;
#if MATADOR_USE_CUBLASLT
    std::call_once(once, [] { if (!g_lt) cublasLtCreate(&g_lt); });
    if (!g_ws) GCK(cudaMalloc(&g_ws, g_wsz));
#else
    (void)once;
#endif
    if (!g_shortfall) GCK(cudaMalloc(&g_shortfall, 4));
}

// Derive-or-fetch one page for the GEMM leg. Returns a device pointer valid until the next call
// (streamed path reuses d_scratch); resident pages are stable for the template's lifetime.
static const i8* coup_get_page(const H256& brs, u32 pid, u32 W, i8* d_scratch)
{
    if (g_coup_bank.resident && g_coup_bank.pages[pid]) return g_coup_bank.pages[pid];
    gpu_expand_mx_dequant(coup_page_seed(brs, pid), W, W, d_scratch);
    return d_scratch;
}

static void coup_prepare_bank(const H256& brs, const CoupP& p)
{
    if (g_coup_bank.valid && std::memcmp(g_coup_bank.brs.data(), brs.data(), 32) == 0 &&
        g_coup_bank.npages == p.bank_pages) return;
    g_coup_bank.drop();
    g_coup_bank.brs = brs; g_coup_bank.npages = p.bank_pages;
    g_coup_bank.page_bytes = (size_t)p.lobe_width * p.lobe_width;
    g_coup_bank.valid = true;

    size_t free_b = 0, total_b = 0;
    GCK(cudaMemGetInfo(&free_b, &total_b));
    const size_t need = g_coup_bank.page_bytes * p.bank_pages;
    const size_t headroom = size_t(4) << 30;   // GEMM workspace + per-nonce buffers + margin
    if (need + headroom <= free_b) {
        const double t0 = now_s();
        g_coup_bank.pages.assign(p.bank_pages, nullptr);
        for (u32 pg = 0; pg < p.bank_pages; ++pg) {
            GCK(cudaMalloc(&g_coup_bank.pages[pg], g_coup_bank.page_bytes));
            gpu_expand_mx_dequant(coup_page_seed(brs, pg), p.lobe_width, p.lobe_width,
                                  g_coup_bank.pages[pg]);
        }
        GCK(cudaDeviceSynchronize());
        g_coup_bank.resident = true;
        printf("[rc-coupled-gpu] bank RESIDENT: %u pages x %.1f MiB derived in %.0f ms\n",
               p.bank_pages, g_coup_bank.page_bytes / 1048576.0, (now_s() - t0) * 1000.0);
    } else {
        g_coup_bank.resident = false;
        printf("[rc-coupled-gpu] bank STREAMED: %.1f GiB dequant exceeds free VRAM %.1f GiB "
               "(pages re-derived per use)\n",
               need / 1073741824.0, free_b / 1073741824.0);
    }
}

// bank commitment root: DOUBLE SHA256(BankTag || page0 || ...) streamed through host SHA from
// device pages (resident: D2H the cache; streamed: derive one page at a time). Once per template.
static H256 coup_bank_root_gpu(const H256& brs, const CoupP& p)
{
    coup_ensure_lt();
    coup_prepare_bank(brs, p);
    const size_t page_bytes = (size_t)p.lobe_width * p.lobe_width;
    std::vector<i8> host_page(page_bytes);
    Sha256 h; h.write(kCoupBankTag, std::strlen(kCoupBankTag));
    const size_t mk = g_dev.mark();
    i8* d_scratch = g_coup_bank.resident ? nullptr : g_dev.d_i8(page_bytes);
    for (u32 pg = 0; pg < p.bank_pages; ++pg) {
        const i8* dp = coup_get_page(brs, pg, p.lobe_width, d_scratch);
        GCK(cudaMemcpy(host_page.data(), dp, page_bytes, cudaMemcpyDeviceToHost));
        h.write(host_page.data(), page_bytes);
    }
    g_dev.release_to(mk);
    H256 d1; h.finalize(d1.data());
    Sha256 b; b.write(d1.data(), 32);
    H256 d2; b.finalize(d2.data());
    return d2;
}

// ============================================================ per-nonce coupled episode on GPU
static H256 run_coupled_gpu(const H256& sigma, const H256& brs, const H256& bank_root,
                            const CoupP& p)
{
    coup_ensure_lt();
    coup_prepare_bank(brs, p);
    GCK(cudaMemset(g_shortfall, 0, 4));

    const u32 n = coup_state_bytes(p);
    const u32 M = p.rows_per_lobe, W = p.lobe_width, lobe_stride = M*W;
    if (n < 2 || (n & (n-1))) { printf("!! coup: n=%u not a power of two\n", n); exit(2); }
    u32 bits = 0; for (u32 t = n; t > 1; t >>= 1) ++bits;
    // int32 GEMM accumulation bound over the full page schedule (see coup_gemm8_acc)
    const double worst = 2304.0 * double(W) * double(p.pages_per_barrier_lobe);
    if (worst >= 2147483647.0) {
        printf("!! coup: P-page int32 accumulation would overflow (%.3g); needs per-page widen\n",
               worst);
        exit(2);
    }

    const size_t ep_mark = g_dev.mark();
    i8*  d_state = g_dev.d_i8(n);
    i32* d_acc32 = g_dev.d_i32(n);
    u64* d_acc64 = (u64*)g_dev.get((size_t)n * 8);
    u64* d_tmp64 = (u64*)g_dev.get((size_t)n * 8);
    u32* d_pi    = (u32*)g_dev.get((size_t)n * 4);
    u8*  d_seed  = g_dev.d_u8(32);
    u8*  d_prf   = g_dev.d_u8(32);
    unsigned long long* d_fold = (unsigned long long*)g_dev.get(8);
    i8*  d_page  = g_coup_bank.resident ? nullptr : g_dev.d_i8((size_t)W * W);
    i8*  d_tile  = g_dev.d_i8((size_t)W * W);

    // nonce-fresh lobe activation: first M rows of a WxW MX tile per lobe.
    // Loop var must NOT be named `e`: GCK() declares its own `cudaError_t e`, and a macro
    // argument referencing `e` would self-initialize from the uninitialized inner declaration
    // (caught by the first pc build's -Wuninitialized echo).
    for (u32 lb = 0; lb < p.lobes; ++lb) {
        gpu_expand_mx_dequant(tagged_u32(kCoupLobeTag, sigma, lb), W, W, d_tile);
        GCK(cudaMemcpy(d_state + (size_t)lb*lobe_stride, d_tile, lobe_stride,
                       cudaMemcpyDeviceToDevice));
    }

    const int T = 256;
    const u32 gN = (n + T - 1) / T;
    std::vector<H256> roots(p.barriers);
    std::vector<i8> host_state(n);

    for (u32 b = 0; b < p.barriers; ++b) {
        // C3.a: per lobe, accumulate the P-page GEMM schedule in int32, then widen once.
        for (u32 e = 0; e < p.lobes; ++e) {
            const auto ids = coup_select_pages(b, e, sigma, p);
            i32* C = d_acc32 + (size_t)e * lobe_stride;
            for (u32 k = 0; k < ids.size(); ++k) {
                const i8* dp = coup_get_page(g_coup_bank.brs, ids[k], W, d_page);
                coup_gemm8_acc(d_state + (size_t)e*lobe_stride, dp, C, (int)M, (int)W, (int)W,
                               k == 0 ? 0 : 1);
            }
        }
        k_widen<<<gN, T>>>(d_acc32, (i64*)d_acc64, n); GCK(cudaGetLastError());

        // C3.b balanced permutation (host FY, device scatter)
        {
            const auto pi = coup_fy_perm(tagged_u32(kCoupPermTag, sigma, b), n);
            GCK(cudaMemcpy(d_pi, pi.data(), (size_t)n*4, cudaMemcpyHostToDevice));
            k_coup_scatter64<<<gN, T>>>(d_acc64, d_tmp64, d_pi, n); GCK(cudaGetLastError());
            std::swap(d_acc64, d_tmp64);
        }
        // C3.c all-to-all butterfly mix (mask from the exchange tag's XOF)
        {
            const H256 mix_seed = coup_sha_tag_u32u32(kCoupMatXchgTag, sigma, b, kCoupExchangeRows);
            CoupShaXof xof(mix_seed);
            const u32 mask = xof.next() & (n - 1);
            if (b % kCoupMixPatterns == 0) {
                for (u32 stride = 1; stride < n; stride <<= 1) {
                    k_coup_mix_asc<<<gN, T>>>(d_acc64, mask, stride, n); GCK(cudaGetLastError());
                }
            } else {
                for (int st = (int)bits - 1; st >= 0; --st) {
                    k_coup_mix_desc<<<gN, T>>>(d_acc64, mask, 1u << (u32)st, n, bits);
                    GCK(cudaGetLastError());
                }
            }
        }
        // +V3 material-exchange rounds: fold -> seed -> keystream XOR -> FY lane permutation.
        // The fold is data-dependent, so each round is a genuine GPU->host->GPU round trip.
        for (u32 r = 0; r < p.exchange_rounds; ++r) {
            GCK(cudaMemset(d_fold, 0, 8));
            k_coup_fold<<<256, T>>>(d_acc64, n, d_fold); GCK(cudaGetLastError());
            u64 fold = 0;
            GCK(cudaMemcpy(&fold, d_fold, 8, cudaMemcpyDeviceToHost));
            std::vector<u8> pre;
            pre.insert(pre.end(), kCoupMatXchgRoundsTag,
                       kCoupMatXchgRoundsTag + std::strlen(kCoupMatXchgRoundsTag));
            u8 tail[48]; std::memcpy(tail, sigma.data(), 32);
            cwle32(tail+32, b); cwle32(tail+36, r); cwle64(tail+40, fold);
            pre.insert(pre.end(), tail, tail+48);
            const H256 seed = sha256d(pre.data(), pre.size());
            GCK(cudaMemcpy(d_seed, seed.data(), 32, cudaMemcpyHostToDevice));
            const size_t nblk = ((size_t)n + 3) / 4;
            k_coup_xor_ks<<<(u32)((nblk + T - 1) / T), T>>>(d_acc64, d_seed, n);
            GCK(cudaGetLastError());
            const auto pi = coup_fy_perm(seed, n);       // FRESH XOF, matching xof2
            GCK(cudaMemcpy(d_pi, pi.data(), (size_t)n*4, cudaMemcpyHostToDevice));
            k_coup_scatter64<<<gN, T>>>(d_acc64, d_tmp64, d_pi, n); GCK(cudaGetLastError());
            std::swap(d_acc64, d_tmp64);
        }
        // C3.d Extract (i64 raw -> i8 state), C3.e feed-forward + barrier root on host
        {
            const H256 prf = derive_prf_key(coup_sha_tag_u32u32(kCoupExtractTag, sigma, b, 0));
            GCK(cudaMemcpy(d_prf, prf.data(), 32, cudaMemcpyHostToDevice));
            const u32 n_tiles = n / kBlockLen;
            k_coup_extract64<<<(n_tiles + T - 1) / T, T>>>(d_prf, (const i64*)d_acc64, d_state,
                                                           n_tiles);
            GCK(cudaGetLastError());
        }
        GCK(cudaMemcpy(host_state.data(), d_state, n, cudaMemcpyDeviceToHost));
        std::vector<u8> buf;
        buf.insert(buf.end(), kCoupBarrierTag, kCoupBarrierTag + std::strlen(kCoupBarrierTag));
        u8 le[4]; cwle32(le, b); buf.insert(buf.end(), le, le+4);
        buf.insert(buf.end(), (const u8*)host_state.data(), (const u8*)host_state.data() + n);
        roots[b] = sha256d(buf.data(), buf.size());
    }

    // expansion shortfall check (set by gpu_expand_mx_dequant's device-side guard)
    u32 shortfall = 0;
    GCK(cudaMemcpy(&shortfall, g_shortfall, 4, cudaMemcpyDeviceToHost));
    if (shortfall) { printf("!! coup: MX expansion shortfall -- margin breached\n"); exit(2); }

    g_dev.release_to(ep_mark);

    std::vector<u8> buf;
    buf.insert(buf.end(), kCoupEpisodeTag, kCoupEpisodeTag + std::strlen(kCoupEpisodeTag));
    buf.insert(buf.end(), bank_root.begin(), bank_root.end());
    for (const auto& rt : roots) buf.insert(buf.end(), rt.begin(), rt.end());
    return sha256d(buf.data(), buf.size());
}

// ============================================================ public entry points
#include <span.h>
#include <uint256.h>
#include <matmul/matmul_v4_rc_coupled.h>

namespace matmul::v4::rc {

static CoupP CoupFrom(const CoupParamsV3& p)
{
    return CoupP{p.barriers, p.lobes, p.lobe_width, p.bank_pages, p.rows_per_lobe,
                 p.pages_per_barrier_lobe, p.exchange_rounds};
}
static H256 h256_of(const uint256& v) { H256 o{}; std::memcpy(o.data(), v.data(), 32); return o; }
static uint256 u256_of(const H256& v) { return uint256{Span<const unsigned char>{v.data(), 32}}; }

bool RCCoupledGpuAvailable()
{
    int nd = 0;
    return cudaGetDeviceCount(&nd) == cudaSuccess && nd > 0;
}

// brs = SHA256(BankTag || tmpl_hash || LE32(height)) -- bank_root_seed in the CPU reference.
static H256 coup_brs(const uint256& bank_template_hash, uint32_t height)
{
    return tagged_u32(kCoupBankTag, h256_of(bank_template_hash), height);
}

uint256 ComputeCoupledBankRootGPU(const uint256& bank_template_hash, uint32_t height,
                                  const CoupParamsV3& params)
{
    return u256_of(coup_bank_root_gpu(coup_brs(bank_template_hash, height), CoupFrom(params)));
}

// Byte-exact to ComputeCoupledDigestV3WithBank (the CPU oracle) -- same digest, GPU throughput.
// bank_root comes from ComputeCoupledBankRootGPU (or the CPU BuildCoupledBankV3), once per
// template; pages are template-cached / re-derived on device, never uploaded.
uint256 ComputeCoupledDigestV3GPU(const uint256& sigma, const uint256& bank_template_hash,
                                  uint32_t height, const uint256& bank_root,
                                  const CoupParamsV3& params)
{
    const H256 d = run_coupled_gpu(h256_of(sigma), coup_brs(bank_template_hash, height),
                                   h256_of(bank_root), CoupFrom(params));
    return u256_of(d);
}

}  // namespace matmul::v4::rc
