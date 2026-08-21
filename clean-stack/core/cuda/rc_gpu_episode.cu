#include <cstdio>
// rc_gpu_solver.cu -- Phases C-E of our v4.5 ENC_RC solver: the whole episode on GPU.
//
// Phase A (rc_cpu_solver.cpp) gave a byte-exact CPU oracle reproducing his frozen golden
// (v4.6 fused-FFN: 5b1bff3c; the pre-fused b339d0ff is dead upstream). Phase B
// (rc_gpu_extract.cu) moved the MX Extract to GPU at 1041x, bit-exact.
// v4.6 RE-SYNC 2026-07-23: Phase 2 is the fused 2-layer FFN (H=Extract(X·W_up),
// X_out=Extract(H·W_down+X)); bwd/wgrad GONE; stream = Z ‖ X[1..L]. Mode 3 runs the
// DATACENTER profile-2 shape (Config W sigma-shared K/V + single W pair, row-block X0).
// This file moves the rest -- both phase-1 GEMMs, all three phase-2 GEMMs per layer, and the
// Extracts between them -- and gates the whole thing on the same golden.
//
// KEY SIMPLIFICATION (why phase 1 is just GEMM->Extract->GEMM->Extract):
// his phase-1 walks 32-wide blocks with a pending buffer, committing seg_row into acc_Z at
// kRCSegLen boundaries. That grouping only changes the ORDER of int64 additions, and integer
// addition is associative, so a straight int64 reduction over t is bit-identical. The sequential
// pending/segment machinery is a memory-locality device on CPU, not a semantic constraint.
//
// Bounds: forward/dgrad accumulate in int64 throughout (his oracle widens before Extract anyway,
// "avoids a latent int32-trunc fork"), so we never depend on the int32 range.
//
// build: nvcc -O3 -arch=sm_XX rc_gpu_solver.cu -o rc_gpu_solver
// usage: rc_gpu_solver            (toy dims, golden-gated)

#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <thread>
#include <climits>
#include <utility>
#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
// MATADOR_USE_CUBLASLT: compile the cuBLASLt GEMM backend in at all. Linking it statically costs
// ~530 MB of the shipped binary (2,785 precompiled kernels over 11 archs + ~293 MB of dispatch
// tables), for a call surface of exactly one int8 GEMM shape. With CUTLASS available we compile
// cuBLASLt out entirely and that weight goes with it. Force it back on with -DMATADOR_USE_CUBLASLT=1
// to get both backends in one binary for A/B work.
#ifndef MATADOR_USE_CUBLASLT
#  ifdef MATADOR_HAVE_CUTLASS
#    define MATADOR_USE_CUBLASLT 0
#  else
#    define MATADOR_USE_CUBLASLT 1
#  endif
#endif
#if MATADOR_USE_CUBLASLT
#include <cublasLt.h>
#endif
#ifdef MATADOR_HAVE_CUTLASS
#include "rc_gemm_i8_cutlass.cuh"   // the GEMM backend (see gemm8 below)
#endif
#define RC_GPU_SOLVER 1

#include <cstdint>
#include <cstring>
#include <array>
#include <set>
#include <string>
#include <algorithm>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

using u8 = uint8_t; using u32 = uint32_t; using u64 = uint64_t;
using i8 = int8_t;  using i32 = int32_t; using i64 = int64_t;

// ============================================================ SHA-256
struct Sha256 {
    u32 s[8]{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 buf[64]{}; size_t len = 0; u64 bits = 0;
    static u32 ror(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const u8* p) {
        static const u32 K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        u32 w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (u32(p[i*4]) << 24) | (u32(p[i*4+1]) << 16) | (u32(p[i*4+2]) << 8) | u32(p[i*4+3]);
        for (int i = 16; i < 64; ++i) {
            u32 s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3);
            u32 s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        u32 a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        for (int i = 0; i < 64; ++i) {
            u32 S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
            u32 ch = (e & f) ^ (~e & g);
            u32 t1 = h + S1 + ch + K[i] + w[i];
            u32 S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
            u32 mj = (a & b) ^ (a & c) ^ (b & c);
            u32 t2 = S0 + mj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
    }
    void write(const void* data, size_t n) {
        const u8* p = (const u8*)data; bits += u64(n) * 8;
        while (n) {
            size_t take = 64 - len; if (take > n) take = n;
            std::memcpy(buf + len, p, take); len += take; p += take; n -= take;
            if (len == 64) { block(buf); len = 0; }
        }
    }
    void finalize(u8 out[32]) {
        u64 b = bits; u8 pad = 0x80; write(&pad, 1);
        u8 z = 0; while (len != 56) write(&z, 1);
        u8 be[8]; for (int i = 0; i < 8; ++i) be[i] = u8(b >> (56 - 8*i));
        write(be, 8);
        for (int i = 0; i < 8; ++i) {
            out[i*4]   = u8(s[i] >> 24); out[i*4+1] = u8(s[i] >> 16);
            out[i*4+2] = u8(s[i] >> 8);  out[i*4+3] = u8(s[i]);
        }
    }
};
using H256 = std::array<u8, 32>;
// uint256::GetHex() prints .data() REVERSED (little-endian display). Hashing always uses
// .data() order, so conversions must reverse -- getting this backwards silently poisons every
// derived value downstream (it cost the first run of this file).
static std::string hex(const H256& h) {
    static const char* d = "0123456789abcdef"; std::string s;
    for (int i = 31; i >= 0; --i) { s += d[h[i] >> 4]; s += d[h[i] & 15]; }
    return s;
}

// ============================================================ ChaCha20 (RFC8439)
// Core's Nonce96{first(u32), second(u64)} maps to state words 13,14,15; word 12 is the counter.
struct ChaCha20 {
    u32 in[16]{};
    explicit ChaCha20(const u8 key[32]) {
        static const char sigma_c[] = "expand 32-byte k";
        for (int i = 0; i < 4; ++i)
            in[i] = u32(u8(sigma_c[i*4])) | (u32(u8(sigma_c[i*4+1])) << 8) |
                    (u32(u8(sigma_c[i*4+2])) << 16) | (u32(u8(sigma_c[i*4+3])) << 24);
        for (int i = 0; i < 8; ++i)
            in[4+i] = u32(key[i*4]) | (u32(key[i*4+1]) << 8) |
                      (u32(key[i*4+2]) << 16) | (u32(key[i*4+3]) << 24);
    }
    void seek(u32 nonce_first, u64 nonce_second, u32 counter) {
        in[12] = counter;
        in[13] = nonce_first;
        in[14] = u32(nonce_second);
        in[15] = u32(nonce_second >> 32);
    }
    static u32 rotl(u32 x, int n) { return (x << n) | (x >> (32 - n)); }
    void keystream(u8* out, size_t n) {
        while (n) {
            u32 x[16]; std::memcpy(x, in, sizeof(x));
            for (int r = 0; r < 10; ++r) {
                auto qr = [&](int a, int b, int c, int d) {
                    x[a]+=x[b]; x[d]=rotl(x[d]^x[a],16);
                    x[c]+=x[d]; x[b]=rotl(x[b]^x[c],12);
                    x[a]+=x[b]; x[d]=rotl(x[d]^x[a], 8);
                    x[c]+=x[d]; x[b]=rotl(x[b]^x[c], 7);
                };
                qr(0,4,8,12); qr(1,5,9,13); qr(2,6,10,14); qr(3,7,11,15);
                qr(0,5,10,15); qr(1,6,11,12); qr(2,7,8,13); qr(3,4,9,14);
            }
            u8 blk[64];
            for (int i = 0; i < 16; ++i) {
                u32 v = x[i] + in[i];
                blk[i*4] = u8(v); blk[i*4+1] = u8(v >> 8);
                blk[i*4+2] = u8(v >> 16); blk[i*4+3] = u8(v >> 24);
            }
            size_t take = n < 64 ? n : 64;
            std::memcpy(out, blk, take); out += take; n -= take;
            ++in[12];
        }
    }
};

// ============================================================ M11 mantissa table
// FP4 E2M1 magnitude by (exp,man): exp0->{0,.5} exp1->{1,1.5} exp2->{2,3} exp3->{4,6}.
// Half-integer codes and negative zero are holes => exactly 11 of 16 accepted (rejects 1,3,8,9,11).
struct M11 {
    i8 value[16]{}; bool accepted[16]{};
    constexpr M11() {
        for (u8 nib = 0; nib < 16; ++nib) {
            const u8 sign = (nib >> 3) & 1, exp = (nib >> 1) & 3, man = nib & 1;
            int mag = 0; bool integer = true;
            switch (exp) {
            case 0: mag = 0; integer = (man == 0); break;
            case 1: mag = 1; integer = (man == 0); break;
            case 2: mag = (man == 0) ? 2 : 3; break;
            case 3: mag = (man == 0) ? 4 : 6; break;
            }
            if (!integer || (sign && mag == 0)) { accepted[nib] = false; value[nib] = 0; continue; }
            accepted[nib] = true; value[nib] = i8(sign ? -mag : mag);
        }
    }
};
static constexpr M11 kM11{};

// ============================================================ MatExpand MX primitives (LT-derived)
static constexpr char kPrfTag[]     = "BTX_MATEXPAND_MXPRF_V44LT";
static constexpr char kMxScaleTag[] = "BTX_MATEXPAND_MXSCALE_V44LT";
static constexpr u32  kLaneMxBlock  = 0x4D58424Cu;   // 'MXBL'
static constexpr u32  kMxBlockLen   = 32;

static H256 derive_prf_key(const H256& seed) {
    Sha256 h; h.write(kPrfTag, sizeof(kPrfTag) - 1); h.write(seed.data(), 32);
    H256 o; h.finalize(o.data()); return o;
}
static u8 derive_mx_scale(const H256& prf, u32 i, u32 bj) {
    Sha256 h; h.write(kMxScaleTag, sizeof(kMxScaleTag) - 1); h.write(prf.data(), 32);
    u8 il[4], bl[4];
    for (int k = 0; k < 4; ++k) { il[k] = u8(i >> (8*k)); bl[k] = u8(bj >> (8*k)); }
    h.write(il, 4); h.write(bl, 4);
    H256 o; h.finalize(o.data()); return u8(o[0] & 0x3);
}
static u32 mix_bits_from_i64(i64 y) {
    if (y >= INT32_MIN && y <= INT32_MAX) return u32(i32(y));
    const u64 u = u64(y);
    return u32(u) ^ u32(u >> 32);
}
// One ChaCha20 stream per (i,bj) tile; each accepted nibble XOR-mixed with its cell's raw,
// so Extract depends on the data (non-XOF).
static void extract_tile_mantissas(const H256& prf, u32 i, u32 bj, const i64 raw64[32], i8 mu[32]) {
    u32 filled = 0, remix = 0;
    while (filled < kMxBlockLen) {
        ChaCha20 cc(prf.data());
        cc.seek(bj ^ kLaneMxBlock, (u64(i) << 32) | u64(bj), remix);
        u8 ks[64]; cc.keystream(ks, sizeof(ks));
        for (size_t b = 0; b < sizeof(ks) && filled < kMxBlockLen; ++b) {
            for (int shift : {0, 4}) {
                if (filled >= kMxBlockLen) break;
                const u8 nibble = u8((ks[b] >> shift) & 0x0F);
                const u32 raw_u = mix_bits_from_i64(raw64[filled]);
                const u8 mixed = u8((nibble ^ u8((raw_u * 0x9E3779B9u) >> 28)) & 0x0F);
                if (kM11.accepted[mixed]) mu[filled++] = kM11.value[mixed];
            }
        }
        ++remix;
    }
}
static void extract_mx_tile_i64(const H256& prf, u32 i, u32 bj, const i64 raw64[32], i8 out[32]) {
    i8 mu[32];
    extract_tile_mantissas(prf, i, bj, raw64, mu);
    const u8 e = derive_mx_scale(prf, i, bj);
    for (u32 t = 0; t < kMxBlockLen; ++t) out[t] = i8(i32(mu[t]) * (i32{1} << e));
}


// ============================================================ MX operand expansion
// ExpandMxDequantInt8(seed,rows,cols) = DequantMxPacked(ExpandMxPacked(seed,rows,cols,RowBlock)):
//   mu     = counter-mode SHA256 XOF, domain 'm', LOW nibble of each byte then HIGH, M11-rejected
//   scales = counter-mode SHA256 XOF, domain 'e', 2 bits per code from the LSB up, no rejection
//   out    = mu << scale, with RowBlock scale index i*(cols/32) + j/32
// Both streams key off SeedBytesLE(seed) -- the REVERSE of .data() (same endianness trap again).
static constexpr u8 kMantissaDomain = 0x6D;   // 'm'
static constexpr u8 kScaleDomain    = 0x65;   // 'e'

static void seed_bytes_le(const H256& seed, u8 out[32]) {
    for (int i = 0; i < 32; ++i) out[i] = seed[31 - i];
}
static void expand_mantissa_stream(const H256& seed, size_t count, i8* out) {
    u8 sb[32]; seed_bytes_le(seed, sb);
    size_t filled = 0; u64 block = 0;
    while (filled < count) {
        Sha256 h; h.write(sb, 32); h.write(&kMantissaDomain, 1);
        u8 ble[8]; for (int k = 0; k < 8; ++k) ble[k] = u8(block >> (8*k));
        h.write(ble, 8);
        u8 hash[32]; h.finalize(hash);
        for (size_t i = 0; i < 32 && filled < count; ++i) {
            const u8 nibs[2] = { u8(hash[i] & 0x0F), u8((hash[i] >> 4) & 0x0F) };
            for (u8 nib : nibs) {
                if (kM11.accepted[nib]) { out[filled++] = kM11.value[nib]; if (filled == count) break; }
            }
        }
        ++block;
    }
}
static void expand_scale_stream(const H256& seed, size_t count, u8* out) {
    u8 sb[32]; seed_bytes_le(seed, sb);
    size_t filled = 0; u64 block = 0;
    while (filled < count) {
        Sha256 h; h.write(sb, 32); h.write(&kScaleDomain, 1);
        u8 ble[8]; for (int k = 0; k < 8; ++k) ble[k] = u8(block >> (8*k));
        h.write(ble, 8);
        u8 hash[32]; h.finalize(hash);
        for (size_t i = 0; i < 32 && filled < count; ++i)
            for (int shift = 0; shift < 8 && filled < count; shift += 2)
                out[filled++] = u8((hash[i] >> shift) & 0x03);
        ++block;
    }
}
// RowBlock axis (the consensus oracle layout for Q/K and Phase-2 X).
static std::vector<i8> expand_mx_dequant_i8(const H256& seed, u32 rows, u32 cols) {
    const size_t n = size_t(rows) * cols;
    std::vector<i8> mu(n);
    expand_mantissa_stream(seed, n, mu.data());
    const u32 nblk = cols / kMxBlockLen;
    std::vector<u8> scales(size_t(rows) * nblk);
    expand_scale_stream(seed, scales.size(), scales.data());
    std::vector<i8> out(n);
    for (u32 i = 0; i < rows; ++i)
        for (u32 j = 0; j < cols; ++j) {
            const size_t idx = size_t(i) * cols + j;
            out[idx] = i8(i32(mu[idx]) * (i32{1} << scales[size_t(i) * nblk + (j / kMxBlockLen)]));
        }
    return out;
}

// ============================================================ tagged-seed helpers (RC)
static H256 sha256_tagged(const char* tag, const u8* d, size_t n) {
    Sha256 h; h.write(tag, std::strlen(tag)); if (n) h.write(d, n);
    H256 o; h.finalize(o.data()); return o;
}
static H256 derive_operand_seed(const H256& seed_r, const char* tag) {
    return sha256_tagged(tag, seed_r.data(), 32);
}


// ============================================================ episode (toy dims)
struct Params {
    u32 rounds=1,d_head=32,n_q=32,n_ctx=64,L_lyr=2,d_model=32,d_ff=128,b_seq=32,T_leaf=64;
    // Datacenter (profile 2) semantics -- both false for the toy/base golden.
    bool share_ep=false;    // Config W: K/V + one (W_up,W_down) pair sigma-derived episode-wide
    bool rowblock_x0=false; // X0 expanded as independent 32-row blocks (X0_ROW_BLOCK tag)
};
static constexpr u32 kSegLen = 32768;          // kRCSegLen
static constexpr u8  kLeafTag=0x00, kNodeTag=0x01, kPadLeafTag=0x02;
static constexpr char kRoundTag[]="BTX_RC_ROUND_V1", kEpisodeTag[]="BTX_RC_EPISODE_V1",
                      kPadTag[]="BTX_RC_PAD";

static H256 sha256d(const u8* d, size_t n) {
    Sha256 a; a.write(d,n); H256 x; a.finalize(x.data());
    Sha256 b; b.write(x.data(),32); H256 y; b.finalize(y.data()); return y;
}
static H256 hash_node(const H256& l, const H256& r) {
    u8 buf[65]; buf[0]=kNodeTag; std::memcpy(buf+1,l.data(),32); std::memcpy(buf+33,r.data(),32);
    return sha256d(buf,sizeof(buf));
}
static H256 hash_leaf(const u8* b, u32 t_leaf) {
    std::vector<u8> pre; pre.push_back(kLeafTag); pre.insert(pre.end(), b, b+t_leaf);
    return sha256d(pre.data(), pre.size());
}
static H256 pad_leaf_hash() {
    std::vector<u8> pre; pre.push_back(kPadLeafTag);
    pre.insert(pre.end(), (const u8*)kPadTag, (const u8*)kPadTag + sizeof(kPadTag)-1);
    return sha256d(pre.data(), pre.size());
}
// Absorb int8 stream into T_leaf-sized leaves; zero-pad tail; pad to pow2 with pad-leaf; fold.
struct Merkle {
    u32 t_leaf; std::vector<u8> partial; std::vector<H256> leaves;
    explicit Merkle(u32 t): t_leaf(t) {}
    void absorb(const std::vector<i8>& v) {
        for (i8 b : v) {
            partial.push_back(u8(b));
            if (partial.size() == t_leaf) { leaves.push_back(hash_leaf(partial.data(), t_leaf)); partial.clear(); }
        }
    }
    H256 finalize_root() {
        if (!partial.empty()) { partial.resize(t_leaf, 0); leaves.push_back(hash_leaf(partial.data(), t_leaf)); partial.clear(); }
        if (leaves.empty()) leaves.push_back(pad_leaf_hash());
        size_t target = 1; while (target < leaves.size()) target <<= 1;
        while (leaves.size() < target) leaves.push_back(pad_leaf_hash());
        std::vector<H256> lvl = leaves;
        while (lvl.size() > 1) {
            std::vector<H256> up; up.reserve(lvl.size()/2);
            for (size_t i = 0; i < lvl.size(); i += 2) up.push_back(hash_node(lvl[i], lvl[i+1]));
            lvl.swap(up);
        }
        return lvl[0];
    }
};
static void extract_mx_matrix_i64(const H256& prf, const i64* Y, u32 rows, u32 cols, i8* out) {
    const u32 nblk = cols / kMxBlockLen;
    for (u32 i = 0; i < rows; ++i)
        for (u32 bj = 0; bj < nblk; ++bj) {
            const size_t base = size_t(i)*cols + size_t(bj)*kMxBlockLen;
            extract_mx_tile_i64(prf, i, bj, Y + base, out + base);
        }
}
// ExactGemmS8S8: C(rows x cols) = L(rows x inner) * R(inner x cols)
static std::vector<i32> gemm_s8s8(const std::vector<i8>& L, const std::vector<i8>& R,
                                  u32 rows, u32 inner, u32 cols) {
    std::vector<i32> o(size_t(rows)*cols, 0);
    for (u32 i = 0; i < rows; ++i)
        for (u32 k = 0; k < inner; ++k) {
            const i32 a = L[size_t(i)*inner + k]; if (!a) continue;
            for (u32 j = 0; j < cols; ++j) o[size_t(i)*cols + j] += a * i32(R[size_t(k)*cols + j]);
        }
    return o;
}
static std::vector<i8> transpose_s8(const std::vector<i8>& M, u32 rows, u32 cols) {
    std::vector<i8> T(size_t(cols)*rows);
    for (u32 r = 0; r < rows; ++r) for (u32 c = 0; c < cols; ++c)
        T[size_t(c)*rows + r] = M[size_t(r)*cols + c];
    return T;
}
// int8·int8 -> int64 GEMM, natural layouts: out[m×n] = A[m×k]·B[k×n] (byte-identical to his
// K-panel FusedExactGemmInt64; panels exist only for FP32 accelerators).
static std::vector<i64> gemm_s8s8_i64(const std::vector<i8>& A, const std::vector<i8>& B,
                                      u32 m, u32 k, u32 n) {
    std::vector<i64> o(size_t(m)*n, 0);
    for (u32 i = 0; i < m; ++i)
        for (u32 t = 0; t < k; ++t) {
            const i64 a = A[size_t(i)*k + t]; if (!a) continue;
            for (u32 j = 0; j < n; ++j) o[size_t(i)*n + j] += a * i64(B[size_t(t)*n + j]);
        }
    return o;
}
// One fused 2-layer FFN (v4.6): H = Extract(X·W_up) internal; X_out = Extract(H·W_down + X).
static std::vector<i8> fused_ffn_layer(const std::vector<i8>& X, const std::vector<i8>& W_up,
                                       const std::vector<i8>& W_dn, const H256& prf_up,
                                       const H256& prf_dn, u32 b_seq, u32 d_model, u32 d_ff) {
    auto h64 = gemm_s8s8_i64(X, W_up, b_seq, d_model, d_ff);
    std::vector<i8> H(h64.size());
    extract_mx_matrix_i64(prf_up, h64.data(), b_seq, d_ff, H.data());
    h64 = std::vector<i64>();
    auto y64 = gemm_s8s8_i64(H, W_dn, b_seq, d_ff, d_model);
    for (u32 i = 0; i < b_seq; ++i)
        for (u32 j = 0; j < d_model; ++j)
            y64[size_t(i)*d_model + j] += i64(X[size_t(i)*d_model + j]);
    std::vector<i8> out(y64.size());
    extract_mx_matrix_i64(prf_dn, y64.data(), b_seq, d_model, out.data());
    return out;
}
static H256 tagged_u32(const char* tag, const H256& a, u32 le32) {
    u8 buf[36]; std::memcpy(buf, a.data(), 32);
    for (int k = 0; k < 4; ++k) buf[32+k] = u8(le32 >> (8*k));
    return sha256_tagged(tag, buf, 36);
}

static H256 run_episode(const H256& sigma, const Params& p) {
    std::vector<H256> roots(p.rounds);
    H256 seed_r = tagged_u32(kRoundTag, sigma, 0);
    for (u32 r = 0; r < p.rounds; ++r) {
        if (r > 0) seed_r = tagged_u32(kRoundTag, roots[r-1], r);

        // ---- Phase 1: associative recall ----
        // Q always per-round; K/V episode-wide (sigma) under Config W, else per-round.
        const H256& kv_seed = p.share_ep ? sigma : seed_r;
        const H256 prf_S = derive_prf_key(derive_operand_seed(seed_r, "BTX_RC_PRF_S_V1"));
        const H256 prf_Z = derive_prf_key(derive_operand_seed(seed_r, "BTX_RC_PRF_Z_V1"));
        const auto Q = expand_mx_dequant_i8(derive_operand_seed(seed_r, "BTX_RC_Q_V1"),     p.n_q,   p.d_head);
        const auto K = expand_mx_dequant_i8(derive_operand_seed(kv_seed, "BTX_RC_KV_K_V1"), p.n_ctx, p.d_head);
        const auto V = expand_mx_dequant_i8(derive_operand_seed(kv_seed, "BTX_RC_KV_V_V1"), p.n_ctx, p.d_head);
        std::vector<i8> Z(size_t(p.n_q)*p.d_head, 0);
        for (u32 i = 0; i < p.n_q; ++i) {
            i64 pending[kMxBlockLen]; u32 fill=0, bj=0, t0=0, cur_seg=0;
            std::vector<i64> seg_row(p.d_head, 0), acc_Z(p.d_head, 0);
            for (u32 t = 0; t < p.n_ctx; ++t) {
                i64 acc = 0;
                for (u32 d = 0; d < p.d_head; ++d)
                    acc += i64(Q[size_t(i)*p.d_head + d]) * i64(K[size_t(t)*p.d_head + d]);
                pending[fill++] = acc;
                if (fill == kMxBlockLen) {
                    i8 S_tile[kMxBlockLen];
                    extract_mx_tile_i64(prf_S, i, bj, pending, S_tile);
                    for (u32 off = 0; off < kMxBlockLen; ++off) {
                        const u32 tt = t0 + off, seg = tt / kSegLen;
                        if (seg != cur_seg) {
                            for (u32 d = 0; d < p.d_head; ++d) acc_Z[d] += seg_row[d];
                            std::fill(seg_row.begin(), seg_row.end(), 0);
                            cur_seg = seg;
                        }
                        for (u32 d = 0; d < p.d_head; ++d)
                            seg_row[d] += i64(S_tile[off]) * i64(V[size_t(tt)*p.d_head + d]);
                    }
                    fill = 0; ++bj; t0 += kMxBlockLen;
                }
            }
            for (u32 d = 0; d < p.d_head; ++d) acc_Z[d] += seg_row[d];
            const u32 nblk = p.d_head / kMxBlockLen;
            for (u32 b = 0; b < nblk; ++b)
                extract_mx_tile_i64(prf_Z, i, b, acc_Z.data() + b*kMxBlockLen,
                                    Z.data() + size_t(i)*p.d_head + b*kMxBlockLen);
        }

        // ---- Phase 2: fused-FFN forward (bwd/wgrad GONE in v4.6) ----
        std::vector<std::vector<i8>> X(p.L_lyr+1);
        std::vector<std::vector<i8>> Wup_l, Wdn_l;
        std::vector<i8> Wup_s, Wdn_s;
        std::vector<H256> prf_up(p.L_lyr), prf_dn(p.L_lyr);
        const H256 seed_x0 = derive_operand_seed(seed_r, "BTX_RC_X0_V1");
        if (p.rowblock_x0) {
            X[0].resize(size_t(p.b_seq)*p.d_model);
            for (u32 b = 0; b < p.b_seq/32; ++b) {
                const H256 bs = tagged_u32("BTX_RC_X0_ROW_BLOCK_V1", seed_x0, b);
                const auto blk = expand_mx_dequant_i8(bs, 32, p.d_model);
                std::copy(blk.begin(), blk.end(), X[0].begin() + size_t(b)*32*p.d_model);
            }
        } else {
            X[0] = expand_mx_dequant_i8(seed_x0, p.b_seq, p.d_model);
        }
        if (p.share_ep) {
            Wup_s = expand_mx_dequant_i8(derive_operand_seed(sigma, "BTX_RC_WUP_V1"), p.d_model, p.d_ff);
            Wdn_s = expand_mx_dequant_i8(derive_operand_seed(sigma, "BTX_RC_WDN_V1"), p.d_ff, p.d_model);
        } else {
            Wup_l.resize(p.L_lyr); Wdn_l.resize(p.L_lyr);
        }
        for (u32 l = 0; l < p.L_lyr; ++l) {
            char tag[40];
            if (!p.share_ep) {
                std::snprintf(tag,sizeof(tag),"BTX_RC_WUP_%u_V1",l);
                Wup_l[l] = expand_mx_dequant_i8(derive_operand_seed(seed_r,tag), p.d_model, p.d_ff);
                std::snprintf(tag,sizeof(tag),"BTX_RC_WDN_%u_V1",l);
                Wdn_l[l] = expand_mx_dequant_i8(derive_operand_seed(seed_r,tag), p.d_ff, p.d_model);
            }
            std::snprintf(tag,sizeof(tag),"BTX_RC_PRF_UP_%u_V1",l);
            prf_up[l] = derive_prf_key(derive_operand_seed(seed_r,tag));
            std::snprintf(tag,sizeof(tag),"BTX_RC_PRF_DN_%u_V1",l);
            prf_dn[l] = derive_prf_key(derive_operand_seed(seed_r,tag));
        }
        for (u32 l = 0; l < p.L_lyr; ++l)
            X[l+1] = fused_ffn_layer(X[l], p.share_ep ? Wup_s : Wup_l[l],
                                     p.share_ep ? Wdn_s : Wdn_l[l],
                                     prf_up[l], prf_dn[l], p.b_seq, p.d_model, p.d_ff);

        // ---- Phase 3: stream into Merkle (Z ‖ X[1..L]; H never committed) ----
        Merkle m(p.T_leaf);
        m.absorb(Z);
        for (u32 l = 0; l < p.L_lyr; ++l) m.absorb(X[l+1]);
        roots[r] = m.finalize_root();
    }
    std::vector<u8> buf;
    buf.insert(buf.end(), (const u8*)kEpisodeTag, (const u8*)kEpisodeTag + sizeof(kEpisodeTag)-1);
    for (const auto& rt : roots) buf.insert(buf.end(), rt.begin(), rt.end());
    return sha256d(buf.data(), buf.size());
}

// Phase B device code names the block length kBlockLen; the CPU oracle calls it
// kMxBlockLen. Same constant (32) -- alias rather than diverge.
static constexpr u32 kBlockLen = kMxBlockLen;

// ============================================================ device SHA-256
__constant__ u32 dK256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ u32 dror(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

// Host-visible copy of the SHA-256 round constants (dK256 is __constant__ = device-only).
// Used by the midstate precomputation below; values identical to dK256/Sha256::block.
static const u32 hK256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

// SHA MIDSTATE PRECOMPUTE (2026-08-02). Both hot SHA families here hash messages whose PREFIX
// is constant across millions of device hashes: the opgen stream preimage is
// seed(32)||domain||LE64(blk) (first 8 words = the seed), and the Extract scale preimage is
// tag(27)||prf(32)||LE32(i)||LE32(bj) (first 14 words constant). SHA-256 round t consumes only
// w[t], so the working registers after round N are a pure function of w[0..N] -- precompute them
// ONCE on host and start every device hash at round N+1. Bit-identical by construction (same
// math from a checkpoint); the toy golden gates it like everything else.
// Runs rounds 0..nr-1 from H0 given the first nr message words; captures the working registers.
static void sha_rounds_from_h0(const u32* w, int nr, u32 st[8])
{
    u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,
        e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    for (int i = 0; i < nr; ++i) {
        const u32 S1 = Sha256::ror(e,6) ^ Sha256::ror(e,11) ^ Sha256::ror(e,25);
        const u32 ch = (e & f) ^ (~e & g);
        const u32 t1 = h + S1 + ch + hK256[i] + w[i];
        const u32 S0 = Sha256::ror(a,2) ^ Sha256::ror(a,13) ^ Sha256::ror(a,22);
        const u32 mj = (a & b) ^ (a & c) ^ (b & c);
        const u32 t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]=a; st[1]=b; st[2]=c; st[3]=d; st[4]=e; st[5]=f; st[6]=g; st[7]=h;
}
static u32 be32_load(const u8* p) {
    return (u32(p[0])<<24) | (u32(p[1])<<16) | (u32(p[2])<<8) | u32(p[3]);
}

// ---- fixed block-2 schedule for the MX scale hash (2026-08-10) -------------------------------
// The 67-byte scale preimage pads to two SHA blocks. Block 2's message is bytes 64..66 =
// bl[1..3] of LE32(bj), then 0x80, zeros, and the constant bit length 536. At consensus dims
// every tile has bj < 65536 (max bj = n_ctx/32 = 24576), so bl[2] = bl[3] = 0 and the whole
// 16-word message is a function of ONE byte, bl1 = (bj >> 8) & 0xFF < 96. The 48-word schedule
// expansion w[16..63] is a pure function of the message, so it is a COMPILE-TIME table:
// kW2Sched[bl1][t-16], uploaded once to __constant__. Access is warp-uniform (tiles order bj
// fastest, so a warp shares bl1 except at 1-in-8 boundaries) -- broadcast, no replays. This
// replaces ~19% of the scale-SHA ALU with cached loads and RETIRES the block-2 rolling
// schedule registers -- the opposite direction from the four register-pressure kills.
static constexpr u32 cxror(u32 x, int n) { return (x >> n) | (x << (32 - n)); }
struct W2Sched { u32 w[96][48]; };
static constexpr W2Sched kW2Sched = [] {
    W2Sched s{};
    for (int b1 = 0; b1 < 96; ++b1) {
        u32 w[64] = {};
        w[0] = (u32(b1) << 24) | 0x80u;          // bl1 || 00 || 00 || 0x80
        w[15] = 536u;                             // bit length of the 67-byte message
        for (int t = 16; t < 64; ++t) {
            const u32 s0 = cxror(w[t-15],7) ^ cxror(w[t-15],18) ^ (w[t-15] >> 3);
            const u32 s1 = cxror(w[t-2],17) ^ cxror(w[t-2],19) ^ (w[t-2] >> 10);
            w[t] = w[t-16] + s0 + w[t-7] + s1;
        }
        for (int t = 16; t < 64; ++t) s.w[b1][t-16] = w[t];
    }
    return s;
}();
__device__ __constant__ u32 c_w2sched[96][48];
// spot-check the constexpr math against hand-computed rows (w16 = w0; w17 = s1(536)):
static_assert(kW2Sched.w[0][0] == 0x00000080u, "W2 schedule drifted (w16, bl1=0)");
static_assert(kW2Sched.w[0][1] == 0x014F0000u, "W2 schedule drifted (w17)");
static_assert(kW2Sched.w[95][0] == ((95u << 24) | 0x80u), "W2 schedule drifted (w16, bl1=95)");

// Single-block SHA-256 over a <=55-byte message (our scale preimage is 32+32+4+4 = 72 -> 2 blocks).
static __device__ void dsha256(const u8* msg, u32 len, u8 out[32])
{
    u32 h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[128];
    u32 total = len;
    for (u32 i = 0; i < len; ++i) blk[i] = msg[i];
    blk[len] = 0x80;
    u32 padded = ((len + 9 + 63) / 64) * 64;
    for (u32 i = len + 1; i < padded; ++i) blk[i] = 0;
    const u64 bits = u64(total) * 8;
    for (int i = 0; i < 8; ++i) blk[padded - 1 - i] = u8(bits >> (8 * i));
    for (u32 b = 0; b < padded; b += 64) {
        u32 w[64];
#pragma unroll
        for (int i = 0; i < 16; ++i)
            w[i] = (u32(blk[b+i*4]) << 24) | (u32(blk[b+i*4+1]) << 16) |
                   (u32(blk[b+i*4+2]) << 8) | u32(blk[b+i*4+3]);
        for (int i = 16; i < 64; ++i) {
            u32 s0 = dror(w[i-15],7) ^ dror(w[i-15],18) ^ (w[i-15] >> 3);
            u32 s1 = dror(w[i-2],17) ^ dror(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        u32 a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            u32 S1 = dror(e,6) ^ dror(e,11) ^ dror(e,25);
            u32 ch = (e & f) ^ (~e & g);
            u32 t1 = hh + S1 + ch + dK256[i] + w[i];
            u32 S0 = dror(a,2) ^ dror(a,13) ^ dror(a,22);
            u32 mj = (a & bb) ^ (a & c) ^ (bb & c);
            u32 t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=bb; bb=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=bb; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i*4]=u8(h[i]>>24); out[i*4+1]=u8(h[i]>>16);
        out[i*4+2]=u8(h[i]>>8); out[i*4+3]=u8(h[i]);
    }
}

// ============================================================ device ChaCha20
__device__ __forceinline__ u32 drotl(u32 x, int n) { return (x << n) | (x >> (32 - n)); }

static __device__ void dchacha_block(const u32 key[8], u32 nonce_first, u64 nonce_second,
                              u32 counter, u8 out[64])
{
    u32 in[16];
    in[0]=0x61707865; in[1]=0x3320646e; in[2]=0x79622d32; in[3]=0x6b206574;
#pragma unroll
    for (int i = 0; i < 8; ++i) in[4+i] = key[i];
    in[12] = counter;
    in[13] = nonce_first;
    in[14] = u32(nonce_second);
    in[15] = u32(nonce_second >> 32);
    u32 x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = in[i];
    for (int r = 0; r < 10; ++r) {
#define QR(a,b,c,d) { x[a]+=x[b]; x[d]=drotl(x[d]^x[a],16); \
                      x[c]+=x[d]; x[b]=drotl(x[b]^x[c],12); \
                      x[a]+=x[b]; x[d]=drotl(x[d]^x[a], 8); \
                      x[c]+=x[d]; x[b]=drotl(x[b]^x[c], 7); }
        QR(0,4,8,12) QR(1,5,9,13) QR(2,6,10,14) QR(3,7,11,15)
        QR(0,5,10,15) QR(1,6,11,12) QR(2,7,8,13) QR(3,4,9,14)
#undef QR
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        u32 v = x[i] + in[i];
        out[i*4]=u8(v); out[i*4+1]=u8(v>>8); out[i*4+2]=u8(v>>16); out[i*4+3]=u8(v>>24);
    }
}

// ============================================================ M11 table (device constant)
// value[] / accepted[] for the 16 E2M1 codes; rejects {1,3,8,9,11}.
__constant__ i8 dM11v[16] = {0,0,1,0,2,3,4,6, 0,0,-1,0,-2,-3,-4,-6};
__constant__ u8 dM11a[16] = {1,0,1,0,1,1,1,1, 0,0, 1,0, 1, 1, 1, 1};
// REGISTER-IMMEDIATE M11 (2026-08-02): __constant__ arrays indexed by a DATA-DIVERGENT nibble
// serialize into per-unique-index replays (up to 32-way) -- and the hot loops do ~128 such
// lookups per tile/block. Both tables fit in immediates: acceptance is one 16-bit mask, values
// pack into one u64 of 4-bit two's-complement nibbles (sign-extend via (x^8)-8). Derived from
// kM11 at compile time so they can never drift from the consensus table.
static constexpr u32 kM11AccMask = [] {
    u32 m = 0; for (int i = 0; i < 16; ++i) if (kM11.accepted[i]) m |= (1u << i); return m;
}();
static constexpr u64 kM11ValPack = [] {
    u64 t = 0; for (int i = 0; i < 16; ++i) t |= (u64(u8(kM11.value[i]) & 0xF)) << (4*i); return t;
}();
static_assert(kM11AccMask == 0xF4F5u, "M11 acceptance mask drifted");
static_assert(((i32((kM11ValPack >> (4*15)) & 0xF) ^ 8) - 8) == -6, "M11 value pack drifted");
__device__ __forceinline__ bool m11_accept(u32 nib) { return (kM11AccMask >> nib) & 1u; }
__device__ __forceinline__ i8  m11_value(u32 nib) {
    return i8((i32((kM11ValPack >> (4*nib)) & 0xF) ^ 8) - 8);
}

__device__ __forceinline__ u32 dmix_from_i64(i64 y)
{
    if (y >= (i64)INT32_MIN && y <= (i64)INT32_MAX) return u32((i32)y);
    const u64 u = (u64)y;
    return u32(u) ^ u32(u >> 32);
}

// Word-emitting ChaCha block: identical math to dchacha_block but returns the 16 keystream words
// in REGISTERS instead of staging 64 bytes through a local array. ncu (2026-08-02) showed
// k_extract_tiles is L2-BOUND at 84% with ~10x the necessary traffic -- local-memory spill from
// the dynamically-indexed ks[64] byte array. The nibble order is preserved exactly: within a
// little-endian word, byte b's low nibble then high nibble = (v >> 4k) & 0xF for k = 0..7.
static __device__ __forceinline__ void dchacha_block_w(const u32 key[8], u32 nonce_first,
                                                       u64 nonce_second, u32 counter, u32 out[16])
{
    u32 in[16];
    in[0]=0x61707865; in[1]=0x3320646e; in[2]=0x79622d32; in[3]=0x6b206574;
#pragma unroll
    for (int i = 0; i < 8; ++i) in[4+i] = key[i];
    in[12] = counter;
    in[13] = nonce_first;
    in[14] = u32(nonce_second);
    in[15] = u32(nonce_second >> 32);
    u32 x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = in[i];
    for (int r = 0; r < 10; ++r) {
#define QRW(a,b,c,d) { x[a]+=x[b]; x[d]=drotl(x[d]^x[a],16); \
                       x[c]+=x[d]; x[b]=drotl(x[b]^x[c],12); \
                       x[a]+=x[b]; x[d]=drotl(x[d]^x[a], 8); \
                       x[c]+=x[d]; x[b]=drotl(x[b]^x[c], 7); }
        QRW(0,4,8,12) QRW(1,5,9,13) QRW(2,6,10,14) QRW(3,7,11,15)
        QRW(0,5,10,15) QRW(1,6,11,12) QRW(2,7,8,13) QRW(3,4,9,14)
#undef QRW
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = x[i] + in[i];
}

// One thread per (i,bj) tile -- fully independent, which is exactly why this moves to GPU well.
// raw is INT32, straight off cuBLASLt. mix_bits_from_i64 returns u32(i32(y)) for any y in
// int32 range, and GuardInt32Bound proves every accumulator here is, so reading int32 and
// casting is BIT-IDENTICAL to widening to int64 first. Dropping the widen pass removes ~180 GB
// of traffic per episode (a 4B->8B expansion plus 8B instead of 4B reads here).
// Specialized SHA-256 for the MX scale: message is EXACTLY 67 bytes
// (tag[27] || prf[32] || LE32(i) || LE32(bj)) -> 2 fixed blocks. Builds the 16 message words
// directly (no blk[128] staging array, no runtime padding loop) and returns only state word 0
// (the scale needs (h0>>24)&3). Fully unrolled: every byte index is a compile-time constant, so
// the per-byte branch chain folds to direct tag-constant / prf-load / i,bj-shift assignments.
// Byte-identical to dsha256(msg,67) -> gated by the toy golden 5b1bff3c.
template <bool W2C>
__device__ __forceinline__ u32 dsha256_scale_h0(const u8* __restrict__ prf, u32 i, u32 bj)
{
    const char tag[] = "BTX_MATEXPAND_MXSCALE_V44LT";  // 27 bytes
    u32 s0h=0x6a09e667,s1h=0xbb67ae85,s2h=0x3c6ef372,s3h=0xa54ff53a,
        s4h=0x510e527f,s5h=0x9b05688c,s6h=0x1f83d9ab,s7h=0x5be0cd19;
    u32 w[64];
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            u32 word = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = blk*64 + t*4 + j;          // logical byte position (compile const)
                u32 by;
                if (idx < 27)       by = u32(u8(tag[idx]));
                else if (idx < 59)  by = u32(prf[idx-27]);
                else if (idx < 63)  by = (i  >> (8*(idx-59))) & 0xFFu;
                else if (idx < 67)  by = (bj >> (8*(idx-63))) & 0xFFu;
                else if (idx == 67) by = 0x80u;             // padding start (msg len 67)
                else if (idx == 126) by = 0x02u;            // bit-length 536 = 0x0218, hi byte
                else if (idx == 127) by = 0x18u;            // lo byte
                else by = 0u;
                word = (word << 8) | by;
            }
            w[t] = word;
        }
        // Block 2's schedule is a compile-time table row (see kW2Sched) whenever bj < 65536 --
        // always true at consensus dims. The guard keeps foreign shapes exact via the computed
        // path. Warp-uniform access: no constant-cache replays.
        if (W2C && blk == 1 && bj < 24576u) {
            const u32* ws = c_w2sched[(bj >> 8) & 0xFFu];
#pragma unroll
            for (int t = 16; t < 64; ++t) w[t] = ws[t - 16];
        } else {
#pragma unroll
            for (int t = 16; t < 64; ++t) {
                u32 a0 = dror(w[t-15],7) ^ dror(w[t-15],18) ^ (w[t-15] >> 3);
                u32 a1 = dror(w[t-2],17) ^ dror(w[t-2],19) ^ (w[t-2] >> 10);
                w[t] = w[t-16] + a0 + w[t-7] + a1;
            }
        }
        u32 a=s0h,b=s1h,c=s2h,d=s3h,e=s4h,f=s5h,g=s6h,hh=s7h;
#pragma unroll
        for (int t = 0; t < 64; ++t) {
            u32 S1 = dror(e,6) ^ dror(e,11) ^ dror(e,25);
            u32 ch = (e & f) ^ (~e & g);
            u32 t1 = hh + S1 + ch + dK256[t] + w[t];
            u32 S0 = dror(a,2) ^ dror(a,13) ^ dror(a,22);
            u32 mj = (a & b) ^ (a & c) ^ (b & c);
            u32 t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        s0h+=a; s1h+=b; s2h+=c; s3h+=d; s4h+=e; s5h+=f; s6h+=g; s7h+=hh;
    }
    return s0h;
}

// MEASURED DEAD (2026-08-02): a midstate-resumed scale SHA (precompute w[0..13] + registers
// after round 13 on host, resume per tile) is byte-exact but ~1% SLOWER end-to-end: the 23-u32
// ctx loads + extra register pressure in this kernel cost more than the 14 skipped rounds save.
// The same trick WINS in the opgen stream kernels (sha_stream_block_mid below) where the whole
// hash is one block and the byte-staging path is also eliminated. Do not re-add here without an A/B.
// MEASURED DEAD x2 (2026-08-02 evening, locked 2600, x2 interleaved, byte-exact both arms;
// code removed, this comment is the ledger pointer):
//   SEVENTH kill -- 256-bit PACKMIX: four ld.global.v8.b32 preloads of the tile's 32 raw i32,
//   mix nibbles ((raw*M)>>28) pre-folded into TWO u64 registers, rejection loop reads a
//   register funnel-shift instead of a per-attempt global load = +10.9% SLOWER
//   (1068.0/1068.0 vs base 963.3/963.3 ms). LDG.E.256 is REAL on sm_120 (ptxas 13.3 emits it;
//   probe pc:~/matador-prof/ld256_probe.cu) but useless HERE: the per-attempt loads were ~free
//   L1 hits (91.5%) and the pack pays register pressure for traffic that was already cached.
//   EIGHTH kill -- PDL: extract launched via cudaLaunchKernelEx + programmatic-stream-
//   serialization on cudaStreamPerThread (the GEMM's stream; sm_120 cuBLAS kernels are
//   PDL-enabled since 13.0 U1) with the prf-only prologue (key unpack + 2-block scale SHA)
//   hoisted before cudaGridDependencySynchronize = +29% SLOWER (1239.4/1246.6 ms). Failure
//   mode: the secondary's early CTAs occupy SMs running the SHA then SPIN at the dependency
//   wait, crowding out the producer GEMM's own remaining waves -- PDL is for SMALL secondaries
//   behind a draining primary, not a full-device grid behind a multi-wave GEMM.
//   With these, EVERY glue-class lever from the 08-02 NVIDIA-docs sweep is measured dead on
//   the 5090 at P1 dims (batch upload, PRF-only staging, PACKMIX, PDL): the episode keeps the
//   GPU saturated and the estimated launch/copy overhead was never real. See
//   research/matmul-v4/FINDINGS-nvidia-docs-sweep-2026-08-02.md.
// HAS_RESID (2026-08-05): fold the down-projection residual into the extract instead of
// materialising y = H.W_down + X first. The old path cost a k_widen8_32 pass (reads X int8,
// writes 268 MB of int32 at profile-1 dims) plus a beta=1 GEMM whose epilogue re-reads that
// 268 MB. cuBLASLt hides that C read behind its mainloop and pays ~nothing for beta=1
// (3.406 vs 3.426 ms); our CUTLASS epilogue does it serially and pays 3.7 points, which is the
// bulk of the -2.5% episode gap. Running the GEMM at beta=0 and adding X here removes BOTH.
//
// Byte-exact: identical int32 arithmetic in identical order, just performed one kernel later.
// Templated, not branched, so every existing call site emits exactly the code it did before --
// this kernel has six documented dead restructures and must not regress for the other extracts.
// PACK (2026-08-10, ncu-guided; template so each variant keeps its own register allocation --
// this kernel sits at the 3-CTAs/SM occupancy cliff and six restructures died to register
// pressure): the final store loop wrote the 32-byte tile as 32 single-byte stores; warp lanes
// sit 32 B apart, so that is one sector op per lane per instruction -- ncu measured EXACTLY
// 32 store sectors per tile and ~22 L2 write sectors of partial-sector amplification. The tile
// span is 16-byte aligned by construction (base = i*cols + bj*32, every consensus cols is a
// multiple of 32). PACK=1: eight u32 stores (one live accumulator, minimal register cost);
// PACK=2: two int4 stores (fewest transactions, pk[8] live at the tail). Values, positions and
// emission order identical to PACK=0 -- only the store width changes; the digest gates it.
// Env BTX_RC_EXTRACT_PACK (0/1/2) picks the instantiation at the launch site.
template <bool HAS_RESID, int PACK, bool W2C>
static __global__ void k_extract_tiles(const u8* __restrict__ prf,   // 32 bytes
                                const i32* __restrict__ raw,  // rows*cols, int32
                                i8* __restrict__ out,         // rows*cols
                                u32 rows, u32 cols,
                                const i8* __restrict__ resid) // rows*cols, or null
{
    // MEASURED DEAD (2026-08-21, locked, interleaved x3): __launch_bounds__(256, 4). ncu shows
    // this kernel at ~50% SM / ~35% DRAM and Block Limit Registers = 3 (79 regs), which reads
    // like a latency-bound kernel begging for occupancy. It is not. Capping to 64 regs buys the
    // 4th block (24 -> 32 warps/SM) at a 16 B spill store / 8 B spill load, and measured -5.0%
    // (1.470 -> 1.396 ep/s), digest unchanged. Occupancy does not help because the limiter is L2
    // SECTOR TRAFFIC, not warp latency: extra warps add L2 pressure rather than hiding it, and
    // the spill routes more traffic through the same L1/L2. SEVENTH dead restructure -- check
    // lts__ throughput before believing an SM%/DRAM% pair that looks latency-bound.
    // MEASURED DEAD (2026-08-02, locked, x2): warp-cooperative smem staging of the raw span
    // (coalesced 4 KiB/warp, transposed, conflict-free reads) = +8% SLOWER despite directly
    // targeting the ncu-measured L2-sector limiter. The 32 KiB/block smem carveout shrinks the
    // unified L1 that was delivering the 91.5% hit rate, and the transposed stores eat 32-way
    // bank conflicts. SIXTH dead restructure on this kernel; the direct L1-cached read pattern
    // wins. Remaining accepted truth: extract is L2-sector-bound and the layout is consensus-fixed.
    const u32 nblk = cols / kBlockLen;
    const u64 total = u64(rows) * nblk;
    const u64 tile = u64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tile >= total) return;
    // POW2 strength reduction (2026-08-10): every consensus cols is a power of two, so nblk is
    // too; the u64 div/mod pair is a ~40-instruction subroutine per thread. Warp-uniform branch,
    // same arithmetic values either way; foreign dims keep the computed path.
    const bool nb2 = (nblk & (nblk - 1u)) == 0u;
    const u32 nbsh = nb2 ? (31u - u32(__clz(nblk))) : 0u;
    const u32 i  = nb2 ? u32(tile >> nbsh) : u32(tile / nblk);
    const u32 bj = nb2 ? (u32(tile) & (nblk - 1u)) : u32(tile % nblk);
    const size_t base = size_t(i) * cols + size_t(bj) * kBlockLen;

    u32 key[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
        key[k] = u32(prf[k*4]) | (u32(prf[k*4+1])<<8) | (u32(prf[k*4+2])<<16) | (u32(prf[k*4+3])<<24);

    i8 mu[kBlockLen];
    u32 filled = 0, remix = 0;
    // Rejection-loop load hoist (2026-08-21). m11_accept takes 11 of 16 nibbles
    // (kM11AccMask=0xF4F5 = 68.75%), so `filled` STALLS on ~31% of inner iterations -- and the
    // address raw[base+filled] is invariant while it stalls. The straight-line version re-issues
    // that load (and resid's) on every retry: ~1.45 loads per accepted value, ~45% of them
    // redundant. ncu puts this kernel at 86% L2 with L1 at only 42%, i.e. limited by REQUEST /
    // sector count, which is exactly what the redundant re-issues inflate. So load once per
    // `filled` and refresh only when it advances.
    // Byte-exact by construction: identical values consumed in identical order; only the number
    // of times the same address is fetched changes.
    // MEASURED (2026-08-21, order-balanced A/B/B/A x3 at plateau): +0.61% episode throughput
    // (1.4418 -> 1.4507 ep/s), and the hoisted arm did it at a LOWER mean clock (2271 vs
    // 2277 MHz), so ~+0.9% clock-normalised. Consistent per quad (+0.55/+0.66/+0.63%), digest
    // 42e74cd6 unchanged, 0 bytes spill either way. Small, but it is the only lever on this
    // kernel that has gone the right way: the axes ledger below is otherwise all kills.
    i32 cur_raw = HAS_RESID ? (raw[base] + i32(resid[base])) : raw[base];
    while (filled < kBlockLen) {
        // Keystream in 16 REGISTER words (not a local ks[64] byte array): kills the dominant
        // L1<->L2 spill traffic (ncu: kernel L2-bound at 84% with ~10x necessary traffic).
        // Nibble order within a LE word is (v >> 4k) & 0xF, k=0..7 -- identical to the byte walk.
        u32 ksw[16];
        dchacha_block_w(key, bj ^ kLaneMxBlock, (u64(i) << 32) | u64(bj), remix, ksw);
#pragma unroll
        for (int wi = 0; wi < 16; ++wi) {
            if (filled >= kBlockLen) break;
            const u32 v = ksw[wi];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (filled >= kBlockLen) break;
                const u8 nib = u8((v >> (4*k)) & 0x0F);
                // The residual add rides on a load the kernel already performs; the extra int8
                // fetch hits the same L1 line pattern that gives this kernel its 91.5% hit rate.
                const u32 raw_u = u32(cur_raw);              // == dmix_from_i64 in int32 range
                const u8 mixed = u8((nib ^ u8((raw_u * 0x9E3779B9u) >> 28)) & 0x0F);
                if (m11_accept(mixed)) {
                    mu[filled++] = m11_value(mixed);
                    if (filled < kBlockLen)
                        cur_raw = HAS_RESID ? (raw[base + filled] + i32(resid[base + filled]))
                                            : raw[base + filled];
                }
            }
        }
        ++remix;
    }
    // scale: SHA256(tag || prf || LE32(i) || LE32(bj)) digest byte0 & 3 = (state h0 >> 24) & 3.
    // Specialized 2-block SHA (no staging array) -- see dsha256_scale_h0.
    const u8 e = u8((dsha256_scale_h0<W2C>(prf, i, bj) >> 24) & 0x3);
    if (PACK == 2) {
        u32 pk[8];
#pragma unroll
        for (int q = 0; q < 8; ++q)
            pk[q] = u32(u8(i8(i32(mu[4*q  ]) * (i32{1} << e))))
                  | (u32(u8(i8(i32(mu[4*q+1]) * (i32{1} << e)))) << 8)
                  | (u32(u8(i8(i32(mu[4*q+2]) * (i32{1} << e)))) << 16)
                  | (u32(u8(i8(i32(mu[4*q+3]) * (i32{1} << e)))) << 24);
        int4* dst = reinterpret_cast<int4*>(out + base);
        dst[0] = make_int4((i32)pk[0], (i32)pk[1], (i32)pk[2], (i32)pk[3]);
        dst[1] = make_int4((i32)pk[4], (i32)pk[5], (i32)pk[6], (i32)pk[7]);
    } else if (PACK == 1) {
        u32* dst = reinterpret_cast<u32*>(out + base);
#pragma unroll
        for (int q = 0; q < 8; ++q)
            dst[q] = u32(u8(i8(i32(mu[4*q  ]) * (i32{1} << e))))
                   | (u32(u8(i8(i32(mu[4*q+1]) * (i32{1} << e)))) << 8)
                   | (u32(u8(i8(i32(mu[4*q+2]) * (i32{1} << e)))) << 16)
                   | (u32(u8(i8(i32(mu[4*q+3]) * (i32{1} << e)))) << 24);
    } else {
#pragma unroll
        for (u32 t = 0; t < kBlockLen; ++t) out[base + t] = i8(i32(mu[t]) * (i32{1} << e));
    }
}

// MEASURED DEAD (2026-08-02, locked 2600 MHz, x2 interleaved): ILP-restructured extract
// (precompute all 32 mix nibbles via int4 loads, then issue scale-SHA + first ChaCha block as
// adjacent straight-line chains so ptxas can interleave them, rejection loop reduced to
// xor/lookup) = +3.5% SLOWER (877.1/878.9 OFF vs 907.1/909.3 ON), byte-exact. The extended live
// ranges (mixn + ks + the SHA schedule all resident at once) cost more in register pressure /
// spills than the cross-chain ILP wins. FOURTH dead axis on this kernel (occupancy x3, ring
// schedule, scale-midstate, ILP reorder): the legacy short-live-range order IS the local optimum.
// The kernel's ~17% ALU efficiency is real but every restructure loses more elsewhere -- do not
// re-attack without ncu evidence of a different limiter.


// ============================================================ GPU episode kernels
#define GCK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- cuBLASLt INT8 tensor-core GEMM -------------------------------------------------------
// cuBLASLt is COLUMN-major. Using  row-major X(r,c) == col-major X^T(c,r) at the same ld,
// row-major C = A.B becomes col-major C' = B'.A'  (since (A.B)^T = B^T.A^T). So pass B first,
// then A, with NO transpose ops:
//     B row-major (k,n) ld=n -> layout (n,k,n)
//     A row-major (m,k) ld=k -> layout (k,m,k)
//     C row-major (m,n) ld=n -> layout (n,m,n)
// Getting this wrong yields a WRONG DIGEST and drops off the fast path -- an earlier attempt used
// (k,n,k)/(m,n,m) with TRANSA=T and did both. The toy golden is what caught it.
//
// cuBLASLt emits int32 while the Extracts consume int64. Safe because Extract outputs are
// mu*(1<<e) with |mu|<=6, e<=3 => |value| <= 48, so the accumulator bound is 2304*contraction.
// GuardInt32Bound refuses rather than silently forking the digest if a shape ever breaks that.
#if MATADOR_USE_CUBLASLT
static cublasLtHandle_t g_lt = nullptr;
static thread_local void* g_ws = nullptr;
// Workspace bounds the cuBLASLt algo search: bigger unlocks split-K / more aggressive kernels the
// autotune can then pick. Env BTX_RC_GEMM_WS_MB overrides the 256 MiB default (novel A/B knob).
static size_t GemmWorkspaceBytes() {
    const char* v = getenv("BTX_RC_GEMM_WS_MB");
    size_t mb = 256;
    if (v != nullptr) { long m = atol(v); if (m >= 16 && m <= 8192) mb = (size_t)m; }
    return mb << 20;
}
static size_t g_wsz = GemmWorkspaceBytes();

// ---- cuBLASLt per-shape algo autotune (NOVEL 2026-07-31) ------------------------------------
// The GEMMs previously passed a nullptr algo, letting cuBLASLt apply a generic default heuristic.
// Our episode shapes are FIXED and recur every round/layer, so a one-time measured autotune per
// shape (heuristic candidates -> time each -> cache the fastest algo) is amortized to nothing and
// can beat the default kernel selection. Byte-exact: every int8 IMMA algo produces the identical
// int32 product; only tiling/kernel differs. Gated by BTX_RC_GEMM_TUNE (default ON; =0 = old
// nullptr path) so the A/B is one binary. The cache is keyed by the full descriptor shape.
struct GemmAlgoKey {
    int op0, op1, r0, c0, ld0, r1, c1, ld1, rc, cc, ldc, bt;
};
static bool operator<(const GemmAlgoKey& a, const GemmAlgoKey& b) {
    return std::memcmp(&a, &b, sizeof(GemmAlgoKey)) < 0;
}
static std::mutex g_algo_mtx;
static std::map<GemmAlgoKey, cublasLtMatmulAlgo_t> g_algo_cache;
// BTX_RC_GEMM_TUNE: 0 = off (nullptr algo), 1/unset = timed heuristic candidates (07-31 lever),
// 2 = DEEP search (08-02): enumerate ALL int8 algo ids + their capability space (tile ids, stage
// counts, split-K, CTA swizzle) instead of only what the heuristic returns. Split-K is byte-exact
// here because the int32 accumulation is integer addition (associative, exact) regardless of the
// reduction order. AlgoCheck prunes invalid combos cheaply; only survivors are timed. One-time
// cost per shape (~seconds, absorbed by the warm-up episode), cached like level 1.
static int GemmTuneLevel() {
    const char* v = getenv("BTX_RC_GEMM_TUNE");
    if (v == nullptr) return 1;
    return atoi(v);
}
static bool GemmTuneEnabled() { return GemmTuneLevel() > 0; }

// ---- disk-persisted tune cache (NOVEL 2026-08-02, NVIDIA-docs sweep) ------------------------
// cuBLASLt 13.3 §3.3.5 documents cublasLtMatmulAlgo_t as "trivially serialized and later
// restored for use with the same version of cuBLAS library". g_algo_cache is per-PROCESS, so a
// one-shot run (matador-verify share audit, cold replay) re-pays the timed autotune every time
// -- the tune is most of the 1.38 s cold vs 0.85 s warm episode gap. Persist tuned algos keyed
// on (cublasLt version, GPU name, workspace, tune level); any header mismatch = ignore and
// overwrite. A disk-loaded algo is AlgoCheck-validated at its first use and falls back to a
// fresh tune rather than trusting the file. Gate BTX_RC_TUNE_CACHE (default ON; =0 = no disk
// IO); path override BTX_RC_TUNE_CACHE_PATH (default ~/.cache/matador-miner/rc_gemm_tune.bin).
// Byte-exact by the same argument as the autotune: every int8 IMMA algo yields the identical
// int32 product; only kernel selection differs.
static std::set<GemmAlgoKey> g_algo_unverified;   // disk-loaded, AlgoCheck pending (first use)
static bool TuneCacheEnabled() {
    const char* v = getenv("BTX_RC_TUNE_CACHE");
    return v == nullptr || (v[0] != '0');
}
struct TuneCacheHeader {
    u32 magic, fver;          // 'RCT1', format version
    u64 lt_ver;               // cublasLtGetVersion()
    u64 ws;                   // workspace bytes (candidate set is workspace-gated)
    i32 tune_level;
    char dev[64];             // cudaDeviceProp.name
};
static std::string TuneCachePath() {
    if (const char* p = getenv("BTX_RC_TUNE_CACHE_PATH")) return std::string(p);
    const char* home = getenv("HOME");
    if (home == nullptr || home[0] == 0) return std::string();
    return std::string(home) + "/.cache/matador-miner/rc_gemm_tune.bin";
}
static TuneCacheHeader TuneCacheHdr() {
    TuneCacheHeader h;
    std::memset(&h, 0, sizeof(h));   // padding participates in the memcmp match; zero it all
    h.magic = 0x31544352u;    // "RCT1"
    h.fver = 1;
    h.lt_ver = (u64)cublasLtGetVersion();
    h.ws = (u64)g_wsz;
    h.tune_level = GemmTuneLevel();
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        std::strncpy(h.dev, prop.name, sizeof(h.dev) - 1);
    return h;
}
// Both run under g_algo_mtx. Load once per process; save rewrites the whole (tiny) map
// atomically via tmp+rename so a concurrent process never reads a torn file.
static void TuneCacheLoad() {
    const std::string path = TuneCachePath();
    if (path.empty()) return;
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) return;
    const TuneCacheHeader want = TuneCacheHdr();
    TuneCacheHeader got{};
    u32 count = 0;
    bool ok = fread(&got, sizeof(got), 1, f) == 1 &&
              std::memcmp(&got, &want, sizeof(got)) == 0 &&
              fread(&count, sizeof(count), 1, f) == 1 && count <= 256;
    u32 loaded = 0;
    for (u32 i = 0; ok && i < count; ++i) {
        GemmAlgoKey k{}; cublasLtMatmulAlgo_t a{};
        if (fread(&k, sizeof(k), 1, f) != 1 || fread(&a, sizeof(a), 1, f) != 1) break;
        g_algo_cache[k] = a;
        g_algo_unverified.insert(k);
        ++loaded;
    }
    fclose(f);
    if (loaded) printf("[rc-tune-cache] loaded %u tuned algo(s) from %s\n", loaded, path.c_str());
}
static void TuneCacheSave() {
    const std::string path = TuneCachePath();
    if (path.empty()) return;
    const size_t slash = path.rfind('/');
    if (slash != std::string::npos) {   // best-effort mkdir -p (two levels covers ~/.cache/X)
        const std::string dir = path.substr(0, slash);
        const size_t s2 = dir.rfind('/');
        if (s2 != std::string::npos) ::mkdir(dir.substr(0, s2).c_str(), 0755);
        ::mkdir(dir.c_str(), 0755);
    }
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) return;
    const TuneCacheHeader h = TuneCacheHdr();
    const u32 count = (u32)g_algo_cache.size();
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1 && fwrite(&count, sizeof(count), 1, f) == 1;
    for (auto it = g_algo_cache.begin(); ok && it != g_algo_cache.end(); ++it)
        ok = fwrite(&it->first, sizeof(it->first), 1, f) == 1 &&
             fwrite(&it->second, sizeof(it->second), 1, f) == 1;
    fclose(f);
    if (ok) ::rename(tmp.c_str(), path.c_str()); else ::remove(tmp.c_str());
}
#endif  // MATADOR_USE_CUBLASLT (handle, workspace, algo tune-cache)

static void GuardInt32Bound(const char* which, u32 contraction)
{
    const double worst = 2304.0 * double(contraction);
    if (worst >= 2147483647.0) {
        printf("!! %s: contraction %u -> worst case %.3g >= 2^31; int32 GEMM would overflow and "
               "fork the digest. Needs an int64 path.\n", which, contraction, worst);
        exit(2);
    }
}
// One entry point; the three row-major forms below differ only in descriptors. Materialising a
// transpose costs a full read+write pass (dGt alone is 1 GiB/round), and cuBLASLt can consume the
// untransposed operand directly via TRANSA/TRANSB.
// Pool accessors for the beta!=0 tune scratch (the Dev pool is defined further down).
static size_t TuneScratchMark();
static i32*   TuneScratchI32(size_t n);
static void   TuneScratchRelease(size_t mk);
static i8*    TuneScratchI8(size_t n);   // B^T staging for the cutlass backend
// beta_v: 0 = overwrite (default), 1 = D = A.B + C with C preloaded (residual fusion; int32
// addition is exact under any order, GuardInt32Bound covers the sum). Tuning with beta!=0 must
// NOT write the real C repeatedly -- candidates are timed into a throwaway pool buffer instead.
#if MATADOR_USE_CUBLASLT
static void gemm8_ex(const i8* P0, int r0, int c0, int ld0, cublasOperation_t op0,
                     const i8* P1, int r1, int c1, int ld1, cublasOperation_t op1,
                     i32* C, int rc, int cc, int ldc, const char* what, i32 beta_v = 0)
{
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &op0, sizeof(op0));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &op1, sizeof(op1));
    cublasLtMatrixLayout_t l0, l1, lc;
    cublasLtMatrixLayoutCreate(&l0, CUDA_R_8I,  r0, c0, ld0);
    cublasLtMatrixLayoutCreate(&l1, CUDA_R_8I,  r1, c1, ld1);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, rc, cc, ldc);
    i32 alpha = 1, beta = beta_v;

    // Select the algo: cached best, freshly-tuned, or nullptr (tune disabled).
    const cublasLtMatmulAlgo_t* algo_ptr = nullptr;
    cublasLtMatmulAlgo_t chosen;
    if (GemmTuneEnabled()) {
        const GemmAlgoKey key{op0, op1, r0, c0, ld0, r1, c1, ld1, rc, cc, ldc, (int)beta_v};
        std::lock_guard<std::mutex> lk(g_algo_mtx);
        static std::once_flag tc_once;
        if (TuneCacheEnabled()) std::call_once(tc_once, TuneCacheLoad);
        auto it = g_algo_cache.find(key);
        if (it != g_algo_cache.end()) {
            chosen = it->second;
            // Disk-loaded algo: validate against THIS descriptor before first use. A stale or
            // foreign config fails AlgoCheck cheaply; drop it and fall through to a fresh tune.
            auto uv = g_algo_unverified.find(key);
            if (uv != g_algo_unverified.end()) {
                g_algo_unverified.erase(uv);
                cublasLtMatmulHeuristicResult_t chk{};
                if (cublasLtMatmulAlgoCheck(g_lt, op, l0, l1, lc, lc, &chosen, &chk)
                        != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > g_wsz) {
                    printf("[rc-tune-cache] AlgoCheck rejected disk algo (%s); re-tuning\n", what);
                    g_algo_cache.erase(it);
                    it = g_algo_cache.end();
                }
            }
        }
        if (it != g_algo_cache.end()) {
            algo_ptr = &chosen;
        } else {
            // With beta!=0 the timing loop must not accumulate into the caller's C.
            const size_t tune_mk = TuneScratchMark();
            i32* Ct = (beta_v != 0) ? TuneScratchI32((size_t)rc * cc) : C;
            cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_wsz, sizeof(g_wsz));
            cublasLtMatmulHeuristicResult_t cand[16]; int ncand = 0;
            cublasLtMatmulAlgoGetHeuristic(g_lt, op, l0, l1, lc, lc, pref, 16, cand, &ncand);
            cublasLtMatmulPreferenceDestroy(pref);
            bool have_best = false; float best_ms = 1e30f;
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            // time one config: warmup once, then timed iters into the real C (beta=0 ->
            // byte-identical result for every config, so C is left correct after tuning)
            auto time_algo = [&](const cublasLtMatmulAlgo_t& a, int iters) -> float {
                if (cublasLtMatmul(g_lt, op, &alpha, P0, l0, P1, l1, &beta, Ct, lc, Ct, lc,
                                   &a, g_ws, g_wsz, cudaStreamPerThread) != CUBLAS_STATUS_SUCCESS)
                    return -1.f;
                cudaStreamSynchronize(cudaStreamPerThread);
                cudaEventRecord(e0, cudaStreamPerThread);
                for (int r = 0; r < iters; ++r)
                    cublasLtMatmul(g_lt, op, &alpha, P0, l0, P1, l1, &beta, Ct, lc, Ct, lc,
                                   &a, g_ws, g_wsz, cudaStreamPerThread);
                cudaEventRecord(e1, cudaStreamPerThread);
                cudaEventSynchronize(e1);
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                return ms / iters;
            };
            for (int i = 0; i < ncand; ++i) {
                const float ms = time_algo(cand[i].algo, 5);
                if (ms >= 0.f && ms < best_ms) { best_ms = ms; chosen = cand[i].algo; have_best = true; }
            }
            // Level 2: full config-space search beyond the heuristic's picks (see GemmTuneLevel).
            if (GemmTuneLevel() >= 2) {
                int ids[64]; int nids = 0; int timed = 0;
                if (cublasLtMatmulAlgoGetIds(g_lt, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I,
                                             CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, 64, ids, &nids)
                    == CUBLAS_STATUS_SUCCESS) {
                    for (int a = 0; a < nids && timed < 128; ++a) {
                        cublasLtMatmulAlgo_t algo;
                        if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32I, CUDA_R_32I, CUDA_R_8I,
                                                   CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, ids[a], &algo)
                            != CUBLAS_STATUS_SUCCESS) continue;
                        size_t w = 0;
                        u32 tiles[32]; size_t tsz = 0;
                        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS,
                                                          tiles, sizeof(tiles), &tsz);
                        int ntile = (int)(tsz / sizeof(u32));
                        if (ntile == 0) { tiles[0] = 0; ntile = 1; }
                        u32 stg[16]; size_t ssz = 0;
                        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                                          stg, sizeof(stg), &ssz);
                        int nstg = (int)(ssz / sizeof(u32));
                        if (nstg == 0) { stg[0] = 0; nstg = 1; }
                        i32 splitk_ok = 0;
                        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
                                                          &splitk_ok, sizeof(splitk_ok), &w);
                        u32 swiz_ok = 0;
                        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
                                                          &swiz_ok, sizeof(swiz_ok), &w);
                        u32 red_mask = 0;
                        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
                                                          &red_mask, sizeof(red_mask), &w);
                        static const i32 splits[4] = {1, 2, 4, 8};
                        for (int ti = 0; ti < ntile && ti < 8; ++ti)
                        for (int si = 0; si < nstg && si < 4; ++si)
                        for (int ki = 0; ki < (splitk_ok ? 4 : 1); ++ki)
                        for (u32 sw = 0; sw <= (swiz_ok ? 1u : 0u); ++sw) {
                            if (timed >= 128) break;
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                                                 &tiles[ti], sizeof(tiles[ti]));
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                                                 &stg[si], sizeof(stg[si]));
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                                 &splits[ki], sizeof(splits[ki]));
                            // split-K needs a reduction scheme; int32 adds are exact under any order
                            const u32 red = (splits[ki] > 1)
                                ? ((red_mask & CUBLASLT_REDUCTION_SCHEME_INPLACE)
                                       ? (u32)CUBLASLT_REDUCTION_SCHEME_INPLACE : 0u)
                                : (u32)CUBLASLT_REDUCTION_SCHEME_NONE;
                            if (splits[ki] > 1 && red == 0u) continue;
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                                 &red, sizeof(red));
                            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                                                 &sw, sizeof(sw));
                            cublasLtMatmulHeuristicResult_t chk;
                            if (cublasLtMatmulAlgoCheck(g_lt, op, l0, l1, lc, lc, &algo, &chk)
                                != CUBLAS_STATUS_SUCCESS) continue;
                            if (chk.workspaceSize > g_wsz) continue;
                            const float ms = time_algo(algo, 3);
                            if (ms < 0.f) continue;
                            ++timed;
                            if (ms < best_ms) { best_ms = ms; chosen = algo; have_best = true; }
                        }
                    }
                }
            }
            cudaEventDestroy(e0); cudaEventDestroy(e1);
            TuneScratchRelease(tune_mk);   // stream drained by the last cudaEventSynchronize
            if (have_best) {
                algo_ptr = &chosen;
                g_algo_cache[key] = chosen;
                if (TuneCacheEnabled()) TuneCacheSave();
            }
        }
    }

    cublasStatus_t st = cublasLtMatmul(g_lt, op, &alpha, P0, l0, P1, l1, &beta, C, lc, C, lc,
                                       algo_ptr, g_ws, g_wsz, cudaStreamPerThread);
    if (st != CUBLAS_STATUS_SUCCESS) {
        printf("!! cublasLtMatmul FAILED status=%d (%s)\n", (int)st, what);
        exit(2);
    }
    cublasLtMatrixLayoutDestroy(l0); cublasLtMatrixLayoutDestroy(l1);
    cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
}
#endif  // MATADOR_USE_CUBLASLT (gemm8_ex)

// BACKEND SWITCH (2026-08-05). BTX_RC_GEMM_BACKEND selects who does the int8 GEMM:
//   "cublaslt" (default) -- the static cuBLASLt link, ~530 MB of the shipped binary
//   "cutlass"            -- cuda/rc_gemm_i8_cutlass.cuh, the same kernel family compiled from
//                           CUTLASS source with only the configs we use
// Both are bit-identical by construction (int8 into int32 is exact integer arithmetic and
// integer addition is associative), so this switch cannot move the digest -- which is what the
// rc_gpu_accel_probe golden gate verifies rather than assumes. Default stays cuBLASLt until the
// episode A/B and the golden gate have both cleared on the cutlass path.
static bool RcGemmCutlassBackend()
{
#ifdef MATADOR_HAVE_CUTLASS
#if !MATADOR_USE_CUBLASLT
    return true;                 // cuBLASLt is not compiled in; there is nothing to switch to
#else
    static const bool on = [] {
        const char* v = getenv("BTX_RC_GEMM_BACKEND");
        return v != nullptr && std::strcmp(v, "cutlass") == 0;
    }();
    return on;
#endif
#else
    return false;
#endif
}

// row-major C(m,n) = A(m,k) . B(k,n).   C' = B'.A'   (beta_v=1: C preloaded, D = A.B + C)
static void gemm8(const i8* A, const i8* B, i32* C, int m, int n, int k, i32 beta_v = 0)
{
#ifdef MATADOR_HAVE_CUTLASS
    if (RcGemmCutlassBackend()) {
        // Scratch for B^T, taken from the same pool everything else uses and released straight
        // away: the release only rewinds a bump pointer, and any reuse is a later call on the
        // same stream, so it cannot race the kernel still reading it.
        const size_t mk = TuneScratchMark();
        i8* scratch = TuneScratchI8((size_t)k * n);
        const cudaError_t e =
            rcgemm::gemm_i8_nn_auto(A, B, C, m, n, k, beta_v, scratch, cudaStreamPerThread);
        TuneScratchRelease(mk);
        if (e != cudaSuccess) {
            printf("!! rc gemm (cutlass) FAILED: %s\n", cudaGetErrorString(e));
            exit(2);
        }
        return;
    }
#endif
#if MATADOR_USE_CUBLASLT
    gemm8_ex(B, n, k, n, CUBLAS_OP_N, A, k, m, k, CUBLAS_OP_N, C, n, m, n,
             beta_v ? "nn_b1" : "nn", beta_v);
#else
    printf("!! rc gemm: no backend compiled in (need CUTLASS or MATADOR_USE_CUBLASLT)\n");
    exit(2);
#endif
}
// i8 -> i32 widen (residual preload for the beta=1 down-proj GEMM)
static __global__ void k_widen8_32(const i8* __restrict__ in, i32* __restrict__ out, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (i32)in[i];
}
// MEASURED DEAD (2026-07-20): folding the transpose into TRANSA/TRANSB is byte-exact but 10%
// SLOWER than materialising it (1.998 -> 1.804 ep/s on a 5090). INT8 IMMA wants specific operand
// layouts; the transposed path drops cuBLASLt onto a worse kernel, and that costs far more than
// the ~8 ms of traffic a separate transpose spends. A bandwidth-only model predicted +1.5%.
// Kept for the record so nobody re-derives it. Do not re-enable without an A/B.
#if MATADOR_USE_CUBLASLT
// row-major C(m,n) = A(m,k) . B(n,k)^T. C' = (B')^T.A'  -- B stays untransposed in memory
static void gemm8_nt(const i8* A, const i8* B, i32* C, int m, int n, int k)
{
    gemm8_ex(B, k, n, k, CUBLAS_OP_T, A, k, m, k, CUBLAS_OP_N, C, n, m, n, "nt");
}
// row-major C(m,n) = A(k,m)^T . B(k,n). C' = B'.(A')^T  -- A stays untransposed in memory
static void gemm8_tn(const i8* A, const i8* B, i32* C, int m, int n, int k)
{
    gemm8_ex(B, n, k, n, CUBLAS_OP_N, A, m, k, m, CUBLAS_OP_T, C, n, m, n, "tn");
}
#endif  // MATADOR_USE_CUBLASLT (unused nt/tn forms)
static __global__ void k_widen(const i32* __restrict__ in, i64* __restrict__ out, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (i64)in[i];
}

// C(rows x cols) = A(rows x inner) * B(inner x cols), int8 in, int64 accumulate.
static __global__ void k_gemm_i64(const i8* __restrict__ A, const i8* __restrict__ B,
                           i64* __restrict__ C, u32 rows, u32 inner, u32 cols)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    i64 acc = 0;
    for (u32 k = 0; k < inner; ++k)
        acc += (i64)A[(size_t)r * inner + k] * (i64)B[(size_t)k * cols + c];
    C[(size_t)r * cols + c] = acc;
}
// C(rows x cols) = A^T(rows x inner) * B(inner x cols) where A is (inner x rows)  [wgrad G^T.X]
static __global__ void k_gemm_at_i64(const i8* __restrict__ A, const i8* __restrict__ B,
                              i64* __restrict__ C, u32 rows, u32 inner, u32 cols)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    i64 acc = 0;
    for (u32 k = 0; k < inner; ++k)
        acc += (i64)A[(size_t)k * rows + r] * (i64)B[(size_t)k * cols + c];
    C[(size_t)r * cols + c] = acc;
}
// residual: y[i][j] += X[i][j]   (inside the single forward Extract, per his H5 pin)
static __global__ void k_add_resid(i32* __restrict__ y, const i8* __restrict__ X, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += (i32)X[i];   // bounded by GuardInt32Bound + |resid| <= 48
}
// transpose int8
static __global__ void k_transpose_i8(const i8* __restrict__ M, i8* __restrict__ T, u32 rows, u32 cols)
{
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols) return;
    T[(size_t)c * rows + r] = M[(size_t)r * cols + c];
}

// Pooling device allocator. The naive version cudaMalloc'd every buffer: ~200 per episode from
// the round/layer loops plus 4 per gpu_expand_mx_dequant call (21 calls/round = 336 more).
// cudaMalloc/cudaFree serialise against the null stream, so that churn showed up as GPU idle --
// util sat at 68.7% on a B200, 93.1% on a 5090. Stalls cost a wide machine more than a narrow
// one, which biased the fleet comparison against the widest card.
//
// mark()/release_to() lets a callee recycle its own temporaries without freeing the caller's.
struct Dev {
    std::vector<std::pair<size_t,void*>> free_, used_;
    void* get(size_t n) {
        // BEST fit, not first fit. First fit hands a 64 MiB block to a 32-byte request, the next
        // 64 MiB request then allocates fresh, and the pool grows without bound -> OOM at
        // production dims. Sizes repeat exactly every round, so best fit converges on perfect
        // reuse: round 2+ allocates nothing.
        size_t best = SIZE_MAX; long bi = -1;
        for (size_t i = 0; i < free_.size(); ++i)
            if (free_[i].first >= n && free_[i].first < best) { best = free_[i].first; bi = (long)i; }
        if (bi >= 0) {
            auto e = free_[(size_t)bi];
            free_.erase(free_.begin() + bi);
            used_.push_back(e);
            return e.second;
        }
        void* p = nullptr;
        if (cudaMalloc(&p, n) != cudaSuccess) {        // reclaim and retry before giving up
            cudaGetLastError();
            for (auto& e : free_) cudaFree(e.second);
            free_.clear();
            GCK(cudaMalloc(&p, n));
        }
        used_.push_back({n, p});
        return p;
    }
    size_t mark() const { return used_.size(); }
    void release_to(size_t m) {
        while (used_.size() > m) { free_.push_back(used_.back()); used_.pop_back(); }
    }
    void release_all() { release_to(0); }
    void destroy() {
        for (auto& e : free_) cudaFree(e.second);
        for (auto& e : used_) cudaFree(e.second);
        free_.clear(); used_.clear();
    }
    i8*  d_i8 (size_t n) { return (i8*) get(n); }
    i64* d_i64(size_t n) { return (i64*)get(n * sizeof(i64)); }
    i32* d_i32(size_t n) { return (i32*)get(n * sizeof(i32)); }
    u8*  d_u8 (size_t n) { return (u8*) get(n); }
};
// THREAD_LOCAL: with --default-stream per-thread each host thread gets its own implicit stream,
// so N threads run N episodes concurrently without touching a single kernel launch site. Every
// piece of mutable per-episode state must therefore be per-thread: the pool, the cuBLASLt
// workspace (concurrent matmuls sharing one scratch buffer would corrupt each other), the
// shortfall flag, and the pad-leaf constant.
static thread_local Dev g_dev;
static size_t TuneScratchMark() { return g_dev.mark(); }
static i32*   TuneScratchI32(size_t n) { return g_dev.d_i32(n); }
static void   TuneScratchRelease(size_t mk) { g_dev.release_to(mk); }
static i8*    TuneScratchI8(size_t n) { return reinterpret_cast<i8*>(g_dev.d_u8(n)); }

// ---- round-batched upload arena (NOVEL 2026-08-02, NVIDIA-docs sweep) -----------------------
// The round loop used to issue ~70 SYNCHRONOUS pageable cudaMemcpys per round (34 x 32 B PRF
// keys + 36 x 96 B opgen seed/midstate blocks, profile 1), each a full submit+wait against the
// stream, when only the per-round root D2H is an algorithmically required sync -- every one of
// those blocks is derivable at round start from seed_r. So: pack them all into ONE pinned
// staging buffer and issue ONE cudaMemcpyAsync on the legacy stream. Ordering is unchanged:
// the copy is stream-ordered ahead of every consumer kernel (legacy stream), and the
// cudaStreamPerThread GEMMs synchronize with the legacy stream. Source lifetime is safe: the
// pinned buffer is only rewritten at the NEXT round's begin(), after the round-root sync D2H
// has drained the stream. reserve() must be called before the first stage() of a round -- a
// mid-round grow would invalidate already-returned device pointers, so overflow is fatal.
// Gate BTX_RC_BATCH_UPLOAD (default ON; =0 = legacy per-call sync path). Byte-exact:
// identical bytes land at the addresses the identical kernel sequence reads.
struct UploadArena {
    u8* h = nullptr; u8* d = nullptr; size_t cap = 0, cur = 0;
    void reserve(size_t n) {
        if (n <= cap) return;
        if (h) { cudaFreeHost(h); cudaFree(d); }
        cap = (n + 4095) & ~size_t(4095);
        GCK(cudaHostAlloc(&h, cap, cudaHostAllocDefault));
        GCK(cudaMalloc(&d, cap));
    }
    void begin() { cur = 0; }
    // Copies src into the pinned buffer; returns the DEVICE address it lands at after flush().
    // 32 B slot granularity keeps the +32 opgen midstate ctx u32-aligned.
    u8* stage(const void* src, size_t n) {
        const size_t off = cur;
        cur += (n + 31) & ~size_t(31);
        if (cur > cap) { printf("!! upload arena overflow (%zu > %zu)\n", cur, cap); exit(2); }
        std::memcpy(h + off, src, n);
        return d + off;
    }
    void flush() { if (cur) GCK(cudaMemcpyAsync(d, h, cur, cudaMemcpyHostToDevice, 0)); }
};
static thread_local UploadArena g_up;
// 0 = off (legacy per-call sync copies), 1 = FULL (PRF keys + opgen ctx blocks), 2 = PRF-ONLY.
// MEASURED DEAD 2026-08-02 (5090, mode 1, locked 2600, x2 interleaved): monotone SLOWER --
// off 958.9/968.9 ms, PRF-only 967.8/973.5, full 979.3/983.8. Even the zero-added-host-work
// PRF-only arm loses, so the ~280 sync pageable copies per episode are NOT a cost on this
// card: the launch queue runs deep enough that the GPU never starves during a sync-copy
// drain, and anything moved to the idle round boundary is a pure loss. Default OFF. Kept
// (gated) because CUDA-graph/PDL work would need graph-legal copies -- re-A/B there, and on
// wider cards (B200-class) where host pacing may differ.
static int BatchUploadLevel() {
    static const int lvl = [] {
        const char* v = getenv("BTX_RC_BATCH_UPLOAD");
        return v ? atoi(v) : 0;   // default OFF (measured dead, see above)
    }();
    return lvl;
}

// The attribution syncs (~80/episode) exist ONLY to make the per-stage timers meaningful.
// They are pure overhead on the fast path, so they are opt-in: RC_PROFILE=1 for a breakdown,
// unset for speed. Without them the g_t_* numbers are not attributable (kernels are async).
static bool g_prof = (getenv("RC_PROFILE") != nullptr);
static inline void prof_sync() { if (g_prof) GCK(cudaDeviceSynchronize()); }

// staged variant: the 32 B PRF key is already device-resident (round upload arena)
// BTX_RC_EXTRACT_PACK: 0 = legacy byte stores, 1 = eight u32 stores, 2 = two int4 stores
// (default). Separate template instantiations so the revert path keeps its exact old SASS.
static int ExtractPackMode()
{
    static const int m = [] {
        const char* v = getenv("BTX_RC_EXTRACT_PACK");
        if (v == nullptr) return 2;               // default: fewest store transactions
        const int n = atoi(v);
        return (n >= 0 && n <= 2) ? n : 2;
    }();
    return m;
}
// BTX_RC_SCALE_W2C: 1 (default) = block-2 schedule of the scale SHA from the compile-time
// table; 0 = computed schedule (byte-identical either way; the goldens gate both).
static bool ScaleW2C()
{
    static const bool on = [] {
        const char* v = getenv("BTX_RC_SCALE_W2C");
        return v == nullptr || v[0] != '0';
    }();
    return on;
}
static void gpu_extract_staged(const u8* dprf, const i32* draw, i8* dout, u32 rows, u32 cols,
                               const i8* dresid = nullptr)
{
    const u32 nblk = cols / kMxBlockLen;
    const size_t ntile = (size_t)rows * nblk;
    const int T = 256;
    const u32 blocks = (u32)((ntile + T - 1) / T);
    const int pk = ExtractPackMode();
    const bool wc = ScaleW2C();
#define XT(R, P, W) k_extract_tiles<R, P, W><<<blocks, T>>>(dprf, draw, dout, rows, cols, dresid)
    if (dresid != nullptr) {
        if (pk == 2)      { if (wc) XT(true, 2, true); else XT(true, 2, false); }
        else if (pk == 1) { if (wc) XT(true, 1, true); else XT(true, 1, false); }
        else              { if (wc) XT(true, 0, true); else XT(true, 0, false); }
    } else {
        if (pk == 2)      { if (wc) XT(false, 2, true); else XT(false, 2, false); }
        else if (pk == 1) { if (wc) XT(false, 1, true); else XT(false, 1, false); }
        else              { if (wc) XT(false, 0, true); else XT(false, 0, false); }
    }
#undef XT
    GCK(cudaGetLastError());
}
static void gpu_extract(const H256& prf, const i32* draw, i8* dout, u32 rows, u32 cols, u8* dprf,
                        const i8* dresid = nullptr)
{
    GCK(cudaMemcpy(dprf, prf.data(), 32, cudaMemcpyHostToDevice));
    gpu_extract_staged(dprf, draw, dout, rows, cols, dresid);
}

static double now_s()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double g_t_d2h = 0, g_t_absorb = 0, g_t_final = 0, g_t_p3 = 0, g_t_p12 = 0;
static double g_t_opgen = 0, g_t_h2d = 0, g_t_gemm = 0, g_t_extract = 0;



// ================= GPU Merkle =========================================================
// Phase 3 measured 45.17 s of a 65.5 s episode (69%): 9.0 GiB streamed to a single-threaded
// host SHA256 at 310 MB/s. The host path is doubly pathological -- absorb() push_back's one
// byte at a time, and hash_leaf() heap-allocates a 1025-byte vector PER LEAF (2.36M per round).
//
// Leaf hashing is embarrassingly parallel (one independent SHA256d per T_leaf tile), so it maps
// the same way the MX Extract did. The tree fold is a standard level-by-level halving.
//
// Byte-exactness contract -- identical leaves in identical ORDER, then identical fold:
//   leaf i   = SHA256d(0x00 || bytes[i*T .. i*T+T))
//   pad leaf = SHA256d(0x02 || "BTX_RC_PAD")
//   node     = SHA256d(0x01 || left || right)
//   stream order per round: Z, then per layer l: X[l+1], G[l], D[l]
// Buffer sizes must be multiples of T_leaf or leaves would straddle buffers and the ORDER would
// silently differ from the host stream; asserted at runtime rather than assumed.
__device__ __constant__ u32 c_K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ u32 drotr(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

struct DSha {
    u32 s[8]; u8 buf[64]; u32 blen; u64 total;
    __device__ void init() {
        s[0]=0x6a09e667; s[1]=0xbb67ae85; s[2]=0x3c6ef372; s[3]=0xa54ff53a;
        s[4]=0x510e527f; s[5]=0x9b05688c; s[6]=0x1f83d9ab; s[7]=0x5be0cd19;
        blen = 0; total = 0;
    }
    __device__ void compress(const u8* p) {
        u32 w[64];
        #pragma unroll
        for (int i = 0; i < 16; ++i)
            w[i] = (u32(p[i*4])<<24)|(u32(p[i*4+1])<<16)|(u32(p[i*4+2])<<8)|u32(p[i*4+3]);
        #pragma unroll
        for (int i = 16; i < 64; ++i) {
            const u32 s0 = drotr(w[i-15],7) ^ drotr(w[i-15],18) ^ (w[i-15] >> 3);
            const u32 s1 = drotr(w[i-2],17) ^ drotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        u32 a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            const u32 S1 = drotr(e,6) ^ drotr(e,11) ^ drotr(e,25);
            const u32 ch = (e & f) ^ ((~e) & g);
            const u32 t1 = h + S1 + ch + c_K[i] + w[i];
            const u32 S0 = drotr(a,2) ^ drotr(a,13) ^ drotr(a,22);
            const u32 mj = (a & b) ^ (a & c) ^ (b & c);
            const u32 t2 = S0 + mj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
    }
    __device__ void update(const u8* p, u32 n) {
        total += n;
        while (n) {
            const u32 take = min(64u - blen, n);
            for (u32 i = 0; i < take; ++i) buf[blen+i] = p[i];
            blen += take; p += take; n -= take;
            if (blen == 64) { compress(buf); blen = 0; }
        }
    }
    __device__ void final(u8* out) {
        const u64 bits = total * 8;
        buf[blen++] = 0x80;
        if (blen > 56) { while (blen < 64) buf[blen++] = 0; compress(buf); blen = 0; }
        while (blen < 56) buf[blen++] = 0;
        #pragma unroll
        for (int i = 7; i >= 0; --i) buf[blen++] = u8((bits >> (i*8)) & 0xff);
        compress(buf);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i*4  ] = u8(s[i] >> 24); out[i*4+1] = u8(s[i] >> 16);
            out[i*4+2] = u8(s[i] >>  8); out[i*4+3] = u8(s[i]);
        }
    }
};
__device__ __forceinline__ void dsha256d(const u8* tag1, const u8* a, u32 na,
                                         const u8* b, u32 nb, u8* out)
{
    DSha h; h.init();
    if (tag1) h.update(tag1, 1);
    if (a) h.update(a, na);
    if (b) h.update(b, nb);
    u8 mid[32]; h.final(mid);
    DSha g; g.init(); g.update(mid, 32); g.final(out);
}
static __global__ void k_hash_leaves(const i8* __restrict__ data, u32 t_leaf, size_t nleaf, u8* out)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nleaf) return;
    const u8 tag = 0x00;
    dsha256d(&tag, (const u8*)(data + i * (size_t)t_leaf), t_leaf, nullptr, 0, out + i*32);
}
// LEAF VECTOR PATH (2026-08-10, ncu-guided, BTX_RC_LEAF_VEC default ON): the DSha path
// byte-copies every leaf through buf[64] -- ncu measured 1024 load sectors PER LEAF (one per
// byte; 67.1M sector-ops in one 65,536-leaf emit). The leaf preimage is 0x00 || t_leaf bytes
// at a t_leaf-aligned base, so the data reads can be aligned u32 loads with the one-byte tag
// offset absorbed by a PRMT byte window (selector 0x7012: out = prev.b3 || cur.b0..b2 in
// big-endian message order). Identical preimage bytes -> identical SHA256d; the goldens gate it.
// t_leaf must be a multiple of 4 (every consensus T_leaf is 64-aligned; guarded at the launch).
static __global__ void k_hash_leaves_v2(const i8* __restrict__ data, u32 t_leaf, size_t nleaf,
                                        u8* __restrict__ out)
{
    const size_t li = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nleaf) return;
    const u32* d32 = reinterpret_cast<const u32*>(data + li * (size_t)t_leaf);
    const u32 nwords = t_leaf >> 2;          // data u32 words per leaf
    const u32 nblk   = (t_leaf + 1 + 9 + 63) >> 6;   // preimage blocks incl. tag + padding
    const u64 bits   = (u64)(t_leaf + 1) * 8;

    u32 h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u32 prev = 0;                            // data word j-1 (LE); byte3 feeds the next BE word
    for (u32 b = 0; b < nblk; ++b) {
        u32 w[64];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            const u32 wi = (u32)b * 16 + (u32)t;   // message word index; covers m[4wi..4wi+3]
            u32 word;
            if (wi <= nwords) {
                // m[4wi] = (wi==0 ? tag 0x00 : data[4wi-1]); m[4wi+1..3] = data[4wi..4wi+2],
                // except the word at wi==nwords closes the data: [data_last, 0x80, 0, 0].
                const u32 cur = (wi < nwords) ? d32[wi] : 0x00000080u;  // 0x80 as byte0 (LE)
                word = __byte_perm(cur, prev, 0x7012);                  // BE: prev.b3,cur.b0..b2
                if (wi == nwords) word = (word & 0xFF000000u) | 0x00800000u;
                prev = cur;
            } else if (wi == (u32)nblk * 16 - 2) {
                word = (u32)(bits >> 32);
            } else if (wi == (u32)nblk * 16 - 1) {
                word = (u32)bits;
            } else {
                word = 0u;
            }
            w[t] = word;
        }
        #pragma unroll
        for (int t = 16; t < 64; ++t) {
            const u32 s0 = drotr(w[t-15],7) ^ drotr(w[t-15],18) ^ (w[t-15] >> 3);
            const u32 s1 = drotr(w[t-2],17) ^ drotr(w[t-2],19) ^ (w[t-2] >> 10);
            w[t] = w[t-16] + s0 + w[t-7] + s1;
        }
        u32 a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        #pragma unroll
        for (int t = 0; t < 64; ++t) {
            const u32 S1 = drotr(e,6) ^ drotr(e,11) ^ drotr(e,25);
            const u32 ch = (e & f) ^ ((~e) & g);
            const u32 t1 = hh + S1 + ch + c_K[t] + w[t];
            const u32 S0 = drotr(a,2) ^ drotr(a,13) ^ drotr(a,22);
            const u32 mj = (a & bb) ^ (a & c) ^ (bb & c);
            const u32 t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=bb; bb=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=bb; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    // second SHA256: one fixed 64-byte block over the 32-byte digest (32 || 0x80 || len 256).
    u32 w2[64];
    #pragma unroll
    for (int t = 0; t < 8; ++t) w2[t] = h[t];
    w2[8] = 0x80000000u;
    #pragma unroll
    for (int t = 9; t < 15; ++t) w2[t] = 0u;
    w2[15] = 256u;
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        const u32 s0 = drotr(w2[t-15],7) ^ drotr(w2[t-15],18) ^ (w2[t-15] >> 3);
        const u32 s1 = drotr(w2[t-2],17) ^ drotr(w2[t-2],19) ^ (w2[t-2] >> 10);
        w2[t] = w2[t-16] + s0 + w2[t-7] + s1;
    }
    u32 a=0x6a09e667,bb=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,
        e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,hh=0x5be0cd19;
    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        const u32 S1 = drotr(e,6) ^ drotr(e,11) ^ drotr(e,25);
        const u32 ch = (e & f) ^ ((~e) & g);
        const u32 t1 = hh + S1 + ch + c_K[t] + w2[t];
        const u32 S0 = drotr(a,2) ^ drotr(a,13) ^ drotr(a,22);
        const u32 mj = (a & bb) ^ (a & c) ^ (bb & c);
        const u32 t2 = S0 + mj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=bb; bb=a; a=t1+t2;
    }
    const u32 s[8] = {0x6a09e667+a, 0xbb67ae85+bb, 0x3c6ef372+c, 0xa54ff53a+d,
                      0x510e527f+e, 0x9b05688c+f, 0x1f83d9ab+g, 0x5be0cd19+hh};
    u8* op = out + li * 32;
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        op[t*4  ] = u8(s[t] >> 24); op[t*4+1] = u8(s[t] >> 16);
        op[t*4+2] = u8(s[t] >>  8); op[t*4+3] = u8(s[t]);
    }
}
static __global__ void k_fill_pad(u8* out, size_t from, size_t to, const u8* __restrict__ padh)
{
    const size_t i = from + (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= to) return;
    #pragma unroll
    for (int k = 0; k < 32; ++k) out[i*32+k] = padh[k];
}
static __global__ void k_reduce_level(const u8* __restrict__ in, u8* out, size_t nout)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nout) return;
    const u8 tag = 0x01;
    dsha256d(&tag, in + (2*i)*32, 32, in + (2*i+1)*32, 32, out + i*32);
}
// PAD-SPINE FOLD (2026-08-10, BTX_RC_FOLD_SPINE default ON): ~half the padded tree hashes
// constant all-pad subtrees (P1: 1,048,512 pad leaves of 2^21). P_l -- the level-l all-pad
// subtree root -- is a tiny host-computed chain (P_0 = pad leaf; P_{l+1} = node(P_l, P_l)),
// so the fold only ever hashes nodes touching real data: node j at input width nin_real takes
// in[2j+1] when it exists and P_l otherwise. Identical node values, identical root; the pad
// half of the tree is never materialised and k_fill_pad disappears.
static __global__ void k_reduce_level_spine(const u8* __restrict__ in, u8* __restrict__ out,
                                            size_t nout_real, size_t nin_real,
                                            const u8* __restrict__ padl)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nout_real) return;
    const u8 tag = 0x01;
    const u8* right = (2*i + 1 < nin_real) ? in + (2*i+1)*32 : padl;
    dsha256d(&tag, in + (2*i)*32, 32, right, 32, out + i*32);
}

// ================= GPU MX operand generation ==========================================
// Measured: host operand gen was 19.57 s of a 20.3 s episode (96%), while actual GPU work
// was 0.55 s. The card finished the episode in six tenths of a second and then waited on one
// CPU core to hand it 2.42 G operand elements.
//
// Same class of bug as the host Merkle, one layer down: Phase B moved the *Extract* to GPU
// (1041x) but left *operand generation* -- which uses the same ChaCha/SHA + M11 primitives --
// on the host, where it then became the bottleneck.
//
// Two streams, two different parallel shapes:
//   scale stream    -- no rejection; block b yields exactly 128 2-bit values, so index -> block
//                      is a direct mapping. One thread per block, writes its 128.
//   mantissa stream -- M11 REJECTION sampling (11/16 accepted). Each SHA block is independently
//                      computable, but a block's output OFFSET depends on how many nibbles every
//                      prior block accepted. That is a real sequential dependency, resolved the
//                      standard stream-compaction way: count per block -> exclusive scan -> write.
//                      Emission order within a block (low nibble then high, bytes ascending) is
//                      preserved exactly, which is what keeps the digest byte-exact.
__device__ __constant__ i8 c_m11_val[16];
__device__ __constant__ u8 c_m11_acc[16];

__device__ __forceinline__ void sha_stream_block(const u8* __restrict__ sb, u8 domain,
                                                 u64 blk, u8* out)
{
    DSha h; h.init();
    h.update(sb, 32);
    h.update(&domain, 1);
    u8 ble[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) ble[k] = u8(blk >> (8*k));
    h.update(ble, 8);
    h.final(out);              // single SHA256, matching the host stream
}
// Midstate-resumed stream hash: octx = [sbw(8) | st7(8)] (seed words + registers after round 7,
// host-precomputed once per operand). The 41-byte message sb||domain||LE64(blk) pads to ONE
// block whose words are: w[0..7]=seed, w[8]=domain||blk bytes 0..2, w[9]=blk bytes 3..6,
// w[10]=blk byte 7||0x80||00 00, w[11..14]=0, w[15]=328. Valid for blk < 2^24 (w[9] would go
// nonzero past that -- the caller guards); also skips DSha's byte staging entirely.
__device__ __forceinline__ void sha_stream_block_mid(const u32* __restrict__ octx, u8 domain,
                                                     u64 blk, u8* out)
{
    u32 w[64];
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = octx[i];
    w[8]  = (u32(domain) << 24) | ((u32(blk) & 0xFFu) << 16) |
            ((u32(blk >> 8) & 0xFFu) << 8) | (u32(blk >> 16) & 0xFFu);
    w[9]  = 0u; w[10] = 0x00800000u;
    w[11] = 0u; w[12] = 0u; w[13] = 0u; w[14] = 0u; w[15] = 328u;
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        const u32 a0 = drotr(w[t-15],7) ^ drotr(w[t-15],18) ^ (w[t-15] >> 3);
        const u32 a1 = drotr(w[t-2],17) ^ drotr(w[t-2],19) ^ (w[t-2] >> 10);
        w[t] = w[t-16] + a0 + w[t-7] + a1;
    }
    u32 a=octx[8],b=octx[9],c=octx[10],d=octx[11],e=octx[12],f=octx[13],g=octx[14],h=octx[15];
    #pragma unroll
    for (int t = 8; t < 64; ++t) {
        const u32 S1 = drotr(e,6) ^ drotr(e,11) ^ drotr(e,25);
        const u32 ch = (e & f) ^ ((~e) & g);
        const u32 t1 = h + S1 + ch + c_K[t] + w[t];
        const u32 S0 = drotr(a,2) ^ drotr(a,13) ^ drotr(a,22);
        const u32 mj = (a & b) ^ (a & c) ^ (b & c);
        const u32 t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    const u32 s[8] = {0x6a09e667+a, 0xbb67ae85+b, 0x3c6ef372+c, 0xa54ff53a+d,
                      0x510e527f+e, 0x9b05688c+f, 0x1f83d9ab+g, 0x5be0cd19+h};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i*4  ] = u8(s[i] >> 24); out[i*4+1] = u8(s[i] >> 16);
        out[i*4+2] = u8(s[i] >>  8); out[i*4+3] = u8(s[i]);
    }
}
// Word-emitting midstate hash: identical math to sha_stream_block_mid but returns the 8 state
// words in registers -- the callers below unpack nibbles/codes straight from words, and the
// hash cache stores words planar. Byte order is preserved exactly: output byte 4i+j of the
// digest is u8(s[i] >> (24 - 8*j)) (big-endian words), same as DSha::final wrote it.
__device__ __forceinline__ void sha_stream_block_mid_w(const u32* __restrict__ octx, u8 domain,
                                                       u64 blk, u32 s[8])
{
    u32 w[64];
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = octx[i];
    w[8]  = (u32(domain) << 24) | ((u32(blk) & 0xFFu) << 16) |
            ((u32(blk >> 8) & 0xFFu) << 8) | (u32(blk >> 16) & 0xFFu);
    w[9]  = 0u; w[10] = 0x00800000u;
    w[11] = 0u; w[12] = 0u; w[13] = 0u; w[14] = 0u; w[15] = 328u;
    #pragma unroll
    for (int t = 16; t < 64; ++t) {
        const u32 a0 = drotr(w[t-15],7) ^ drotr(w[t-15],18) ^ (w[t-15] >> 3);
        const u32 a1 = drotr(w[t-2],17) ^ drotr(w[t-2],19) ^ (w[t-2] >> 10);
        w[t] = w[t-16] + a0 + w[t-7] + a1;
    }
    u32 a=octx[8],b=octx[9],c=octx[10],d=octx[11],e=octx[12],f=octx[13],g=octx[14],h=octx[15];
    #pragma unroll
    for (int t = 8; t < 64; ++t) {
        const u32 S1 = drotr(e,6) ^ drotr(e,11) ^ drotr(e,25);
        const u32 ch = (e & f) ^ ((~e) & g);
        const u32 t1 = h + S1 + ch + c_K[t] + w[t];
        const u32 S0 = drotr(a,2) ^ drotr(a,13) ^ drotr(a,22);
        const u32 mj = (a & b) ^ (a & c) ^ (b & c);
        const u32 t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s[0]=0x6a09e667+a; s[1]=0xbb67ae85+b; s[2]=0x3c6ef372+c; s[3]=0xa54ff53a+d;
    s[4]=0x510e527f+e; s[5]=0x9b05688c+f; s[6]=0x1f83d9ab+g; s[7]=0x5be0cd19+h;
}
// Fallback (no midstate ctx): full byte-path hash, then repack to big-endian words so both
// paths hand the consumers the identical word view of the identical digest bytes.
__device__ __forceinline__ void sha_stream_block_w(const u8* __restrict__ sb, u8 domain,
                                                   u64 blk, u32 s[8])
{
    u8 hash[32];
    sha_stream_block(sb, domain, blk, hash);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        s[i] = (u32(hash[i*4]) << 24) | (u32(hash[i*4+1]) << 16) |
               (u32(hash[i*4+2]) << 8) | u32(hash[i*4+3]);
}
// digest byte j (0..31) from the word view -- the exact DSha::final byte order.
__device__ __forceinline__ u8 sha_word_byte(const u32 s[8], int j)
{
    return u8(s[j >> 2] >> (24 - 8 * (j & 3)));
}
// STORE COALESCING (2026-08-10, ncu-guided). ncu on the shipped binary showed the opgen
// kernels are L2-STORE-TRANSACTION-bound, not SHA-bound: k_scale_stream ran at 82% L2
// throughput issuing one sector op per single-byte store (3.15M for a K/V-sized operand),
// and k_mant_count issued 18.6M store sectors writing the hash cache as eight u32 stores
// per thread strided 32 B apart (every lane a different sector, every instruction). The
// fix is pure layout/packing of INTERNAL scratch and store width -- the emitted bytes and
// their order are unchanged, so the digest cannot move:
//   - scale stream packs its 128 output bytes into 32 u32s and stores them 16 B at a time
//     (byte tail only for the final partial block of an operand);
//   - the hash cache becomes word-PLANAR (hcache[plane*nb + b], 8 planes of u32): the warp's
//     lanes then write/read consecutive u32s per plane -- fully coalesced both directions.
static __global__ void k_scale_stream(const u8* __restrict__ sb, size_t count, u8* __restrict__ out,
                                      const u32* __restrict__ octx, int pk)
{
    const u64 blk = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = (size_t)blk * 128;
    if (base >= count) return;
    u32 s[8];
    if (octx) sha_stream_block_mid_w(octx, 0x65, blk, s);
    else      sha_stream_block_w(sb, 0x65, blk, s);
    if (pk && base + 128 <= count && ((reinterpret_cast<uintptr_t>(out) + base) & 15u) == 0) {
        // Full block, 16-byte aligned destination: pack 4 codes per output u32 and emit 8 int4
        // stores -- 8 store transactions where the byte loop issued 128. Identical bytes in
        // identical positions.
        int4* dst = reinterpret_cast<int4*>(out + base);
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            u32 pkw[4];
            #pragma unroll
            for (int wq = 0; wq < 4; ++wq) {
                u32 v = 0;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = q * 16 + wq * 4 + j;
                    v |= u32((sha_word_byte(s, r >> 2) >> ((r & 3) * 2)) & 0x03) << (8 * j);
                }
                pkw[wq] = v;
            }
            dst[q] = make_int4((i32)pkw[0], (i32)pkw[1], (i32)pkw[2], (i32)pkw[3]);
        }
    } else {                                        // operand tail / revert: byte-exact scalar
        for (u32 r = 0; r < 128 && base + r < count; ++r)
            out[base + r] = u8((sha_word_byte(s, r >> 2) >> ((r & 3) * 2)) & 0x03);
    }
}
// MANT HASH CACHE (2026-08-02): count and write previously EACH ran the full SHA-256 stream
// block -- every mantissa hash computed twice per operand. The count pass now stores its 32-byte
// digest (hcache, ~32B/block) and the write pass reloads it instead of re-hashing: a full SHA-256
// compress swapped for a 32-byte coalesced read. Byte-exact by construction (identical bytes,
// identical emission order); gated by BTX_RC_MANT_CACHE (default ON, =0 reverts in one binary).
// soa != 0: the hash cache is word-PLANAR -- hc32[plane * nblk + b] holds state word `plane`
// of block b's digest (8 planes of u32). A warp's lanes then hit consecutive u32s per plane:
// 8 coalesced transactions instead of 8 strided ones (every lane its own sector). The digest
// BYTES and their consumption order are identical either way (sha_word_byte == DSha byte order).
static __global__ void k_mant_count(const u8* __restrict__ sb, u32 nblk, u32* __restrict__ cnt,
                                    u8* __restrict__ hcache, const u32* __restrict__ octx, int soa)
{
    const u32 b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    u32 s[8];
    if (octx) sha_stream_block_mid_w(octx, 0x6D, b, s);
    else      sha_stream_block_w(sb, 0x6D, b, s);
    u32 c = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const u32 w = s[i];
        #pragma unroll
        for (int j = 0; j < 32; j += 4)
            c += (kM11AccMask >> ((w >> j) & 0x0F)) & 1u;
    }
    cnt[b] = c;
    if (hcache) {
        u32* hc32 = reinterpret_cast<u32*>(hcache);
        if (soa) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) hc32[(size_t)i * nblk + b] = s[i];
        } else {
            // legacy row layout (BTX_RC_OPGEN_COALESCE=0): digest bytes contiguous per block
            u32* dst = hc32 + (size_t)b * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                dst[i] = (s[i] >> 24) | ((s[i] >> 8) & 0xFF00u) |
                         ((s[i] << 8) & 0xFF0000u) | (s[i] << 24);
        }
    }
}
static __global__ void k_mant_write_dequant(const u8* __restrict__ sb, u32 nblk,
                                     const u32* __restrict__ off, size_t count,
                                     u32 cols, u32 nblkrow, const u8* __restrict__ scales,
                                     const u8* __restrict__ hcache,
                                     const u32* __restrict__ octx,
                                     i8* __restrict__ out, int soa)
{
    const u32 b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    size_t pos = off[b];
    if (pos >= count) return;
    u32 s[8];
    if (hcache) {
        const u32* hc32 = reinterpret_cast<const u32*>(hcache);
        if (soa) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) s[i] = hc32[(size_t)i * nblk + b];
        } else {
            const u32* src = hc32 + (size_t)b * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const u32 v = src[i];
                s[i] = (v >> 24) | ((v >> 8) & 0xFF00u) | ((v << 8) & 0xFF0000u) | (v << 24);
            }
        }
    } else if (octx) {
        sha_stream_block_mid_w(octx, 0x6D, b, s);
    } else {
        sha_stream_block_w(sb, 0x6D, b, s);
    }
    // POW2 strength reduction (2026-08-10): cols is a power of two at every consensus dim
    // (128 / 4096 / 16384), and `p / cols` is a u64-by-u32 divide -- a heavyweight subroutine
    // PER ACCEPTED NIBBLE (~55/thread). Shift/mask when pow2; foreign dims keep the divide.
    const bool c2  = (cols & (cols - 1u)) == 0u;
    const u32 csh  = c2 ? (31u - u32(__clz(cols))) : 0u;
    if (soa) {
        // Packed-store path: same values at the same positions, only the store WIDTH changes.
        // Walk the accepted nibbles twice -- scalar bytes until `out+pos` is 4-aligned, then
        // pack each aligned quad into one u32 store, scalar again for the tail. The nibble
        // emission order (low then high, bytes ascending) is untouched.
        u32 acc = 0, accn = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const u8 by = sha_word_byte(s, i);
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                const u8 nib = t ? u8((by >> 4) & 0x0F) : u8(by & 0x0F);
                if (!m11_accept(nib)) continue;
                const size_t p = pos + accn;
                if (p >= count) {                       // flush pending, then done
                    for (u32 q = 0; q < accn; ++q)
                        out[pos + q] = i8(u8(acc >> (8 * q)));
                    return;
                }
                const size_t row = c2 ? (p >> csh) : (p / cols);
                const u32 cw = c2 ? (u32(p) & (cols - 1u)) : u32(p - row * cols);
                const u8 e = scales[row * nblkrow + (cw >> 5)];
                const u8 v = u8(i8(i32(m11_value(nib)) * (i32{1} << e)));
                if (accn == 0 && (p & 3) == 0) {        // start an aligned quad
                    acc = v; accn = 1;
                } else if (accn > 0) {                  // extend the quad
                    acc |= u32(v) << (8 * accn);
                    if (++accn == 4) {
                        *reinterpret_cast<u32*>(out + pos) = acc;
                        pos += 4; acc = 0; accn = 0;
                    }
                } else {                                // scalar until aligned
                    out[p] = i8(v);
                    ++pos;
                }
            }
        }
        for (u32 q = 0; q < accn; ++q) out[pos + q] = i8(u8(acc >> (8 * q)));
        return;
    }
    for (int i = 0; i < 32; ++i) {                      // legacy scalar path (revert gate)
        const u8 by = sha_word_byte(s, i);
        const u8 nb[2] = { u8(by & 0x0F), u8((by >> 4) & 0x0F) };
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (m11_accept(nb[t])) {
                if (pos >= count) return;
                const size_t row = c2 ? (pos >> csh) : (pos / cols);
                const u32 cw = c2 ? (u32(pos) & (cols - 1u)) : u32(pos - row * cols);
                const u8 e = scales[row * nblkrow + (cw >> 5)];
                out[pos] = i8(i32(m11_value(nb[t])) * (i32{1} << e));
                ++pos;
            }
        }
    }
}

static __global__ void k_check_total(const u32* __restrict__ off, const u32* __restrict__ cnt,
                              u32 nb, size_t n, u32* __restrict__ flag)
{
    if ((size_t)off[nb-1] + (size_t)cnt[nb-1] < n) atomicAdd(flag, 1u);
}
static thread_local u32* g_shortfall = nullptr;   // device flag, read ONCE per episode

static std::once_flag g_m11_once;
// Opgen midstate ctx gate (see sha_stream_block_mid): [sbw(8) | st7(8)] appended after sb.
// Both streams share it -- the domain byte enters at w[8], after the checkpoint.
static bool OpgenShaMid() {
    static const bool on = [] {
        const char* v = getenv("BTX_RC_SHA_MIDSTATE");
        return v == nullptr || (v[0] != '0');   // default ON
    }();
    return on;
}
// Builds the block the opgen kernels consume: 32 B seed-bytes-LE, plus the 64 B SHA midstate
// ctx when BTX_RC_SHA_MIDSTATE is on. Returns the byte count (32 or 96). Shared by the legacy
// per-call upload path and the round-batched arena path so the bytes are identical by
// construction.
static size_t build_opgen_ctx(const H256& seed, u8 out[96])
{
    u8 sb[32]; seed_bytes_le(seed, sb);
    std::memcpy(out, sb, 32);
    if (!OpgenShaMid()) return 32;
    u32 octx[16];
    for (int i = 0; i < 8; ++i) octx[i] = be32_load(sb + 4*i);
    sha_rounds_from_h0(octx, 8, octx + 8);
    std::memcpy(out + 32, octx, sizeof(octx));
    return 96;
}
// Core expansion; dsb = device-resident ctx block from build_opgen_ctx (either the pooled
// per-call upload below, or a slot in the round upload arena).
// pool/st (2026-08-10, opgen-overlap): every kernel, the scan, and the scratch are taken from
// the CALLER'S pool and launched on the CALLER'S stream. The defaults (g_dev, legacy stream 0)
// reproduce the historical behavior bit-for-bit at every existing call site; the W-prefetch
// path passes its own disjoint pool + a non-blocking stream so its scratch lifetimes can never
// race the main chain's mark/release (the "same stream" reuse argument holds per pool+stream).
static void gpu_expand_mx_dequant_dev(const u8* dsb, u32 rows, u32 cols, i8* dout,
                                      Dev& pool = g_dev, cudaStream_t st = 0)
{
    std::call_once(g_m11_once, [] {          // __constant__ symbols are global; upload once
        i8 v[16]; u8 a[16];
        for (int i = 0; i < 16; ++i) { v[i] = (i8)kM11.value[i]; a[i] = kM11.accepted[i] ? 1u : 0u; }
        GCK(cudaMemcpyToSymbol(c_m11_val, v, 16));
        GCK(cudaMemcpyToSymbol(c_m11_acc, a, 16));
        GCK(cudaMemcpyToSymbol(c_w2sched, &kW2Sched, sizeof(kW2Sched)));
    });
    const size_t mk = pool.mark();      // recycle this call's scratch, keep the caller's

    const size_t n       = (size_t)rows * cols;
    const u32    nblkrow = cols / kMxBlockLen;
    const size_t nscale  = (size_t)rows * nblkrow;
    const u32* doctx = OpgenShaMid() ? reinterpret_cast<const u32*>(dsb + 32) : nullptr;

    // STORE COALESCING gate (see the kernel comment block): =0 reverts every opgen kernel to
    // the byte-granular stores and the row-layout hash cache in one binary.
    static const int opgen_pk = [] {
        const char* v = getenv("BTX_RC_OPGEN_COALESCE");
        return (v == nullptr || v[0] != '0') ? 1 : 0;   // default ON
    }();
    u8* dscales = pool.d_u8(nscale);
    // Guard the mid path's blk < 2^24 word-layout assumption (w[9] must stay zero).
    const u32* scale_octx = ((nscale+127)/128 < (1u<<24)) ? doctx : nullptr;
    k_scale_stream<<<(u32)(((nscale+127)/128 + 255)/256),256,0,st>>>(dsb, nscale, dscales,
                                                                     scale_octx, opgen_pk);
    GCK(cudaGetLastError());

    // 64 nibbles/block, 11/16 accepted => ~44 per block. Over-provision, then verify ON DEVICE.
    // The old loop cost THREE pipeline stalls per call -- a synchronous thrust scan plus two
    // blocking D2H reads -- purely to confirm the scan covered n. At 84 calls/episode that is 252
    // stalls, and it was the bulk of the remaining GPU idle once the profiling syncs were gated.
    // The check is NOT dropped (byte-exactness must not rest on an expectation); it sets a device
    // flag read once per episode. Margin 1.25 puts a shortfall past ~13 sigma at the ~2.3M blocks
    // a production operand needs, and a shortfall is fatal rather than silent.
    // MARGIN TIGHTENING (2026-08-02): the old 1.25x over-provision hashed ~25% more mantissa
    // blocks than the mean requirement -- at production sizes the actual 16-sigma guard is a few
    // HUNDRED blocks (accepted/block ~ N(44, 13.75); sigma_total/44 blocks), not 25%. Byte-exact:
    // the write pass consumes exactly n in identical order regardless of nb; only the count of
    // extra hashed-but-unconsumed tail blocks changes. The shortfall flag stays a loud fatal, so
    // an (astronomically unlikely, ~1e-57) shortage refuses rather than forking the digest.
    static const bool mant_tight = [] {
        const char* v = getenv("BTX_RC_MANT_TIGHT");
        return v == nullptr || (v[0] != '0');   // default ON
    }();
    const double mean_blocks = (double)n * 16.0 / (11.0 * 64.0);
    const size_t nb = mant_tight
        ? (size_t)(mean_blocks + 16.0 * std::sqrt(mean_blocks * 13.75) / 44.0) + 64
        : (size_t)(mean_blocks * 1.25) + 64;
    u32* dcnt = (u32*)pool.get(nb * sizeof(u32));
    u32* doff = (u32*)pool.get(nb * sizeof(u32));
    static const bool mant_cache = [] {
        const char* v = getenv("BTX_RC_MANT_CACHE");
        return v == nullptr || (v[0] != '0');   // default ON
    }();
    const u32* mant_octx = (nb < (1u<<24)) ? doctx : nullptr;   // blk<2^24 word-layout guard
    u8* dhash = mant_cache ? pool.d_u8(nb * 32) : nullptr;
    k_mant_count<<<(u32)((nb+255)/256),256,0,st>>>(dsb, (u32)nb, dcnt, dhash, mant_octx, opgen_pk);
    GCK(cudaGetLastError());
    size_t tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, dcnt, doff, (int)nb);  // sizing only
    if (!tmp_bytes) tmp_bytes = 1;
    void* dtmp = pool.get(tmp_bytes);
    cub::DeviceScan::ExclusiveSum(dtmp, tmp_bytes, dcnt, doff, (int)nb, st); // async, unlike thrust
    k_check_total<<<1,1,0,st>>>(doff, dcnt, (u32)nb, n, g_shortfall);
    GCK(cudaGetLastError());
    k_mant_write_dequant<<<(u32)((nb+255)/256),256,0,st>>>(dsb, (u32)nb, doff, n, cols, nblkrow,
                                                           dscales, dhash, mant_octx, dout,
                                                           opgen_pk);
    GCK(cudaGetLastError());
    pool.release_to(mk);
}
// Legacy per-call path: build the ctx block, sync-upload it into a pooled buffer, expand.
static void gpu_expand_mx_dequant(const H256& seed, u32 rows, u32 cols, i8* dout,
                                  Dev& pool = g_dev, cudaStream_t st = 0)
{
    const size_t mk = pool.mark();
    u8 host[96];
    const size_t csz = build_opgen_ctx(seed, host);
    u8* dsb = pool.d_u8(96);
    GCK(cudaMemcpy(dsb, host, csz, cudaMemcpyHostToDevice));
    gpu_expand_mx_dequant_dev(dsb, rows, cols, dout, pool, st);
    pool.release_to(mk);
}


// ---- streaming-penalty bench (mode 2) ------------------------------------------------------
// The v4.5 coupled bank is EPOCH-DERIVED, not fetched: DeriveCoupledBankPage(header,h,page) is
// ExpandMxDequantInt8(page_seed, lobe_width, lobe_width) -- the identical MX expansion this file
// already reproduces byte-exact. So Resident vs Streamed is not a PCIe question:
//   Resident  bank cached in VRAM, derived once per EPOCH, then READ for every nonce.
//   Streamed  bank does not fit, so every page use RE-DERIVES the page.
// The penalty is therefore (derive one page) / (read one page), scaled by the page-uses per
// episode. Under the full-bank schedule that is barriers*lobes*pages_per = 8*8*12 = 768.
// This matters because kRCPackedBankTargetGiB[] = {40,56,72,96} all exceed a 32 GiB 5090, so a
// consumer card is pushed onto the Streamed path by construction.
static __global__ void k_touch_page(const i8* __restrict__ pg, size_t n4, i32* __restrict__ out)
{
    const int4* v = reinterpret_cast<const int4*>(pg);
    i32 acc = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        const int4 q = v[i];
        acc += q.x ^ q.y ^ q.z ^ q.w;          // consume so the loads cannot be elided
    }
    atomicAdd(out, acc);
}

static void stream_penalty_bench(const H256& sigma, u32 lobe_width, u32 page_uses, int reps)
{
    const size_t n = (size_t)lobe_width * lobe_width;
    printf("  page %u x %u = %.1f MiB int8 | %u page-uses/episode (8 barriers x 8 lobes x 12)\n\n",
           lobe_width, lobe_width, n / 1048576.0, page_uses);

    // The READ arm must come from VRAM, not L2. A single 64 MiB page fits the 5090's ~96 MB L2
    // and measured 3278 GB/s -- above the card's memory bandwidth, i.e. a cache artifact that
    // inflated the penalty. A real 40+ GiB bank is never L2-resident, so read a ring of pages
    // well past L2 and touch a different one each iteration.
    const u32 kRing = 8;                       // 8 x 64 MiB = 512 MiB >> any L2
    i8* dring = g_dev.d_i8(n * kRing);
    i8* dpage = dring;
    i32* dsink; GCK(cudaMalloc(&dsink, 4)); GCK(cudaMemset(dsink, 0, 4));
    const H256 seed = derive_prf_key(sigma);

    gpu_expand_mx_dequant(seed, lobe_width, lobe_width, dpage);   // warm up
    GCK(cudaDeviceSynchronize());

    double t0 = now_s();
    for (int i = 0; i < reps; ++i) gpu_expand_mx_dequant(seed, lobe_width, lobe_width, dpage);
    GCK(cudaDeviceSynchronize());
    const double derive_ms = (now_s() - t0) * 1000.0 / reps;

    for (u32 r = 1; r < kRing; ++r)            // fill the ring so every page is real data
        GCK(cudaMemcpy(dring + (size_t)r * n, dring, n, cudaMemcpyDeviceToDevice));
    const u32 blocks = 4096;
    k_touch_page<<<blocks,256>>>(dring, n / 16, dsink); GCK(cudaDeviceSynchronize());
    t0 = now_s();
    for (int i = 0; i < reps; ++i)
        k_touch_page<<<blocks,256>>>(dring + (size_t)(i % kRing) * n, n / 16, dsink);
    GCK(cudaDeviceSynchronize());
    const double read_ms = (now_s() - t0) * 1000.0 / reps;

    printf("  derive one page (STREAMED path) : %8.3f ms   (%.1f G elem/s)\n",
           derive_ms, n / (derive_ms / 1000.0) / 1e9);
    printf("  read   one page (RESIDENT path) : %8.3f ms   (%.0f GB/s, %u-page ring vs L2)\n",
           read_ms, n / (read_ms / 1000.0) / 1e9, kRing);
    printf("  >> STREAMING PENALTY            : %8.1fx per page use\n\n", derive_ms / read_ms);
    printf("  per episode at %u page-uses:\n", page_uses);
    printf("    Resident (bank cached in VRAM): %8.1f ms\n", read_ms * page_uses);
    printf("    Streamed (re-derive each use) : %8.1f ms\n", derive_ms * page_uses);
    printf("    => a card that cannot hold the bank pays %.1fx on the coupled phase alone\n",
           derive_ms / read_ms);
    cudaFree(dsink);
}

// ---- W-pair opgen prefetch (NOVEL 2026-08-10, opgen-overlap) --------------------------------
// RC_PROFILE on the CUTLASS binary (locked 2600, P1 dims) puts MX operand gen at 180 ms of an
// 877 ms episode -- 20.5%, nearly all of it the 16 per-layer (W_up, W_down) expansions that sit
// SERIALLY between one layer's extract and the next layer's GEMM. But W_l depends only on
// seed_r (never on the layer chain), so the whole round's W opgen is critical-path work only by
// SCHEDULE, not by data. This ring expands W pairs on a dedicated NON-BLOCKING stream while the
// main chain runs GEMM+extract, handing each layer a ready pair through cudaEvents.
//
// Why this dodges every dead overlap lever in the ledger:
//   - row-strip pipelining (+34%) sliced the GEMMs (traffic xS, worse M/S efficiency) -- here
//     the GEMM/extract kernels and their sizes are UNTOUCHED; only W expansion moves streams.
//   - PDL (+29%) parked spinning CTAs on SMs -- here the side stream does real work, no spins.
//   - batch upload (dead) moved HOST glue -- here the moved work is ~40 ms/round of DEVICE SHA.
//   - the -25% "PTS tax" was the --default-stream per-thread BUILD FLAG -- this uses an explicit
//     cudaStreamNonBlocking stream on the standard build; bare launches stay on legacy stream 0.
// Co-residency pre-check (ptxas sm_120): opgen kernels are 56-63 regs / 0 smem; extract runs 72-80
// regs at ~50% occupancy (register headroom for 1-2 opgen CTAs/SM), and the CUTLASS GEMM CTAs are
// smem-bound (~99 KB) leaving register+warp slots free -- the side stream can physically co-run.
//
// BYTE-EXACT by construction: identical expansion kernels, identical seeds, identical output
// buffers; only the launch stream and WHEN it runs differ. Every consumer (the layer's GEMMs)
// waits on the slot's `ready` event; the producer waits on `consumed` before overwriting. Events
// bind correctly because enqueue order is interleaved host-side: produce(l+N) is enqueued only
// AFTER the layer-l consumer waits are (so a slot's `ready` re-record can never precede the wait
// that needs the older record). Gates: BTX_RC_OVERLAP (default **OFF** -- see OverlapLevel(),
// which returns 0 unless the env var is set; this comment said "default ON" until 2026-08-21,
// which was wrong from the moment the autopsy below flipped it), BTX_RC_OVERLAP_RING (slots,
// default 3 = +~400 MB at P1 dims).
// RC_PROFILE forces the sequential path so the per-stage timers keep their meaning.
//
// DEADLOCK WARNING (2026-08-21): BTX_RC_OVERLAP=1 combined with MORE THAN ONE EPISODE IN
// FLIGHT hangs hard -- observed sitting 2h40m at 100% util / 100 W before it was killed. The
// ring's slot state is `static thread_local` (g_dev_og / g_sog below) while the ready/consumed/
// staged handshake assumes a single consumer walking the layer chain in order, so two concurrent
// episodes cross-couple the events and neither can make progress. Shipped rigs are unaffected
// because the default is OFF and the miner runs one episode at a time; it bites anyone pairing
// this ring with the bench's `conc` argument. Fix the thread_local sharing before re-enabling
// on wider silicon. (Concurrency is not the reason to: conc=2 measured -12.9% on a 5090,
// order-interleaved, because the GEMMs already sit at 90-96% of tensor peak.)
static thread_local Dev g_dev_og;                 // side-pool: all g_sog scratch, no lifetime races
static thread_local cudaStream_t g_sog = nullptr; // non-blocking opgen stream
// hctx/dctx: PINNED staging + device landing for the two 96 B opgen ctx blocks. The producer
// must NOT use a synchronous cudaMemcpy -- its legacy-stream semantics would block the host on
// the main chain's queue depth and destroy the ring's lead. cudaMemcpyAsync(pinned -> dctx,
// g_sog) is fully stream-ordered and free of legacy coupling. The pinned slot is rewritten only
// after the slot's `consumed` wait, long after the prior copy drained.
struct WSlot {
    i8* wu = nullptr; i8* wd = nullptr;
    u8* hctx = nullptr;                           // pinned, 192 B: [0..95] W_up ctx, [96..191] W_dn
    u8* dctx = nullptr;                           // device, 192 B, same layout
    // ready:    producer -> consumer (slot's W pair is expanded)
    // consumed: consumer -> producer (both GEMMs that read the pair are enqueued; device-side gate)
    // staged:   copies drained (host-side gate: hctx may be rewritten). Host-syncing `staged`
    //           instead of `consumed` is what keeps the host from pacing itself to the GPU.
    cudaEvent_t ready{}, consumed{}, staged{};
};
static thread_local WSlot  g_wring[8];
static thread_local int    g_wring_n = 0;         // live slots (0 = ring unavailable/disabled)
static thread_local size_t g_wring_sz = 0;        // bytes per W buffer the ring was sized for
// MEASURED DEAD (2026-08-10, locked 2600, 15 reps/arm, byte-exact all arms): ring-only
// 1061.7 ms vs base 881.0/882.2 = +20% SLOWER; ring=2 1050.6, ring=6 1003.5,
// CUDA_DEVICE_MAX_CONNECTIONS=32 and BTX_RC_MANT_CACHE=0 both no help. The ledger's
// device-filling physics held: the side stream's full-device expansion grids ALTERNATE with
// the main chain at kernel granularity instead of co-running (extract's 3 CTAs/SM at 72-80
// regs leave no register room for a 56-reg opgen CTA; only GEMM windows could host one), and
// the per-layer ready-waits turn every alternation into a bubble. NINTH overlap-class kill --
// scheduling cannot hide opgen on this card. What DID come out of the autopsy: ncu showed the
// opgen kernels are L2-store-transaction-bound, not SHA-bound (the store-coalescing lever,
// BTX_RC_OPGEN_COALESCE). Ring kept compiled + gated for wider silicon (B200-class), where
// util headroom is real; default OFF.
static int OverlapLevel()
{
    static const int lvl = [] {
        const char* v = getenv("BTX_RC_OVERLAP");
        return v ? atoi(v) : 0;                   // default OFF (measured dead, see above)
    }();
    return lvl;
}
// Ensure the ring exists at `wbytes` per buffer. Returns live slot count (0 = fall back to the
// sequential path). Allocation failure disables the ring for the process -- overlap is an
// optimization, never a requirement, and a smaller card must keep mining exactly as before.
static int WRingEnsure(size_t wbytes)
{
    static thread_local bool dead = false;
    if (dead) return 0;
    int want = 3;
    if (const char* v = getenv("BTX_RC_OVERLAP_RING")) {
        const int n = atoi(v);
        if (n >= 2 && n <= 8) want = n;
    }
    if (g_wring_n == want && g_wring_sz == wbytes) return g_wring_n;
    for (int i = 0; i < g_wring_n; ++i) {         // resize: params changed (toy vs production)
        cudaFree(g_wring[i].wu); cudaFree(g_wring[i].wd);
        cudaFreeHost(g_wring[i].hctx); cudaFree(g_wring[i].dctx);
        cudaEventDestroy(g_wring[i].ready); cudaEventDestroy(g_wring[i].consumed);
        cudaEventDestroy(g_wring[i].staged);
        g_wring[i] = WSlot{};
    }
    g_wring_n = 0; g_wring_sz = 0;
    if (!g_sog && cudaStreamCreateWithFlags(&g_sog, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError(); g_sog = nullptr; dead = true;
        printf("[rc-overlap] side stream unavailable; keeping the sequential schedule\n");
        return 0;
    }
    auto free_slot = [](WSlot& s) {
        if (s.wu) cudaFree(s.wu);
        if (s.wd) cudaFree(s.wd);
        if (s.hctx) cudaFreeHost(s.hctx);
        if (s.dctx) cudaFree(s.dctx);
        if (s.ready) cudaEventDestroy(s.ready);
        if (s.consumed) cudaEventDestroy(s.consumed);
        if (s.staged) cudaEventDestroy(s.staged);
        s = WSlot{};
    };
    for (int i = 0; i < want; ++i) {
        WSlot& s = g_wring[i];
        if (cudaMalloc(&s.wu, wbytes) != cudaSuccess ||
            cudaMalloc(&s.wd, wbytes) != cudaSuccess ||
            cudaHostAlloc(&s.hctx, 192, cudaHostAllocDefault) != cudaSuccess ||
            cudaMalloc(&s.dctx, 192) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.ready, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.consumed, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&s.staged, cudaEventDisableTiming) != cudaSuccess) {
            cudaGetLastError();
            for (int j = 0; j <= i; ++j) free_slot(g_wring[j]);
            dead = true;
            printf("[rc-overlap] ring alloc failed (%zu B/slot); keeping the sequential schedule\n",
                   wbytes);
            return 0;
        }
    }
    g_wring_n = want; g_wring_sz = wbytes;
    return g_wring_n;
}

// Full episode on GPU; Merkle stays on host (his own docs put phase 3 at <5% of the work).
// roots_out (optional): the per-round Merkle roots the digest commits over -- already
// materialized host-side (one 32 B D2H per round), so exporting them is free. The witness
// path needs them; computing them via the CPU reference instead is hours at Profile-1 dims.
static H256 run_episode_gpu_roots(const H256& sigma, const Params& p, std::vector<H256>* roots_out)
{
    // Episode-scoped mark: EVERYTHING this call takes goes back to the pool on the way out.
    // Without this dtmp32 (1.5 GiB) leaked per episode and 26 back-to-back runs OOM'd a 32 GiB card.
    if (!g_shortfall) GCK(cudaMalloc(&g_shortfall, 4));
    GCK(cudaMemset(g_shortfall, 0, 4));
    const size_t ep_mark = g_dev.mark();
    Dev& dv = g_dev; u8* dprf = g_dev.d_u8(32);
#if MATADOR_USE_CUBLASLT
    static std::once_flag lt_once;
    std::call_once(lt_once, [] { cublasLtCreate(&g_lt); });   // handle is thread-safe
    if (!g_ws) GCK(cudaMalloc(&g_ws, g_wsz));                 // workspace must NOT be shared
#endif
    GuardInt32Bound("phase2 up (k=d_model)", p.d_model);
    GuardInt32Bound("phase2 down (k=d_ff)", p.d_ff);
    i32* dtmp32 = g_dev.d_i32((size_t)p.n_q*p.n_ctx);
    std::vector<H256> roots(p.rounds);
    H256 seed_r = tagged_u32(kRoundTag, sigma, 0);

    // Config W: K/V and the single (W_up,W_down) pair are sigma-derived = EPISODE-invariant.
    // Expand them ONCE here (before round_mark, so release_to(round_mark) can't recycle them)
    // instead of once per round -- 7/8 of that opgen was redundant at DC dims.
    i8 *dK_ep = nullptr, *dV_ep = nullptr, *dWu_ep = nullptr, *dWd_ep = nullptr;
    if (p.share_ep) {
        const size_t nKV_ep = (size_t)p.n_ctx * p.d_head;
        const size_t wupsz_ep = (size_t)p.d_model * p.d_ff;
        dK_ep  = g_dev.d_i8(nKV_ep);  dV_ep  = g_dev.d_i8(nKV_ep);
        dWu_ep = g_dev.d_i8(wupsz_ep); dWd_ep = g_dev.d_i8(wupsz_ep);
        const double _e0 = now_s();
        gpu_expand_mx_dequant(derive_operand_seed(sigma,"BTX_RC_KV_K_V1"), p.n_ctx, p.d_head, dK_ep);
        gpu_expand_mx_dequant(derive_operand_seed(sigma,"BTX_RC_KV_V_V1"), p.n_ctx, p.d_head, dV_ep);
        gpu_expand_mx_dequant(derive_operand_seed(sigma,"BTX_RC_WUP_V1"), p.d_model, p.d_ff, dWu_ep);
        gpu_expand_mx_dequant(derive_operand_seed(sigma,"BTX_RC_WDN_V1"), p.d_ff, p.d_model, dWd_ep);
        prof_sync();
        g_t_opgen += now_s() - _e0;
    }

    const size_t round_mark = g_dev.mark();
    for (u32 r = 0; r < p.rounds; ++r) {
        if (r > 0) seed_r = tagged_u32(kRoundTag, roots[r-1], r);
        const double _p12s = now_s();

        // ---------------- phase 1 ----------------
        // Q always per-round; K/V episode-wide (sigma) under Config W, else per-round.
        const H256& kv_seed = p.share_ep ? sigma : seed_r;
        const H256 prf_S = derive_prf_key(derive_operand_seed(seed_r, "BTX_RC_PRF_S_V1"));
        const H256 prf_Z = derive_prf_key(derive_operand_seed(seed_r, "BTX_RC_PRF_Z_V1"));
        // Layer PRF keys derived here rather than at the top of phase 2 (host-only hoist,
        // byte-neutral) so the upload arena can stage the WHOLE round in one shot.
        std::vector<H256> prf_up(p.L_lyr), prf_dn(p.L_lyr);
        for (u32 l = 0; l < p.L_lyr; ++l) {
            char tag[40];
            std::snprintf(tag,sizeof(tag),"BTX_RC_PRF_UP_%u_V1",l);
            prf_up[l] = derive_prf_key(derive_operand_seed(seed_r,tag));
            std::snprintf(tag,sizeof(tag),"BTX_RC_PRF_DN_%u_V1",l);
            prf_dn[l] = derive_prf_key(derive_operand_seed(seed_r,tag));
        }

        // Round-batched upload (see UploadArena): stage this round's PRF keys (and, at level 1,
        // the opgen ctx blocks), then ONE pinned async H2D replaces the per-call sync memcpys.
        const int  bu       = BatchUploadLevel();
        const bool batch_up = bu >= 1;         // PRF keys staged (levels 1 and 2)
        const bool batch_og = bu == 1;         // opgen ctx blocks staged (level 1 only)
        u8 *dp_S = nullptr, *dp_Z = nullptr, *ds_Q = nullptr, *ds_K = nullptr, *ds_V = nullptr;
        std::vector<u8*> dp_up(p.L_lyr, nullptr), dp_dn(p.L_lyr, nullptr);
        std::vector<u8*> ds_X0, ds_Wu(p.L_lyr, nullptr), ds_Wd(p.L_lyr, nullptr);
        if (batch_up) {
            const double _u0 = now_s();
            const u32 n_x0 = p.rowblock_x0 ? p.b_seq/32 : 1;
            const u32 n_og = batch_og
                ? 1 + (p.share_ep ? 0 : 2) + n_x0 + (p.share_ep ? 0 : 2*p.L_lyr) : 0;
            g_up.reserve((size_t)(2 + 2*p.L_lyr)*32 + (size_t)n_og*96);
            g_up.begin();
            dp_S = g_up.stage(prf_S.data(), 32);
            dp_Z = g_up.stage(prf_Z.data(), 32);
            for (u32 l = 0; l < p.L_lyr; ++l) {
                dp_up[l] = g_up.stage(prf_up[l].data(), 32);
                dp_dn[l] = g_up.stage(prf_dn[l].data(), 32);
            }
            if (batch_og) {
                u8 ctx[96]; size_t csz;
                csz = build_opgen_ctx(derive_operand_seed(seed_r,"BTX_RC_Q_V1"), ctx);
                ds_Q = g_up.stage(ctx, csz);
                if (!p.share_ep) {
                    csz = build_opgen_ctx(derive_operand_seed(kv_seed,"BTX_RC_KV_K_V1"), ctx);
                    ds_K = g_up.stage(ctx, csz);
                    csz = build_opgen_ctx(derive_operand_seed(kv_seed,"BTX_RC_KV_V_V1"), ctx);
                    ds_V = g_up.stage(ctx, csz);
                }
                const H256 seed_x0 = derive_operand_seed(seed_r,"BTX_RC_X0_V1");
                ds_X0.resize(n_x0);
                if (p.rowblock_x0) {
                    for (u32 b = 0; b < n_x0; ++b) {
                        csz = build_opgen_ctx(tagged_u32("BTX_RC_X0_ROW_BLOCK_V1", seed_x0, b), ctx);
                        ds_X0[b] = g_up.stage(ctx, csz);
                    }
                } else {
                    csz = build_opgen_ctx(seed_x0, ctx);
                    ds_X0[0] = g_up.stage(ctx, csz);
                }
                if (!p.share_ep) for (u32 l = 0; l < p.L_lyr; ++l) {
                    char tag[40];
                    std::snprintf(tag,sizeof(tag),"BTX_RC_WUP_%u_V1",l);
                    csz = build_opgen_ctx(derive_operand_seed(seed_r,tag), ctx);
                    ds_Wu[l] = g_up.stage(ctx, csz);
                    std::snprintf(tag,sizeof(tag),"BTX_RC_WDN_%u_V1",l);
                    csz = build_opgen_ctx(derive_operand_seed(seed_r,tag), ctx);
                    ds_Wd[l] = g_up.stage(ctx, csz);
                }
            }
            g_up.flush();
            g_t_h2d += now_s() - _u0;
        }

        const size_t nQ = (size_t)p.n_q * p.d_head, nKV = (size_t)p.n_ctx * p.d_head;
        i8* dQ = dv.d_i8(nQ);
        i8* dK; i8* dV;
        const double _o0 = now_s();
        if (batch_og) gpu_expand_mx_dequant_dev(ds_Q, p.n_q, p.d_head, dQ);
        else gpu_expand_mx_dequant(derive_operand_seed(seed_r,"BTX_RC_Q_V1"), p.n_q, p.d_head, dQ);
        if (p.share_ep) {
            dK = dK_ep; dV = dV_ep;                    // episode-hoisted (sigma-derived)
        } else {
            dK = dv.d_i8(nKV); dV = dv.d_i8(nKV);
            if (batch_og) {
                gpu_expand_mx_dequant_dev(ds_K, p.n_ctx, p.d_head, dK);
                gpu_expand_mx_dequant_dev(ds_V, p.n_ctx, p.d_head, dV);
            } else {
                gpu_expand_mx_dequant(derive_operand_seed(kv_seed,"BTX_RC_KV_K_V1"), p.n_ctx, p.d_head, dK);
                gpu_expand_mx_dequant(derive_operand_seed(kv_seed,"BTX_RC_KV_V_V1"), p.n_ctx, p.d_head, dV);
            }
        }
        prof_sync();
        g_t_opgen += now_s() - _o0;

        // raw = Q . K^T  (K is n_ctx x d_head, so transpose to d_head x n_ctx)
        GuardInt32Bound("phase1 Q.K^T", p.d_head);
        // QKT-DIRECT (2026-08-10): K as expanded is (n_ctx, d_head) row-major -- EXACTLY the
        // k-contiguous B^T layout the int8 IMMA path needs for C = Q.K^T. The old path built
        // K^T with the naive transpose only for the CUTLASS backend to transpose it straight
        // back into scratch: two full passes per round producing bytes identical to dK itself.
        // Feed dK directly. Byte-exact: same operand bytes reach the same kernels; the tune
        // cache is shared with the transpose path (identical shape, identical candidates).
        // BTX_RC_QKT_DIRECT=0 reverts; cuBLASLt builds and ineligible shapes keep the old path.
        static const bool qkt_direct = [] {
            const char* v = getenv("BTX_RC_QKT_DIRECT");
            return v == nullptr || v[0] != '0';
        }();
        bool qk_done = false;
#ifdef MATADOR_HAVE_CUTLASS
        if (qkt_direct && RcGemmCutlassBackend())
            qk_done = rcgemm::gemm_i8_nn_bt_auto(dQ, dK, dtmp32, (int)p.n_q, (int)p.n_ctx,
                                                 (int)p.d_head, 0, cudaStreamPerThread)
                      == cudaSuccess;
#endif
        if (!qk_done) {
            i8* dKt = dv.d_i8(nKV);
            dim3 tb(16,16), tg((p.d_head+15)/16, (p.n_ctx+15)/16);
            k_transpose_i8<<<tg,tb>>>(dK, dKt, p.n_ctx, p.d_head); GCK(cudaGetLastError());
            gemm8(dQ, dKt, dtmp32, p.n_q, p.n_ctx, p.d_head);
        }

        i8* dS = dv.d_i8((size_t)p.n_q * p.n_ctx);
        if (batch_up) gpu_extract_staged(dp_S, dtmp32, dS, p.n_q, p.n_ctx);
        else gpu_extract(prf_S, dtmp32, dS, p.n_q, p.n_ctx, dprf);

        // accZ = S . V   (int64; grouping-independent, see header note)
        dim3 zg((p.d_head+15)/16, (p.n_q+15)/16);
        GuardInt32Bound("phase1 S.V", p.n_ctx);
        gemm8(dS, dV, dtmp32, p.n_q, p.d_head, p.n_ctx);
        i8* dZ = dv.d_i8((size_t)p.n_q * p.d_head);
        if (batch_up) gpu_extract_staged(dp_Z, dtmp32, dZ, p.n_q, p.d_head);
        else gpu_extract(prf_Z, dtmp32, dZ, p.n_q, p.d_head, dprf);

        // ---------------- phase 2 (fused FFN, forward-only) ----------------
        // bwd/wgrad are GONE in v4.6. X is ping-ponged (no X[0..L] residency) and each
        // committed X[l+1] is leaf-hashed the moment it exists -- stream order Z ‖ X[1..L]
        // is preserved and peak residency stays ~2·act + h + h32 even at DC dims (L=24).
        const size_t act = (size_t)p.b_seq * p.d_model;
        const size_t hsz = (size_t)p.b_seq * p.d_ff;
        const size_t wupsz = (size_t)p.d_model * p.d_ff;
        // prf_up/prf_dn derived at round start (upload-arena staging); see phase 1.

        // Merkle leaf plan up front so leaves can be emitted per layer.
        const size_t zsz = (size_t)p.n_q * p.d_head;
        if (zsz % p.T_leaf || act % p.T_leaf) {
            printf("!! buffer not a multiple of T_leaf=%u (z=%zu act=%zu): leaves would "
                   "straddle buffers and reorder vs the oracle stream\n", p.T_leaf, zsz, act);
            exit(2);
        }
        const size_t nleaf = (zsz + (size_t)p.L_lyr * act) / p.T_leaf;
        size_t npad = 1; while (npad < nleaf) npad <<= 1;
        // file-static, plain cudaMalloc: constant for the process, and taking it from the pool
        // would let release_to(round_mark) recycle it out from under a live pointer.
        static thread_local u8* s_dpad = nullptr;
        if (!s_dpad) {
            const H256 ph = pad_leaf_hash();
            GCK(cudaMalloc(&s_dpad, 32));
            GCK(cudaMemcpy(s_dpad, ph.data(), 32, cudaMemcpyHostToDevice));
        }
        u8* dlv  = g_dev.d_u8(npad*32);
        u8* dlv2 = g_dev.d_u8((npad > 1 ? npad/2 : 1)*32);
        size_t off = 0;
        // BTX_RC_LEAF_VEC (default ON, =0 legacy DSha byte path): word-assembled leaf hashing,
        // see k_hash_leaves_v2. Requires 4-byte-divisible T_leaf (every consensus value is).
        static const bool leaf_vec = [] {
            const char* v = getenv("BTX_RC_LEAF_VEC");
            return v == nullptr || v[0] != '0';
        }();
        // BTX_RC_LEAF_ASYNC (2026-08-10, default OFF): launch leaf emits on a non-blocking side
        // stream. The emit is a 256-CTA, 48-reg latency-bound kernel (ncu: 22.8% occupancy) that
        // only phase 3 consumes -- fire-and-forget until the Merkle fold. Unlike the dead W-ring
        // (ninth overlap kill) the MAIN chain never waits mid-round: the per-layer guard events
        // are recorded ~2 layers before they are waited on (long signaled), and the one hard sync
        // sits at the round boundary. Co-residency: 12.3K regs/CTA fits the smem-bound GEMM
        // windows (~23K regs free), not the extract windows -- the bet is emits hide under GEMMs.
        // Byte-exact: same kernels, same data, ordering enforced by events; the digest gates it.
        static const bool leaf_async = [] {
            const char* v = getenv("BTX_RC_LEAF_ASYNC");
            return v != nullptr && v[0] == '1';
        }();
        static thread_local cudaStream_t s_slv = nullptr;
        static thread_local cudaEvent_t  s_lv_src = nullptr;      // producer -> side stream
        static thread_local cudaEvent_t  s_lv_done[2] = {};       // emit done, by X-buffer parity
        static thread_local cudaEvent_t  s_lv_fin = nullptr;      // round-boundary drain
        if (leaf_async && s_slv == nullptr) {
            GCK(cudaStreamCreateWithFlags(&s_slv, cudaStreamNonBlocking));
            GCK(cudaEventCreateWithFlags(&s_lv_src, cudaEventDisableTiming));
            GCK(cudaEventCreateWithFlags(&s_lv_done[0], cudaEventDisableTiming));
            GCK(cudaEventCreateWithFlags(&s_lv_done[1], cudaEventDisableTiming));
            GCK(cudaEventCreateWithFlags(&s_lv_fin, cudaEventDisableTiming));
        }
        auto emit = [&](const i8* src, size_t bytes, int done_slot = -1) {
            const size_t n = bytes / p.T_leaf;
            cudaStream_t st = leaf_async ? s_slv : cudaStream_t(nullptr);
            if (leaf_async) {
                GCK(cudaEventRecord(s_lv_src, nullptr));          // src is ready (legacy stream)
                GCK(cudaStreamWaitEvent(s_slv, s_lv_src, 0));
            }
            if (leaf_vec && (p.T_leaf & 3u) == 0)
                k_hash_leaves_v2<<<(u32)((n+255)/256),256,0,st>>>(src, p.T_leaf, n, dlv + off*32);
            else
                k_hash_leaves<<<(u32)((n+255)/256),256,0,st>>>(src, p.T_leaf, n, dlv + off*32);
            GCK(cudaGetLastError());
            if (leaf_async && done_slot >= 0)
                GCK(cudaEventRecord(s_lv_done[done_slot], s_slv));
            off += n;
        };
        emit(dZ, zsz);

        i8*  dXp  = dv.d_i8(act);   // X[l]
        i8*  dXn  = dv.d_i8(act);   // X[l+1]
        i8*  dH   = dv.d_i8(hsz);
        i32* dh32 = dv.d_i32(hsz);
        i32* dy32 = dv.d_i32(act);
        // W-pair prefetch ring (see WRingEnsure). Not under batch_og: the arena stages opgen ctx
        // for the inline path, and mixing the two ctx sources would be two ways to feed one loop.
        const int wring = (!p.share_ep && !g_prof && OverlapLevel() >= 1 && p.L_lyr > 0 &&
                           BatchUploadLevel() == 0)
                              ? WRingEnsure(wupsz) : 0;
        std::vector<H256> wseed_u, wseed_d;   // per-layer W opgen seeds (producer inputs)
        if (wring > 0) {
            wseed_u.resize(p.L_lyr); wseed_d.resize(p.L_lyr);
            for (u32 l = 0; l < p.L_lyr; ++l) {
                char tag[40];
                std::snprintf(tag,sizeof(tag),"BTX_RC_WUP_%u_V1",l);
                wseed_u[l] = derive_operand_seed(seed_r,tag);
                std::snprintf(tag,sizeof(tag),"BTX_RC_WDN_%u_V1",l);
                wseed_d[l] = derive_operand_seed(seed_r,tag);
            }
        }
        // Enqueue layer l's W expansion on the side stream. Host-syncs `staged` (the slot's prior
        // ctx copies drained -- fired ~ring-size layers ago, so effectively instant) before
        // rewriting the pinned block; the device-side `consumed` wait orders the overwrite of the
        // W buffers themselves behind the last GEMM that read them.
        auto wring_produce = [&](u32 l) {
            WSlot& s = g_wring[l % wring];
            GCK(cudaEventSynchronize(s.staged));
            GCK(cudaStreamWaitEvent(g_sog, s.consumed, 0));
            const size_t cu = build_opgen_ctx(wseed_u[l], s.hctx);
            const size_t cd = build_opgen_ctx(wseed_d[l], s.hctx + 96);
            GCK(cudaMemcpyAsync(s.dctx,      s.hctx,      cu, cudaMemcpyHostToDevice, g_sog));
            GCK(cudaMemcpyAsync(s.dctx + 96, s.hctx + 96, cd, cudaMemcpyHostToDevice, g_sog));
            GCK(cudaEventRecord(s.staged, g_sog));
            gpu_expand_mx_dequant_dev(s.dctx,      p.d_model, p.d_ff, s.wu, g_dev_og, g_sog);
            gpu_expand_mx_dequant_dev(s.dctx + 96, p.d_ff, p.d_model, s.wd, g_dev_og, g_sog);
            GCK(cudaEventRecord(s.ready, g_sog));
        };
        if (wring > 0)                          // prime: the first ring-worth of layers
            for (u32 l = 0; l < p.L_lyr && l < (u32)wring; ++l) wring_produce(l);
        // Shared pair is episode-hoisted (dWu_ep/dWd_ep); per-layer weights reuse one buffer
        // (ring off) or hand out ring slots (ring on; the pool pair is not allocated at all).
        i8*  dWu  = p.share_ep ? dWu_ep : (wring > 0 ? nullptr : dv.d_i8(wupsz));
        i8*  dWd  = p.share_ep ? dWd_ep : (wring > 0 ? nullptr : dv.d_i8(wupsz));

        {   // X0: per-round; row-block expansion under the datacenter profile.
            const double _q0 = now_s();
            if (batch_og) {
                if (p.rowblock_x0) {
                    for (u32 b = 0; b < p.b_seq/32; ++b)
                        gpu_expand_mx_dequant_dev(ds_X0[b], 32, p.d_model,
                                                  dXp + (size_t)b*32*p.d_model);
                } else {
                    gpu_expand_mx_dequant_dev(ds_X0[0], p.b_seq, p.d_model, dXp);
                }
            } else {
                const H256 seed_x0 = derive_operand_seed(seed_r,"BTX_RC_X0_V1");
                if (p.rowblock_x0) {
                    for (u32 b = 0; b < p.b_seq/32; ++b)
                        gpu_expand_mx_dequant(tagged_u32("BTX_RC_X0_ROW_BLOCK_V1", seed_x0, b),
                                              32, p.d_model, dXp + (size_t)b*32*p.d_model);
                } else {
                    gpu_expand_mx_dequant(seed_x0, p.b_seq, p.d_model, dXp);
                }
            }
            prof_sync();
            g_t_opgen += now_s() - _q0;
        }

        const int LT = 256;
        for (u32 l = 0; l < p.L_lyr; ++l) {
            if (!p.share_ep) {
                if (wring > 0) {
                    WSlot& s = g_wring[l % wring];
                    // Binds to produce(l)'s `ready` record: produce(l+ring) for this slot is not
                    // enqueued until after this point (host order), so no later record can steal
                    // the wait. The GEMM stream is the only consumer of W.
                    GCK(cudaStreamWaitEvent(cudaStreamPerThread, s.ready, 0));
                    dWu = s.wu; dWd = s.wd;
                } else if (batch_og) {
                    const double _w0 = now_s();
                    gpu_expand_mx_dequant_dev(ds_Wu[l], p.d_model, p.d_ff, dWu);
                    gpu_expand_mx_dequant_dev(ds_Wd[l], p.d_ff, p.d_model, dWd);
                    prof_sync();
                    g_t_opgen += now_s() - _w0;
                } else {
                    const double _w0 = now_s();
                    char tag[40];
                    std::snprintf(tag,sizeof(tag),"BTX_RC_WUP_%u_V1",l);
                    gpu_expand_mx_dequant(derive_operand_seed(seed_r,tag), p.d_model, p.d_ff, dWu);
                    std::snprintf(tag,sizeof(tag),"BTX_RC_WDN_%u_V1",l);
                    gpu_expand_mx_dequant(derive_operand_seed(seed_r,tag), p.d_ff, p.d_model, dWd);
                    prof_sync();
                    g_t_opgen += now_s() - _w0;
                }
            }
            // Split GEMM vs Extract timing (RC_PROFILE) -- these two are the whole phase-2
            // hot loop, and which one dominates decides the optimization target.
            double _g0 = now_s();
            gemm8(dXp, dWu, dh32, p.b_seq, p.d_ff, p.d_model);       // H32 = X·W_up (natural)
            if (g_prof) { prof_sync(); g_t_gemm += now_s() - _g0; }
            double _x0 = now_s();
            if (batch_up) gpu_extract_staged(dp_up[l], dh32, dH, p.b_seq, p.d_ff);
            else gpu_extract(prf_up[l], dh32, dH, p.b_seq, p.d_ff, dprf);
            if (g_prof) { prof_sync(); g_t_extract += now_s() - _x0; }
            _g0 = now_s();
            // RESIDUAL-IN-GEMM (2026-08-02): beta=1 folds the += X into the down-proj GEMM's
            // existing output pass, replacing the k_add_resid full 268 MB round trip per layer.
            // Byte-exact: int32 addition, |X| <= 48, GuardInt32Bound covers the sum (precedent:
            // the coupled solver's int32 beta=1 lane). Gate BTX_RC_RESID_BETA (default ON).
            static const bool resid_beta = [] {
                const char* v = getenv("BTX_RC_RESID_BETA");
                return v == nullptr || (v[0] != '0');
            }();
            // RESIDUAL-IN-EXTRACT (2026-08-05, BTX_RC_RESID_FUSE, default ON; =0 opts out):
            // beta=0 GEMM, residual added inside the following extract. Removes the k_widen8_32
            // pass AND the beta=1 epilogue C read (~536 MB/layer at profile-1 dims).
            //
            // A/B'd at LOCKED CLOCKS (nvidia-smi -lgc 0,2600), which is mandatory here: at free
            // boost the run-to-run spread was 24 ms, larger than the effect. Locked, 4 runs each:
            //   cuBLASLt          947.0 ms  (spread 5.2)
            //   cutlass no-fuse   967.4 ms  -2.1%
            //   cutlass + fuse    949.9 ms  -0.31%   <- parity, ranges overlap
            // Byte-exact at toy (5b1bff3c) and production (42e74cd6) dims.
            static const bool resid_fuse = [] {
                const char* v = getenv("BTX_RC_RESID_FUSE");
                return v == nullptr || v[0] != '0';
            }();
            if (resid_fuse) {
                gemm8(dH, dWd, dy32, p.b_seq, p.d_model, p.d_ff);    // y = H·W_down, X added later
            } else if (resid_beta) {
                k_widen8_32<<<(u32)((act+LT-1)/LT),LT>>>(dXp, dy32, act); GCK(cudaGetLastError());
                gemm8(dH, dWd, dy32, p.b_seq, p.d_model, p.d_ff, 1); // y = H·W_down + X (fused)
            } else {
                gemm8(dH, dWd, dy32, p.b_seq, p.d_model, p.d_ff);    // y = H·W_down (natural)
                k_add_resid<<<(u32)((act+LT-1)/LT),LT>>>(dy32, dXp, act); GCK(cudaGetLastError());
            }
            if (g_prof) { prof_sync(); g_t_gemm += now_s() - _g0; }
            if (wring > 0) {   // both GEMMs that read this slot's pair are enqueued; recycle it
                WSlot& s = g_wring[l % wring];
                GCK(cudaEventRecord(s.consumed, cudaStreamPerThread));
                if (l + (u32)wring < p.L_lyr) wring_produce(l + (u32)wring);
            }
            _x0 = now_s();
            // leaf_async overwrite guard: this dn-extract WRITES dXn, whose bytes the emit of
            // layer l-2 (same parity) reads on the side stream. The event was recorded two
            // layers ago, so this wait is long signaled -- ordering without a bubble.
            if (leaf_async) GCK(cudaStreamWaitEvent(nullptr, s_lv_done[l & 1], 0));
            const i8* dn_resid = resid_fuse ? dXp : nullptr;
            if (batch_up) gpu_extract_staged(dp_dn[l], dy32, dXn, p.b_seq, p.d_model, dn_resid);
            else gpu_extract(prf_dn[l], dy32, dXn, p.b_seq, p.d_model, dprf, dn_resid);
            if (g_prof) { prof_sync(); g_t_extract += now_s() - _x0; }
            emit(dXn, act, int(l & 1));
            std::swap(dXp, dXn);
        }

        // ---------------- phase 3 (GPU Merkle fold; leaves already emitted) ----------------
        // leaf_async round-boundary drain: the Merkle fold (and next round's buffer reuse)
        // must see every side-stream emit. One wait per round, at a point where the side
        // stream is at most one emit (~0.15 ms) behind.
        if (leaf_async) {
            GCK(cudaEventRecord(s_lv_fin, s_slv));
            GCK(cudaStreamWaitEvent(nullptr, s_lv_fin, 0));
        }
        prof_sync();
        g_t_p12 += now_s() - _p12s;
        const double _p3s = now_s();
        static const bool fold_spine = [] {
            const char* v = getenv("BTX_RC_FOLD_SPINE");
            return v == nullptr || v[0] != '0';
        }();
        if (fold_spine) {
            size_t levels = 0;
            for (size_t n = npad; n > 1; n >>= 1) ++levels;
            // P_l chain: host-computed once, device-resident; grows if a deeper tree shows up.
            static thread_local u8* s_dpadlv = nullptr;
            static thread_local size_t s_padlv_n = 0;
            if (s_padlv_n < levels) {
                if (s_dpadlv) cudaFree(s_dpadlv);
                GCK(cudaMalloc(&s_dpadlv, levels * 32));
                std::vector<u8> chain(levels * 32);
                H256 p = pad_leaf_hash();
                for (size_t l = 0; l < levels; ++l) {
                    std::memcpy(chain.data() + l*32, p.data(), 32);
                    p = hash_node(p, p);
                }
                GCK(cudaMemcpy(s_dpadlv, chain.data(), chain.size(), cudaMemcpyHostToDevice));
                s_padlv_n = levels;
            }
            size_t ninr = off;                    // real width at the current level
            for (size_t l = 0; l < levels; ++l) {
                const size_t nout_real = (ninr + 1) / 2;
                k_reduce_level_spine<<<(u32)((nout_real+255)/256),256>>>(dlv, dlv2, nout_real,
                                                                         ninr, s_dpadlv + l*32);
                GCK(cudaGetLastError());
                u8* t = dlv; dlv = dlv2; dlv2 = t;
                ninr = nout_real;
            }
        } else {
            if (off < npad) {
                k_fill_pad<<<(u32)((npad-off+255)/256),256>>>(dlv, off, npad, s_dpad);
                GCK(cudaGetLastError());
            }
            for (size_t n = npad; n > 1; n >>= 1) {
                k_reduce_level<<<(u32)((n/2+255)/256),256>>>(dlv, dlv2, n/2);
                GCK(cudaGetLastError());
                u8* t = dlv; dlv = dlv2; dlv2 = t;
            }
        }
        GCK(cudaMemcpy(roots[r].data(), dlv, 32, cudaMemcpyDeviceToHost));
        prof_sync();
        g_t_p3 += now_s() - _p3s;

        // recycle the whole round into the pool; round 2+ allocates nothing. dprf/dtmp32/dpad
        // were taken before round_mark so they survive.
        g_dev.release_to(round_mark);
    }
    u32 shortfall = 0;                      // one readback per episode, not 168
    GCK(cudaMemcpy(&shortfall, g_shortfall, 4, cudaMemcpyDeviceToHost));
    if (shortfall) {
        printf("!! mantissa stream under-provisioned in %u operand(s): raise the 1.25 margin. "
               "Digest would be WRONG, refusing.\n", shortfall);
        exit(2);
    }
    g_dev.release_to(ep_mark);   // recycle the episode; steady state allocates nothing
    std::vector<u8> buf;
    buf.insert(buf.end(), (const u8*)kEpisodeTag, (const u8*)kEpisodeTag + sizeof(kEpisodeTag)-1);
    for (const auto& rt : roots) buf.insert(buf.end(), rt.begin(), rt.end());
    if (roots_out != nullptr) *roots_out = std::move(roots);
    return sha256d(buf.data(), buf.size());
}

static H256 run_episode_gpu(const H256& sigma, const Params& p)
{
    return run_episode_gpu_roots(sigma, p, nullptr);
}

static H256 from_hex_s(const char* s) {
    H256 o{}; auto v=[](char c){return u32(c<='9'?c-'0':(c|32)-'a'+10);};
    for (int i=0;i<32;++i) o[31-i]=u8((v(s[i*2])<<4)|v(s[i*2+1]));
    return o;
}

// RC_GPU_SOLVER_AS_LIB: compiled INTO matador_core (via core/matmul/cuda/rc_episode_accel.cu)
// as a byte-exact GPU episode backend -- suppress the standalone bench main() in that mode.
#ifndef RC_GPU_SOLVER_AS_LIB
int main(int argc, char** argv)
{
    // argv[1]=1 -> BASE production dims (epoch-0: n_ctx=786432, b_seq=16384, L=16, rounds=4).
    // argv[1]=3 -> DATACENTER profile-2 dims (what mainnet would activate: rounds=8, L=24,
    //              b_seq=87552, T_leaf=4096, Config W sharing + row-block X0). ~16x base MAC.
    // Toy is the golden gate; production is the only honest speed number -- at 32x32 the GPU
    // measures kernel launch latency, not work.
    const int mode = (argc > 1) ? atoi(argv[1]) : 0;
    const bool production = (mode == 1 || mode == 3);
    cudaDeviceProp pr; GCK(cudaGetDeviceProperties(&pr,0));
    if (mode == 2) {                       // streaming-penalty bench
        printf("ENC_RC coupled bank: STREAMED vs RESIDENT (%s sm_%d%d)\n\n",
               pr.name, pr.major, pr.minor);
        const u32 lw = (argc > 2) ? (u32)atoi(argv[2]) : 8192u;
        stream_penalty_bench(
            from_hex_s("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2"),
            lw, 768, (argc > 3) ? atoi(argv[3]) : 20);
        return 0;
    }
    printf("ENC_RC full episode on GPU (%s sm_%d%d) -- %s dims\n\n",
           pr.name, pr.major, pr.minor,
           mode == 3 ? "DATACENTER (profile 2)" : production ? "BASE production" : "toy");

    const H256 sigma = from_hex_s("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2");
    Params p{};
    if (production) {
        p.d_head=128; p.n_q=512; p.n_ctx=786432; p.d_model=4096; p.d_ff=16384;
        if (mode == 3) {   // datacenter profile 2 (the mainnet-default activation shape)
            p.rounds=8; p.L_lyr=24; p.b_seq=87552; p.T_leaf=4096;
            p.share_ep=true; p.rowblock_x0=true;
        } else {           // epoch-0 base
            p.rounds=4; p.L_lyr=16; p.b_seq=16384; p.T_leaf=1024;
        }
        printf("  KV %.0f MiB | W pair %.0f MiB | H %.2f GiB | %u rounds x %u layers%s\n\n",
               2.0*p.n_ctx*p.d_head/1048576.0,
               2.0*(double)p.d_model*p.d_ff/1048576.0,
               (double)p.b_seq*p.d_ff*5.0/1073741824.0, p.rounds, p.L_lyr,
               p.share_ep ? " | Config W shared + row-block X0" : "");
        // argv[2] = repeat count. One cold episode includes cuBLASLt handle creation and the M11
        // upload, and is too short to sample board power cleanly; repeats give steady state.
        const int reps = (argc > 2) ? atoi(argv[2]) : 1;
        const int conc = (argc > 3) ? atoi(argv[3]) : 1;   // concurrent episodes in flight
        if (reps > 1) printf("  running %d episodes, %d concurrent (steady state)\n\n", reps, conc);
        // The warm-up episode IS the cold one-shot replay case: it carries CUDA context +
        // cuBLASLt handle creation, arena allocation, autotune sweeps, M11 upload. Print it
        // so a single run yields BOTH the cold (validator one-shot) and warm (pool
        // steady-state) numbers.
        const double _c0 = now_s();
        H256 g = run_episode_gpu(sigma, p);          // warm up, not timed below
        GCK(cudaDeviceSynchronize());
        printf("  COLD first episode (ctx+handle+tune+alloc): %.1f ms\n",
               (now_s() - _c0) * 1000.0);
        g_t_p12 = g_t_p3 = g_t_opgen = g_t_h2d = g_t_gemm = g_t_extract = 0;

        const double t0 = now_s();
        if (conc <= 1) {
            for (int it = 0; it < reps; ++it) g = run_episode_gpu(sigma, p);
            GCK(cudaDeviceSynchronize());
        } else {
            // One host thread per in-flight episode. --default-stream per-thread gives each its
            // own stream, so the episodes genuinely overlap instead of serialising on the null
            // stream. A single episode's dependent kernel chain cannot fill a wide GPU; this is
            // the lever for the B200 sitting at 73.8% util while a 3090 rides at 99.4%.
            std::vector<std::thread> th;
            std::vector<H256> out(conc);
            const int per = (reps + conc - 1) / conc;
            for (int t = 0; t < conc; ++t)
                th.emplace_back([&, t] {
                    for (int it = 0; it < per; ++it) out[t] = run_episode_gpu(sigma, p);
                });
            for (auto& x : th) x.join();
            GCK(cudaDeviceSynchronize());
            g = out[0];
            for (int t = 1; t < conc; ++t)
                if (hex(out[t]) != hex(out[0])) {
                    printf("!! episode %d digest %s != %s -- concurrency is NOT isolated\n",
                           t, hex(out[t]).substr(0,16).c_str(), hex(out[0]).substr(0,16).c_str());
                    exit(2);
                }
        }
        const int done = (conc <= 1) ? reps : ((reps + conc - 1) / conc) * conc;
        const float ms = float((now_s() - t0) * 1000.0 / done);
        g_t_p12 /= done; g_t_p3 /= done; g_t_opgen /= done; g_t_h2d /= done;
        printf("  GPU episode : %.1f ms  (%.3f episodes/s)   digest %s\n",
               ms, 1000.0/ms, hex(g).substr(0,16).c_str());
        printf("  his solver  : ~40+ min/episode (1 CPU core, GPU 0.8%% util -- measured on this box)\n");
        printf("  speedup     : ~%.0fx vs his reference\n", (40.0*60.0*1000.0)/ms);
        const double gib = (double)p.rounds *
                           ((double)p.n_q*p.d_head + (double)p.L_lyr*p.b_seq*p.d_model) /
                           1073741824.0;
        printf("\n  WHERE THE TIME GOES\n");
        printf("    phase1+2            : %7.2f s\n", g_t_p12);
        printf("      MX operand gen (GPU): %7.2f s  (2.42 G elem on device)\n", g_t_opgen);
        printf("      H2D upload         : %7.2f s\n", g_t_h2d);
        printf("      actual GPU work    : %7.2f s\n", g_t_p12 - g_t_opgen - g_t_h2d);
        if (g_t_gemm > 0 || g_t_extract > 0) {
        printf("        phase2 GEMM (int8) : %7.2f s  (X·W_up + H·W_down, cuBLASLt IMMA)\n", g_t_gemm);
        printf("        phase2 Extract-MX  : %7.2f s  (over H:b×d_ff + X:b×d_model, ChaCha+M11)\n", g_t_extract);
        }
        printf("    phase3 GPU Merkle   : %7.2f s   <- %.1f GiB hashed on device\n", g_t_p3, gib);
        printf("\n  host baselines replaced: Merkle 45.17 s, operand gen 19.57 s\n");
        return 0;
    }

    const H256 cpu = run_episode(sigma, p);
    const H256 gpu = run_episode_gpu(sigma, p);
    const char* want = "5b1bff3c835b1c8e7816a2cccb181eb2fc30a99d97a971d73108c52a8238acd4";

    printf("  CPU oracle digest : %s  %s\n", hex(cpu).c_str(), hex(cpu)==want ? "PASS":"FAIL");
    printf("  GPU episode digest: %s  %s\n", hex(gpu).c_str(), hex(gpu)==want ? "PASS":"FAIL");
    const bool ok = (hex(gpu)==want);
    printf("\n%s\n", ok ? "PHASES C-E BYTE-EXACT -- whole episode runs on GPU"
                        : "MISMATCH -- bisect stage by stage against the CPU oracle");
    return ok ? 0 : 1;
}
#endif  // RC_GPU_SOLVER_AS_LIB
