// v4.5 ENC_RC episode -- byte-exact CPU reference. See matmul_v4_rc.h.
//
// Lifted from the validated Phase-A implementation (research/matmul-v4/bench/rc_cpu_solver.cpp),
// which reproduces numair's frozen golden b339d0ff...e43a. SHA-256 comes from our vendored
// CSHA256; ChaCha20 and the M11 table are self-contained (btx has its own, but vendoring a second
// crypto dependency for one call site is not worth the coupling to an actively churning upstream).
//
// ENDIANNESS TRAP, do not "simplify" away: uint256::GetHex() prints .data() REVERSED, and
// SeedBytesLE reverses independently again. Getting either backwards silently poisons every
// derived value -- it cost the first run of the Phase-A port.

#include <matmul/matmul_v4_rc.h>

#include <crypto/sha256.h>
#include <span.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <string>

namespace matmul::v4::rc {
namespace {

using u8 = uint8_t; using u32 = uint32_t; using u64 = uint64_t;
using i8 = int8_t;  using i32 = int32_t; using i64 = int64_t;
using H256 = std::array<u8, 32>;

// vendored CSHA256 behind the same shape the Phase-A code expects
struct Sha256 {
    CSHA256 h;
    void write(const void* d, size_t n) { h.Write(static_cast<const unsigned char*>(d), n); }
    void finalize(u8 out[32]) { h.Finalize(out); }
};
static H256 h256_from(const uint256& v) { H256 o{}; std::memcpy(o.data(), v.data(), 32); return o; }
static uint256 to_uint256(const H256& h) { return uint256{Span<const unsigned char>{h.data(), 32}}; }

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
// int8-int8 -> int64 GEMM, natural layouts: out[m x n] = A[m x k] . B[k x n] (byte-identical
// to upstream's K-panel FusedExactGemmInt64; panels exist only for FP32 accelerators).
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
// One fused 2-layer FFN (v4.6): H = Extract(X.W_up) internal; X_out = Extract(H.W_down + X).
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

// out_roots (optional): if non-null, receives the per-round Merkle roots (round 0..R-1) that the
// episode digest is SHA256d'd over. Byte-exact: same math, just copies out the intermediate that
// the witness (poolcore enc-rc-v0) needs. Digest path unchanged (gated by golden 5b1bff3c).
static H256 run_episode_impl(const H256& sigma, const EpisodeParams& p,
                             std::vector<H256>* out_roots = nullptr) {
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
            // Config W: ONE (W_up, W_down) pair, sigma-derived, shared across rounds AND layers.
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

        // ---- Phase 3: stream into Merkle (Z || X[1..L]; H never committed) ----
        Merkle m(p.T_leaf);
        m.absorb(Z);
        for (u32 l = 0; l < p.L_lyr; ++l) m.absorb(X[l+1]);
        roots[r] = m.finalize_root();
    }
    if (out_roots != nullptr) *out_roots = roots;
    std::vector<u8> buf;
    buf.insert(buf.end(), (const u8*)kEpisodeTag, (const u8*)kEpisodeTag + sizeof(kEpisodeTag)-1);
    for (const auto& rt : roots) buf.insert(buf.end(), rt.begin(), rt.end());
    return sha256d(buf.data(), buf.size());
}


} // namespace

EpisodeParams ProductionEpisodeParams()
{
    EpisodeParams p;
    p.rounds = 4; p.d_head = 128; p.n_q = 512; p.n_ctx = 786432;
    p.L_lyr = 16; p.d_model = 4096; p.d_ff = 16384; p.b_seq = 16384; p.T_leaf = 1024;
    return p;
}

EpisodeParams DatacenterEpisodeParams()
{
    // v4.6 profile-2: base intensive dims, extensive axes raised, Config W + row-block X0.
    EpisodeParams p = ProductionEpisodeParams();
    p.rounds = 8; p.L_lyr = 24; p.b_seq = 87552; p.T_leaf = 4096;
    p.share_ep = true; p.rowblock_x0 = true;
    return p;
}

EpisodeParams ActiveProfileEpisodeParams()
{
    // Profile 1 (Epoch A launch tuple, PR#97) unless explicitly overridden to the profile-2
    // datacenter shape. Unknown values fall back to profile 1 -- the launch default must never
    // silently grind the wrong (more expensive) shape.
    const char* v = std::getenv("BTX_MATMUL_RC_PROFILE");
    if (v != nullptr && v[0] == '2' && v[1] == '\0') return DatacenterEpisodeParams();
    return ProductionEpisodeParams();
}

uint8_t DeriveMxScale(const uint256& prf_key, uint32_t i, uint32_t bj)
{
    return derive_mx_scale(h256_from(prf_key), i, bj);
}

void ExtractMxTileInt64(const uint256& prf_key, uint32_t i, uint32_t bj,
                        const int64_t raw64[32], int8_t out[32])
{
    extract_mx_tile_i64(h256_from(prf_key), i, bj, raw64, out);
}

uint256 ComputeEpisodeDigest(const uint256& sigma, const EpisodeParams& params)
{
    return to_uint256(run_episode_impl(h256_from(sigma), params));
}

std::vector<uint256> ComputeEpisodeRoundRoots(const uint256& sigma, const EpisodeParams& params)
{
    std::vector<H256> roots;
    (void)run_episode_impl(h256_from(sigma), params, &roots);
    std::vector<uint256> out;
    out.reserve(roots.size());
    for (const auto& r : roots) out.push_back(to_uint256(r));
    return out;
}

} // namespace matmul::v4::rc
