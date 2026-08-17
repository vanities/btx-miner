// rc_cpu_solver.cpp -- Phase A of our own v4.5 ENC_RC solver: byte-exact CPU replication.
//
// WHY: numair's RC reference runs the whole episode on ONE CPU core (measured on pc: harness at
// 100% of a core, GPU at 0.8% util / 49 W) => ~40+ min per production episode against a ~44 ms
// GPU compute floor. That is a 100-1000x opportunity, but a fast solver is worthless unless it is
// byte-exact, so this file comes first: it reproduces his frozen goldens on CPU and then serves as
// the oracle that every GPU kernel in Phases B-E is diffed against.
//
// Ground truth captured from his own build (btx PR#89 @ bf36b7d, MakeRCHeader(42) + toy params):
//   sigma        86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2
//   seed_r       ed436b24ea7dddec31a2184a5a47dc9954708cee7d399e26700f6ab04e0931e0
//   seed_Q       4b9c8f75faa6a3cb2b9d9494ddc1b7977be9c32334e650f7fceddfd7aadeb1b1
//   seed_K       29cb0fb3a0b4e34a73d9903623a0050837b2dae6d568278135d59e86fc68987f
//   prf_S        13fdf6440502282bfe441a66aac4501b82a80594d7c794c15baf3e7243171ae8
//   scale(prf_S,0,0)=2  scale(prf_S,3,1)=0
//   extract(3,1) on raw[t]=7t-100 -> 3 -6 -2 3 6 1 4 4 4 1 -2 0 -4 3 -6 4 -6 -2 -3 -3 4 0 -1 0
//                                    -1 2 0 1 4 6 -2 0        (scale e=0, so these are raw M11 mu)
//
// v4.6 FUSED-FFN EPISODE (re-synced 2026-07-23, PR#89 @ a4bdefb): Phase 2 is now a fused 2-layer
// FFN per layer -- H = Extract(X·W_up) [b_seq×d_ff, NOT committed], X_out = Extract(H·W_down + X)
// [committed]. Backward + wgrad are GONE (no G/D, no GL seed). Round stream = Z ‖ X[1..L]. New
// per-layer tags: WUP_%u / WDN_%u / PRF_UP_%u / PRF_DN_%u. Natural contraction-major layouts (no
// transpose). The pre-fused golden b339d0ff is DEAD upstream; current toy golden:
//   episode digest 5b1bff3c835b1c8e7816a2cccb181eb2fc30a99d97a971d73108c52a8238acd4
//
// DATACENTER (profile 2 -- what mainnet would activate) semantics, gated by flags below:
// K/V + the single (W_up,W_down) pair are EPISODE-WIDE (sigma-derived, tags WUP/WDN without the
// layer index -- Config W); X0 stays per-round (seed_r) but expands as independent 32-row blocks
// seeded SHA256("BTX_RC_X0_ROW_BLOCK_V1" ‖ seed_x0 ‖ LE32(block)). prf_up/prf_dn stay per-round
// per-layer. DC dims: rounds=8 L=24 d_ff=16384 b_seq=87552 T_leaf=4096 (intensive dims epoch-0).
//
// Self-contained (own SHA256 + ChaCha20 + M11 table) so it builds anywhere with no btx deps:
//   c++ -O2 -std=c++17 rc_cpu_solver.cpp -o rc_cpu_solver && ./rc_cpu_solver

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <string>
#include <algorithm>
#include <vector>

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
// int8·int8 -> int64 GEMM, natural layouts: out[m×n] = A[m×k]·B[k×n]. Direct int64 accumulation
// is byte-identical to his K-panel FusedExactGemmInt64 (panels exist only for FP32 accelerators).
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
// One fused 2-layer FFN: H = Extract(X·W_up) [b_seq×d_ff, internal]; X_out = Extract(H·W_down + X)
// [b_seq×d_model, committed]. Residual +X folded INSIDE the single Extract accumulator.
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
        // Q always per-round (freshness); K/V episode-wide (sigma) under Config W, else per-round.
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
        // X0: per-round always; row-block expansion under the datacenter profile.
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

// ============================================================ self-test vs his pinned values
static H256 from_hex(const char* s) {
    H256 o{}; auto v = [](char c) { return u32(c <= '9' ? c - '0' : (c | 32) - 'a' + 10); };
    for (int i = 0; i < 32; ++i) o[31 - i] = u8((v(s[i*2]) << 4) | v(s[i*2+1]));
    return o;
}
static int fails = 0;
static void chk(const char* what, const std::string& got, const char* want) {
    const bool ok = got == want;
    printf("  %-22s %s\n%s", what, ok ? "PASS" : "FAIL", ok ? "" : ("      got  " + got + "\n      want " + want + "\n").c_str());
    if (!ok) ++fails;
}

int main() {
    printf("ENC_RC Phase A -- byte-exact CPU replication vs numair's pinned values\n\n");

    // seed_r = SHA256("BTX_RC_ROUND_V1" || sigma || LE32(0)); sigma taken as given (we already
    // reproduce DeriveSigma byte-exact in clean-stack).
    const H256 sigma = from_hex("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2");
    u8 rbuf[36]; std::memcpy(rbuf, sigma.data(), 32); std::memset(rbuf + 32, 0, 4);
    const H256 seed_r = sha256_tagged("BTX_RC_ROUND_V1", rbuf, 36);
    chk("seed_r", hex(seed_r), "ed436b24ea7dddec31a2184a5a47dc9954708cee7d399e26700f6ab04e0931e0");

    const H256 sQ = derive_operand_seed(seed_r, "BTX_RC_Q_V1");
    const H256 sK = derive_operand_seed(seed_r, "BTX_RC_KV_K_V1");
    const H256 sS = derive_operand_seed(seed_r, "BTX_RC_PRF_S_V1");
    chk("seed_Q", hex(sQ), "4b9c8f75faa6a3cb2b9d9494ddc1b7977be9c32334e650f7fceddfd7aadeb1b1");
    chk("seed_K", hex(sK), "29cb0fb3a0b4e34a73d9903623a0050837b2dae6d568278135d59e86fc68987f");

    const H256 prfS = derive_prf_key(sS);
    chk("prf_S", hex(prfS), "13fdf6440502282bfe441a66aac4501b82a80594d7c794c15baf3e7243171ae8");

    printf("  %-22s %s (got %u,%u want 2,0)\n", "mx scale(0,0)/(3,1)",
           (derive_mx_scale(prfS,0,0) == 2 && derive_mx_scale(prfS,3,1) == 0) ? "PASS" : "FAIL",
           derive_mx_scale(prfS,0,0), derive_mx_scale(prfS,3,1));
    if (!(derive_mx_scale(prfS,0,0) == 2 && derive_mx_scale(prfS,3,1) == 0)) ++fails;

    // Extract on a known raw tile. scale(3,1)==0, so the output is the raw M11 mantissas --
    // this exercises ChaCha tile keystream + nibble mixing + rejection sampling in isolation.
    i64 raw[32]; for (int t = 0; t < 32; ++t) raw[t] = i64(t * 7 - 100);
    i8 tile[32]; extract_mx_tile_i64(prfS, 3, 1, raw, tile);
    static const i8 want[32] = {3,-6,-2,3,6,1,4,4,4,1,-2,0,-4,3,-6,4,-6,-2,-3,-3,4,0,-1,0,-1,2,0,1,4,6,-2,0};
    bool tok = true; for (int t = 0; t < 32; ++t) if (tile[t] != want[t]) tok = false;
    printf("  %-22s %s\n", "extract(3,1) tile", tok ? "PASS" : "FAIL");
    if (!tok) {
        printf("      got  "); for (int t = 0; t < 32; ++t) printf("%d ", tile[t]);
        printf("\n      want "); for (int t = 0; t < 32; ++t) printf("%d ", want[t]);
        printf("\n"); ++fails;
    }

    // toy params: n_q=32 d_head=32 n_ctx=64
    const auto Q = expand_mx_dequant_i8(sQ, 32, 32);
    const auto K = expand_mx_dequant_i8(sK, 64, 32);
    static const i8 wantQ[16] = {32,16,16,32,16,-8,-24,16,-48,-16,-16,-32,0,-16,-24,-32};
    static const i8 wantK[16] = {0,-2,8,6,4,-12,-4,-2,6,6,-6,-2,2,8,-6,6};
    bool qok = true, kok = true;
    for (int i = 0; i < 16; ++i) { if (Q[i] != wantQ[i]) qok = false; if (K[i] != wantK[i]) kok = false; }
    printf("  %-22s %s\n", "Q[0..15] expand", qok ? "PASS" : "FAIL");
    if (!qok) { printf("      got "); for (int i=0;i<16;++i) printf("%d ", Q[i]); printf("\n"); ++fails; }
    printf("  %-22s %s\n", "K[0..15] expand", kok ? "PASS" : "FAIL");
    if (!kok) { printf("      got "); for (int i=0;i<16;++i) printf("%d ", K[i]); printf("\n"); ++fails; }

    const H256 dig = run_episode(sigma, Params{});
    chk("EPISODE DIGEST", hex(dig), "5b1bff3c835b1c8e7816a2cccb181eb2fc30a99d97a971d73108c52a8238acd4");

    printf("\n%s (%d failure%s)\n", fails ? "PRIMITIVES NOT YET BYTE-EXACT" : "ALL PRIMITIVES BYTE-EXACT",
           fails, fails == 1 ? "" : "s");
    printf("Phase A COMPLETE. Next: Phase B -- move the MX Extract to GPU, gated on this oracle.\n");
    return fails ? 1 : 0;
}
