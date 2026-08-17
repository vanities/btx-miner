// v4.6 ENC_RC COUPLED V3 -- byte-exact CPU reference (see matmul_v4_rc_coupled.h).
//
// Lifted from the validated research oracle research/matmul-v4/bench/rc_coupled_solver_v3.cpp
// (BYTE-EXACT vs upstream medium-V3 golden a4bb0cc4). Primitives duplicated from
// matmul_v4_rc.cpp's anonymous namespace on purpose -- each consensus TU stays self-contained
// so an edit in one cannot silently shift the other's bytes.

#include <matmul/matmul_v4_rc_coupled.h>

#include <matmul/matmul_pow.h>   // ComputeMatMulHeaderHash (bank template hash)
#include <primitives/block.h>
#include <crypto/sha256.h>
#include <span.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace matmul::v4::rc {
namespace {

using u8 = uint8_t; using u32 = uint32_t; using u64 = uint64_t;
using i8 = int8_t;  using i32 = int32_t; using i64 = int64_t;
using H256 = std::array<u8, 32>;

// vendored CSHA256 behind the same shape the research code expects
struct Sha256 {
    CSHA256 h;
    void write(const void* d, size_t n) { h.Write(static_cast<const unsigned char*>(d), n); }
    void finalize(u8 out[32]) { h.Finalize(out); }
};
static H256 h256_from(const uint256& v) { H256 o{}; std::memcpy(o.data(), v.data(), 32); return o; }

static void wle32(u8* p, u32 v){ for(int i=0;i<4;++i) p[i]=u8(v>>(8*i)); }
static void wle64(u8* p, u64 v){ for(int i=0;i<8;++i) p[i]=u8(v>>(8*i)); }

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


static H256 sha256d(const u8* d, size_t n) {
    Sha256 a; a.write(d,n); H256 x; a.finalize(x.data());
    Sha256 b; b.write(x.data(),32); H256 y; b.finalize(y.data()); return y;
}


static H256 sha_tag_u32(const char* tag, const H256& a, u32 x) {
    Sha256 s; s.write(tag, std::strlen(tag)); s.write(a.data(),32);
    u8 le[4]; wle32(le,x); s.write(le,4); H256 o; s.finalize(o.data()); return o;
}
static H256 sha_tag_u32u32(const char* tag, const H256& a, u32 x, u32 y) {
    Sha256 s; s.write(tag, std::strlen(tag)); s.write(a.data(),32);
    u8 le[4]; wle32(le,x); s.write(le,4); wle32(le,y); s.write(le,4);
    H256 o; s.finalize(o.data()); return o;
}

// ENC_RC_V3: independent COUP_*_V3 domain family. NB: the exchange-ROUNDS tag has no
// version suffix cycle -- its one frozen value IS "..._ROUNDS_V3" (V1/V2 never touch it).
static constexpr char kBankTag[]="BTX_RC_COUP_BANK_V3", kLobeTag[]="BTX_RC_COUP_LOBE_V3",
                      kBarrierTag[]="BTX_RC_COUP_BARRIER_V3", kPermTag[]="BTX_RC_COUP_PERM_V3",
                      kMixTag[]="BTX_RC_COUP_MIX_V3", kExtractTag[]="BTX_RC_COUP_EXTRACT_V3",
                      kFullBankTag[]="BTX_RC_COUP_FULL_BANK_V3",
                      kMatXchgTag[]="BTX_RC_COUP_MAT_XCHG_V3",
                      kMatXchgRoundsTag[]="BTX_RC_COUP_MAT_XCHG_ROUNDS_V3",
                      kEpisodeTag[]="BTX_RC_COUP_EPISODE_V3";
static constexpr u32 kMixPatterns = 2, kExchangeRows = 128;

// ---- counter-SHA XOF (ShaXof): LE32 words from SHA256(seed || LE32(ctr++)) -----------------
struct ShaXof {
    H256 seed; u32 ctr=0, pos=32; u8 blk[32]{};
    explicit ShaXof(const H256& s): seed(s) {}
    void refill(){ Sha256 h; h.write(seed.data(),32); u8 le[4]; wle32(le,ctr++); h.write(le,4);
                   h.finalize(blk); pos=0; }
    u32 next(){ if(pos+4>32) refill(); u32 v=u32(blk[pos])|(u32(blk[pos+1])<<8)|
               (u32(blk[pos+2])<<16)|(u32(blk[pos+3])<<24); pos+=4; return v; }
};

// ---- coupled toy params --------------------------------------------------------------------
using Coup = CoupParamsV3;
static u32 state_bytes(const Coup& p) { return p.lobes*p.rows_per_lobe*p.lobe_width; }

// ---- bank ----------------------------------------------------------------------------------
static H256 bank_root_seed(const H256& tmpl_hash, u32 height) {
    return sha_tag_u32(kBankTag, tmpl_hash, height);
}
static std::vector<i8> derive_bank_page(const H256& brs, u32 page, const Coup& p) {
    const H256 page_seed = sha_tag_u32(kBankTag, brs, page);
    return expand_mx_dequant_i8(page_seed, p.lobe_width, p.lobe_width);
}
// BankCommitment: DOUBLE SHA256(BankTag || page0 || page1 || ...) -- coupled.cpp
// Finalize(d1)+SHA(d1). STREAMED into one incremental SHA256 (byte-identical to hashing the
// concatenation): the old concat buffer was a second ~96 GiB transient at production dims.
static H256 bank_commitment(const std::vector<std::vector<i8>>& pages, const Coup& p) {
    Sha256 h; h.write(kBankTag, std::strlen(kBankTag));
    for (u32 i=0;i<p.bank_pages;++i) h.write(pages[i].data(), pages[i].size());
    H256 d1; h.finalize(d1.data());
    Sha256 b; b.write(d1.data(), 32);
    H256 d2; b.finalize(d2.data());
    return d2;
}
// All bank pages for one template, derived in parallel (pages are independent; each slot is
// written by exactly one thread, so assembly order is deterministic regardless of scheduling).
static std::vector<std::vector<i8>> derive_bank_pages(const H256& brs, const Coup& p) {
    std::vector<std::vector<i8>> pages(p.bank_pages);
    const u32 hw = std::max(1u, std::thread::hardware_concurrency());
    const u32 nthreads = std::min(hw, p.bank_pages);
    if (nthreads <= 1) {
        for (u32 pg=0; pg<p.bank_pages; ++pg) pages[pg] = derive_bank_page(brs, pg, p);
        return pages;
    }
    std::vector<std::thread> ts; ts.reserve(nthreads);
    for (u32 t=0;t<nthreads;++t) {
        ts.emplace_back([&, t] {
            for (u32 pg=t; pg<p.bank_pages; pg+=nthreads) pages[pg] = derive_bank_page(brs, pg, p);
        });
    }
    for (auto& th : ts) th.join();
    return pages;
}
// full-bank page schedule: episode-global Fisher-Yates over [0,bank_pages), slot-indexed.
static std::vector<u32> select_pages(u32 barrier, u32 lobe, const H256& sigma, const Coup& p) {
    const H256 perm_seed = sha_tag_u32(kFullBankTag, sigma, p.bank_pages);
    ShaXof xof(perm_seed);
    std::vector<u32> perm(p.bank_pages); for(u32 i=0;i<p.bank_pages;++i) perm[i]=i;
    for (u32 i=p.bank_pages-1;i>0;--i){ u32 j=xof.next()%(i+1); std::swap(perm[i],perm[j]); }
    const u32 P = p.pages_per_barrier_lobe;
    const u64 base = (u64(barrier)*p.lobes + lobe)*u64(P);
    std::vector<u32> out(P);
    for (u32 k=0;k<P;++k) out[k]=perm[size_t((base+k)%p.bank_pages)];
    return out;
}

// ---- lobe seeds / init / GEMM --------------------------------------------------------------
static std::vector<H256> lobe_seeds(const H256& sigma, const Coup& p) {
    std::vector<H256> o(p.lobes);
    for (u32 e=0;e<p.lobes;++e) o[e]=sha_tag_u32(kLobeTag, sigma, e);
    return o;
}
// LobeLocalGemm: MxW rows . WxW page -> MxW, int32 accumulate widened to int64 (M=1 = the V1 GEMV).
// r,k,c loop order (page rows walked sequentially, autovectorizable) ADD-accumulating into the
// caller's i64 dest -- per-element it is the SAME i32 dot-product sum (i32 addition reordering
// only; operands are extract outputs bounded |48| so the K<=8192 sum tops out ~18.9M, far from
// i32 wrap) then the same dest[c] += i64(acc) in page-schedule order. Replaces the c-inner form
// whose page[k*W+c] stride-W walk thrashed at W=8192, and the per-page partial vector.
static void lobe_gemm_add(const i8* rows, const std::vector<i8>& page, u32 M, u32 W, i64* dest) {
    std::vector<i32> accrow(W);
    for (u32 r=0;r<M;++r) {
        std::fill(accrow.begin(), accrow.end(), 0);
        const i8* rowp = rows + size_t(r)*W;
        for (u32 k=0;k<W;++k) {
            const i32 a = i32(rowp[k]);
            if (a == 0) continue;
            const i8* pk = page.data() + size_t(k)*W;
            for (u32 c=0;c<W;++c) accrow[c] += a * i32(pk[c]);
        }
        i64* dr = dest + size_t(r)*W;
        for (u32 c=0;c<W;++c) dr[c] += i64(accrow[c]);
    }
}

// ---- mix (butterfly all-to-all) ------------------------------------------------------------
// V3 (rows_per_lobe >= 32): EXPLICIT uint64 two's-complement wrap -- the post-mix bound
// exceeds int64 at M-row depth, so the ring is defined modular arithmetic, not signed adds.
static void mix_ascending(std::vector<i64>& s, u32 mask, u32 n) {
    for (u32 stage=0; (u32(1)<<stage)<n; ++stage){ u32 stride=u32(1)<<stage;
        for (u32 i=0;i<n;++i){ u32 j=i^stride; if(i>=j) continue; u32 pi=i^mask, pj=j^mask;
            u64 a=u64(s[pi]), b=u64(s[pj]); s[pi]=i64(a+b); s[pj]=i64(a-b); } }
}
static void mix_descending(std::vector<i64>& s, u32 mask, u32 n) {
    if (n<2 || (n&(n-1))) return;
    u32 bits=0; for(u32 t=n;t>1;t>>=1)++bits;
    auto rotl=[bits,n](u32 x,u32 r)->u32{ r%=bits; return ((x<<r)|(x>>(bits-r)))&(n-1); };
    for (int stage=int(bits)-1; stage>=0; --stage){ u32 stride=u32(1)<<u32(stage);
        for (u32 i=0;i<n;++i){ u32 j=i^stride; if(i>=j) continue;
            u32 pi=rotl(i^mask,3), pj=rotl(j^mask,3);
            u64 a=u64(s[pi]), b=u64(s[pj]); s[pi]=i64(a+b); s[pj]=i64(b-a); } }
}
static void all_to_all_mix(std::vector<i64>& s, const H256& sigma, u32 barrier, u32 n) {
    // material_exchange ON (default): mix_seed from the exchange tag with exchange_rows.
    const H256 mix_seed = sha_tag_u32u32(kMatXchgTag, sigma, barrier, kExchangeRows);
    ShaXof xof(mix_seed);
    const u32 mask = xof.next() & (n-1);
    if (barrier % kMixPatterns == 0) mix_ascending(s, mask, n);
    else                             mix_descending(s, mask, n);
}

// ---- V3 material-exchange rounds (digest-affecting; no-op when rounds==0) ------------------
// Per round: fold = XOR of all lanes (u64 view); seed = SHA256d(RoundsTag || sigma || LE32(b)
// || LE32(r) || LE64(fold)); XOR keystream (NextU64 = two LE u32 halves) over every lane from
// a fresh XOF(seed); then Fisher-Yates lane permutation from ANOTHER fresh XOF(seed).
static u64 xof_next_u64(ShaXof& x){ u64 lo=x.next(), hi=x.next(); return lo|(hi<<32); }
static void material_exchange_rounds(std::vector<i64>& s, const H256& sigma, u32 barrier, u32 rounds) {
    for (u32 r=0; r<rounds; ++r) {
        u64 fold=0; for (i64 v : s) fold ^= u64(v);
        std::vector<u8> pre;
        pre.insert(pre.end(), kMatXchgRoundsTag, kMatXchgRoundsTag+std::strlen(kMatXchgRoundsTag));
        u8 tail[48]; std::memcpy(tail, sigma.data(), 32);
        wle32(tail+32, barrier); wle32(tail+36, r); wle64(tail+40, fold);
        pre.insert(pre.end(), tail, tail+48);
        const H256 seed = sha256d(pre.data(), pre.size());
        { ShaXof xof(seed); for (i64& lane : s) lane = i64(u64(lane) ^ xof_next_u64(xof)); }
        ShaXof xof2(seed);                       // fresh XOF for the permutation
        const u32 n = u32(s.size());
        std::vector<u32> pi(n); for(u32 i=0;i<n;++i) pi[i]=i;
        for (u32 i=n-1;i>0;--i){ u32 j=xof2.next()%(i+1); std::swap(pi[i],pi[j]); }
        std::vector<i64> tmp(n); for (u32 i=0;i<n;++i) tmp[pi[i]]=s[i]; s.swap(tmp);
    }
}
// balanced permutation over active state (Fisher-Yates via kPermTag XOF)
static std::vector<u32> balanced_perm(const H256& sigma, u32 barrier, u32 n) {
    const H256 perm_seed = sha_tag_u32(kPermTag, sigma, barrier);
    ShaXof xof(perm_seed);
    std::vector<u32> pi(n); for(u32 i=0;i<n;++i) pi[i]=i;
    for (u32 i=n-1;i>0;--i){ u32 j=xof.next()%(i+1); std::swap(pi[i],pi[j]); }
    return pi;
}
static void apply_perm(std::vector<i64>& s, const std::vector<u32>& pi) {
    std::vector<i64> tmp(s.size());
    for (u32 i=0;i<s.size();++i) tmp[pi[i]]=s[i];
    s.swap(tmp);
}
// non-affine Extract: ExtractMXTileInt64 per 32-wide tile (reuses extract_mx_tile_i64 verbatim)
static void extract_active(const H256& prf, const std::vector<i64>& raw, std::vector<i8>& out) {
    const u32 n_tiles = u32(raw.size()/kMxBlockLen);
    for (u32 t=0;t<n_tiles;++t)
        extract_mx_tile_i64(prf, 0, t, raw.data()+t*kMxBlockLen, out.data()+t*kMxBlockLen);
}
static H256 barrier_root(u32 barrier, const std::vector<i8>& state) {
    // DOUBLE SHA256(BarrierTag || LE32(barrier) || state) -- Sha256dBytes in coupled.cpp
    std::vector<u8> buf; buf.insert(buf.end(), kBarrierTag, kBarrierTag+std::strlen(kBarrierTag));
    u8 le[4]; wle32(le,barrier); buf.insert(buf.end(), le, le+4);
    buf.insert(buf.end(), (const u8*)state.data(), (const u8*)state.data()+state.size());
    return sha256d(buf.data(), buf.size());
}


// ============================================================ coupled episode
// Per-nonce leg against a prebuilt bank. The bank (pages + root) is nonce-independent -- the old
// single-body form re-derived all of it EVERY nonce, which at production dims (~96 GiB of SHA-XOF
// expansion) dominated the "minutes/nonce" cost the solve loop used to pay.
static H256 run_coupled_with_bank(const H256& sigma, const std::vector<std::vector<i8>>& pages,
                                  const H256& bank_root, const Coup& p) {
    const u32 n = state_bytes(p);

    // nonce-fresh lobe activation: first M rows of a WxW MX tile per lobe
    const auto lseeds = lobe_seeds(sigma, p);
    const u32 M = p.rows_per_lobe, W = p.lobe_width, lobe_stride = M*W;
    std::vector<i8> state(n);
    for (u32 e=0;e<p.lobes;++e){
        const auto tile = expand_mx_dequant_i8(lseeds[e], W, W);
        std::memcpy(state.data()+size_t(e)*lobe_stride, tile.data(), lobe_stride);
    }

    std::vector<H256> roots(p.barriers);
    for (u32 b=0;b<p.barriers;++b){
        std::vector<i64> acc(n, 0);
        // C3.a local MxW GEMM per lobe vs selected bank pages (full schedule). Lobes are
        // independent (disjoint dest slices, read-only pages/state) -> one thread per lobe.
        auto lobe_leg = [&](u32 e) {
            const auto ids = select_pages(b, e, sigma, p);
            i64* dest = acc.data()+size_t(e)*lobe_stride;
            for (u32 pid : ids)
                lobe_gemm_add(state.data()+size_t(e)*lobe_stride, pages[pid], M, W, dest);
        };
        if (p.lobes > 1 && std::thread::hardware_concurrency() > 1) {
            std::vector<std::thread> ts; ts.reserve(p.lobes);
            for (u32 e=0;e<p.lobes;++e) ts.emplace_back(lobe_leg, e);
            for (auto& th : ts) th.join();
        } else {
            for (u32 e=0;e<p.lobes;++e) lobe_leg(e);
        }
        // C3.b balanced permutation, C3.c all-to-all mix (+V3 exchange rounds),
        // C3.d Extract, C3.e feed-forward
        apply_perm(acc, balanced_perm(sigma, b, n));
        all_to_all_mix(acc, sigma, b, n);
        material_exchange_rounds(acc, sigma, b, p.exchange_rounds);
        const H256 prf = derive_prf_key(sha_tag_u32u32(kExtractTag, sigma, b, 0));
        extract_active(prf, acc, state);
        roots[b] = barrier_root(b, state);
    }

    // episode_digest = SHA256d(EpisodeTag || bank_root || barrier_roots...)
    std::vector<u8> buf;
    buf.insert(buf.end(), kEpisodeTag, kEpisodeTag+std::strlen(kEpisodeTag));
    buf.insert(buf.end(), bank_root.begin(), bank_root.end());
    for (const auto& r : roots) buf.insert(buf.end(), r.begin(), r.end());
    return sha256d(buf.data(), buf.size());
}


} // namespace

CoupParamsV3 MediumV3CoupParams() { return CoupParamsV3{}; }

CoupParamsV3 ProductionV3CoupParams()
{
    CoupParamsV3 p;
    p.barriers = 8; p.lobes = 8; p.lobe_width = 8192; p.bank_pages = 1536;
    p.rows_per_lobe = 128; p.pages_per_barrier_lobe = 24; p.exchange_rounds = 4;
    return p;
}

CoupledBankV3 BuildCoupledBankV3(const uint256& bank_template_hash, uint32_t height,
                                 const CoupParamsV3& params)
{
    const H256 brs = bank_root_seed(h256_from(bank_template_hash), height);
    CoupledBankV3 bank;
    bank.pages = derive_bank_pages(brs, params);
    const H256 root = bank_commitment(bank.pages, params);
    bank.bank_root = uint256{Span<const unsigned char>{root.data(), 32}};
    return bank;
}

uint256 ComputeCoupledDigestV3WithBank(const uint256& sigma, const CoupledBankV3& bank,
                                       const CoupParamsV3& params)
{
    // Shape gate: a bank built for different params must not silently produce a wrong digest.
    const size_t page_len = size_t(params.lobe_width) * params.lobe_width;
    if (bank.pages.size() != params.bank_pages) return uint256{};
    for (const auto& pg : bank.pages) if (pg.size() != page_len) return uint256{};
    const H256 d = run_coupled_with_bank(h256_from(sigma), bank.pages,
                                         h256_from(bank.bank_root), params);
    return uint256{Span<const unsigned char>{d.data(), 32}};
}

uint256 ComputeCoupledDigestV3(const uint256& sigma, const uint256& bank_template_hash,
                               uint32_t height, const CoupParamsV3& params)
{
    const CoupledBankV3 bank = BuildCoupledBankV3(bank_template_hash, height, params);
    return ComputeCoupledDigestV3WithBank(sigma, bank, params);
}

// Byte-exact to upstream RCBankTemplateHash: nonce/seed/matmul_digest-nulled projection hashed via
// ComputeMatMulHeaderHash (matmul_digest is not part of that preimage). Nonce-independent, so one
// call per template feeds every nonce's ComputeCoupledDigestV3.
uint256 RCBankTemplateHash(const CBlockHeader& header)
{
    CBlockHeader tmpl{header};
    tmpl.nNonce64 = 0;
    tmpl.nNonce = 0;
    tmpl.seed_a.SetNull();
    tmpl.seed_b.SetNull();
    tmpl.matmul_digest.SetNull();
    return matmul::ComputeMatMulHeaderHash(tmpl);
}

} // namespace matmul::v4::rc
