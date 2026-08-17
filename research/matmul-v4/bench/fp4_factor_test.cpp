// fp4_factor_test.cpp -- proves whether the ENC_RC FFN operands can be packed into native MXFP4
// (E2M1 + UE8M0 block scales) byte-exact, using numair's OWN FactorBlockToMx gate verbatim.
//
// Native MXFP4 (mma.sync kind::mxf8f6f4) needs ONE UE8M0 scale per 32-element K-block. FactorBlockToMx
// tries e=3..0 and requires every element in the block to be (M11 mantissa)*2^e for a SINGLE e.
// The consensus weight operand's E8M0 scale is per-(row, free-axis-block) = per-element along K, so a
// K-block spans mixed scales and cannot factor. RESULT (production d_model=4096/d_ff=16384):
//   X activation (A, K=d_model): 100% factor  (MX-native along K)
//   W_up / W_dn weight (B):      0.0% factor  (1 / 2,097,152)  -> his native MXFP4 GEMM DECLINES -> int8
// So byte-exact native MXFP4 is IMPOSSIBLE for the FFN weight GEMMs; int8 IMMA is the fastest LEGAL path.
// The only unlock is a CONSENSUS change: block the weight E8M0 scale along K (not the free axis).
//   c++ -O2 -std=c++17 fp4_factor_test.cpp -o fp4ft && ./fp4ft

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


static H256 from_hex(const char* s){ H256 o{}; auto v=[](char c){return u32(c<='9'?c-'0':(c|32)-'a'+10);};
    for(int i=0;i<32;++i) o[31-i]=u8((v(s[i*2])<<4)|v(s[i*2+1])); return o; }
static bool IsM11v(int mu){ switch(mu){case 0:case 1:case -1:case 2:case -2:case 3:case -3:case 4:case -4:case 6:case -6:return true;} return false; }
static bool FactorBlockToMx(const i8* vals, u32 n){
    for(int e=3;e>=0;--e){ int scale=1<<e; bool ok=true;
        for(u32 i=0;i<n;++i){ int vv=vals[i]; if(vv%scale){ok=false;break;} if(!IsM11v(vv/scale)){ok=false;break;} }
        if(ok) return true; }
    return false;
}
static void report(const char* name, const std::vector<i8>& M, u32 rows, u32 cols, bool as_A){
    const u32 K = as_A ? cols : rows; const u32 kb=(K+31)/32;
    long ok=0,tot=0; i8 blk[32];
    if(as_A){ for(u32 r=0;r<rows;++r) for(u32 bj=0;bj<kb;++bj){ u32 k0=bj*32,n=std::min(32u,K-k0);
        for(u32 t=0;t<n;++t) blk[t]=M[(size_t)r*cols+(k0+t)]; ok+=FactorBlockToMx(blk,n); ++tot; } }
    else{ for(u32 c=0;c<cols;++c) for(u32 bj=0;bj<kb;++bj){ u32 k0=bj*32,n=std::min(32u,K-k0);
        for(u32 t=0;t<n;++t) blk[t]=M[(size_t)(k0+t)*cols+c]; ok+=FactorBlockToMx(blk,n); ++tot; } }
    printf("  %-30s %ld / %ld blocks factor (%.1f%%) -> %s\n", name, ok, tot, 100.0*ok/tot,
           ok==tot?"FP4-VIABLE":"FALLS BACK TO INT8");
}
int main(){
    const H256 sigma=from_hex("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2");
    u8 rb[36]; std::memcpy(rb,sigma.data(),32); std::memset(rb+32,0,4);
    const H256 seed_r=sha256_tagged("BTX_RC_ROUND_V1",rb,36);
    const u32 d_model=4096,d_ff=16384,b_slice=1024;
    printf("FFN operand MX-factorability (production dims, X.W_up + H.W_dn):\n");
    auto Wup=expand_mx_dequant_i8(derive_operand_seed(sigma,"BTX_RC_WUP_V1"),d_model,d_ff);
    auto X=expand_mx_dequant_i8(derive_operand_seed(seed_r,"BTX_RC_X0_V1"),b_slice,d_model);
    report("X activation (A, K=d_model)",X,b_slice,d_model,true);
    report("W_up weight (B, K=d_model)",Wup,d_model,d_ff,false);
    auto Wdn=expand_mx_dequant_i8(derive_operand_seed(sigma,"BTX_RC_WDN_V1"),d_ff,d_model);
    report("W_dn weight (B, K=d_ff)",Wdn,d_ff,d_model,false);
    return 0;
}
