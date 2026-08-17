// Definitive reference check: compiles BTX PR #89's OWN int8_field.cpp +
// matmul_v4.cpp (unchanged) and drives them directly, to prove:
//   (1) operand expansion really does one SHA-256 per matrix element
//       => ~37.7 M SHA-256 per nonce at n=4096  (the SHA-bound mechanism);
//   (2) their optimal sketch (U*A)(B*V) == full-C U*(A*B)*V, byte-identical
//       (their own A12 invariant), confirming our understanding of the work.
// SHA-256 is our own KAT-verified streaming impl (sha256_impl.cpp) with a
// Finalize counter; everything algorithmic is THEIR code.
#include <matmul/int8_field.h>
#include <matmul/matmul_v4.h>
#include <crypto/sha256.h>
#include <primitives/block.h>
#include <uint256.h>
#include <span.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern std::atomic<uint64_t> g_sha_finalize;

// Linker stubs: matmul_v4.cpp references these, but we call the header-free
// building blocks directly, so they are never invoked.
namespace matmul {
uint256 DeriveSigma(const CBlockHeader&) { return uint256(); }
uint256 ComputeMatMulHeaderHash(const CBlockHeader&) { return uint256(); }
} // namespace matmul

using clk = std::chrono::steady_clock;
static double ms(clk::time_point t0){return std::chrono::duration<double,std::milli>(clk::now()-t0).count();}
static uint256 mkseed(int base){ unsigned char b[32]; for(int i=0;i<32;i++) b[i]=(unsigned char)(base+i); return uint256{Span<const unsigned char>{b,32}}; }

// Fixed, byte-reproducible seeds shared with v4_proto.cu --emit (the SAME 32-byte
// patterns), so the reference digest and the GPU-prototype digest can be diffed
// bit-for-bit. Convention: uint256 storage byte b[i] = base+i.
static uint256 fixedseed(unsigned base){ unsigned char b[32]; for(int i=0;i<32;i++) b[i]=(unsigned char)(base+i); return uint256{Span<const unsigned char>{b,32}}; }

// Emit the reference digest for fixed seeds (seed_a=0.., seed_b=64.., seed_u=128..,
// seed_v=192.., sigma=32..) so the GPU prototype can reproduce it byte-for-byte.
static void emit(uint32_t n){
    using namespace matmul::v4;
    uint32_t m=0; if(!ValidateDims(n,kTileB,m)){printf("bad dims\n");return;}
    uint256 sa=fixedseed(0), sb=fixedseed(64), su=fixedseed(128), sv=fixedseed(192), sigma=fixedseed(32);
    auto A=ExpandOperand(sa,n), B=ExpandOperand(sb,n);
    auto U=ExpandProjector(su,m,n), V=ExpandProjector(sv,n,m);
    auto Chat=ComputeSketchOptimal(U,A,B,V,n,m);
    auto payload=SerializeSketch(Chat);
    auto digest=ComputeSketchDigest(sigma,payload);
    printf("REF n=%u m=%u\n", n, m);
    printf("A[0..7] ="); for(int i=0;i<8;i++) printf(" %d",(int)A[i]); printf("\n");
    printf("CHAT[0] =%llu\n",(unsigned long long)Chat[0]);
    printf("DIGEST  ="); for(auto* d=digest.data(); d<digest.data()+32; ++d) printf("%02x",*d); printf("\n");
}

static bool kat_sha256_abc() {
    unsigned char h[32];
    CSHA256().Write(reinterpret_cast<const unsigned char*>("abc"), 3).Finalize(h);
    static const unsigned char want[32] = {
        0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,
        0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
    return std::memcmp(h, want, 32) == 0;
}

int main(int argc, char** argv) {
    using namespace matmul::v4;

    // `refcheck emit <n>` prints the reference digest for fixed seeds (for the
    // byte-exact cross-check against v4_proto.cu --emit).
    if (argc > 1 && std::strcmp(argv[1], "emit") == 0) {
        if (!kat_sha256_abc()) { printf("SHA KAT FAIL\n"); return 1; }
        emit(argc > 2 ? (uint32_t)atoi(argv[2]) : 256);
        return 0;
    }

    printf("SHA-256 KAT (\"abc\"): %s\n", kat_sha256_abc() ? "PASS" : "FAIL -- do not trust results");
    printf("(int8_field.cpp + matmul_v4.cpp are the UNMODIFIED PR #89 reference)\n\n");

    // ---- n=256: full check (count + optimal==full-C + digest) ----
    {
        uint32_t n = 256, m = 0;
        ValidateDims(n, kTileB, m);
        uint256 sa=mkseed(1), sb=mkseed(100), su=mkseed(50), sv=mkseed(200), sigma=mkseed(7);
        g_sha_finalize = 0;
        auto A = ExpandOperand(sa, n); auto B = ExpandOperand(sb, n);
        auto U = ExpandProjector(su, m, n); auto V = ExpandProjector(sv, n, m);
        uint64_t sha_expand = g_sha_finalize.load();
        uint64_t expect = 2ull*n*n + 2ull*(uint64_t)m*n;

        auto Chat_opt  = ComputeSketchOptimal(U, A, B, V, n, m);
        auto C         = ComputeExactProduct(A, B, n);
        auto Chat_full = ComputeSketch(U, C, V, n, m);
        bool eq = Chat_opt.size()==Chat_full.size() &&
                  std::memcmp(Chat_opt.data(), Chat_full.data(), Chat_opt.size()*sizeof(uint64_t))==0;

        auto payload = SerializeSketch(Chat_opt);
        auto digest  = ComputeSketchDigest(sigma, payload);

        printf("n=%u  m=%u\n", n, m);
        printf("  operand-expansion SHA-256 Finalize calls = %llu   (2n^2+2mn = %llu, ratio %.4f incl. rejections)\n",
               (unsigned long long)sha_expand, (unsigned long long)expect, (double)sha_expand/expect);
        printf("  ComputeSketchOptimal (U*A)(B*V) == ComputeSketch full-C U*(A*B)*V : %s\n", eq?"BYTE-IDENTICAL":"MISMATCH");
        printf("  Chat[0..3] = %llu %llu %llu %llu\n",(unsigned long long)Chat_opt[0],(unsigned long long)Chat_opt[1],(unsigned long long)Chat_opt[2],(unsigned long long)Chat_opt[3]);
        printf("  digest H(sigma||Chat) = "); for(auto* d=digest.data(); d<digest.data()+32; ++d) printf("%02x",*d); printf("\n\n");
    }

    // ---- n=4096: SHA count at the launch dimension (the headline number) ----
    {
        uint32_t n = 4096, m = 0;
        ValidateDims(n, kTileB, m);
        uint256 sa=mkseed(11), sb=mkseed(111), su=mkseed(55), sv=mkseed(211);
        g_sha_finalize = 0;
        auto t0 = clk::now();
        auto A = ExpandOperand(sa, n); auto B = ExpandOperand(sb, n);
        auto U = ExpandProjector(su, m, n); auto V = ExpandProjector(sv, n, m);
        double t = ms(t0);
        uint64_t sha_expand = g_sha_finalize.load();
        uint64_t expect = 2ull*n*n + 2ull*(uint64_t)m*n;
        uint64_t gemm_macs = 2ull*n*n*m;
        volatile int8_t sink = A[0]^B.back()^U[0]^V.back(); (void)sink;

        printf("n=%u  m=%u  (LAUNCH DEFAULT)\n", n, m);
        printf("  operand-expansion SHA-256 Finalize calls = %llu  (~%.1f M)\n",
               (unsigned long long)sha_expand, sha_expand/1e6);
        printf("  expected 2n^2+2mn = %llu ; actual/expected = %.4f\n",
               (unsigned long long)expect, (double)sha_expand/expect);
        printf("  vs the actual matmul work = %.2e INT8 MACs  ->  %llu SHA-256 per %llu MACs\n",
               (double)gemm_macs, (unsigned long long)sha_expand, (unsigned long long)gemm_macs);
        printf("  (expansion wall time on this 1 core, generic SHA = %.0f ms; a real miner batches SHA on-GPU)\n", t);
    }
    return 0;
}
