// Refcheck emit: runs the REFERENCE ENC-BMX4C digest pipeline for the fixed
// test header (matmul_v4_bmx4_tests MakeV4Header) and prints every stage value.
// These are the byte-exact golden targets our GPU harness replica must match.
// Build: added as a BUILD_UTIL target next to matmul-v4-report. CPU-only, n=256.
#include <cstdio>
#include <vector>
#include <string>

#include <primitives/block.h>
#include <uint256.h>
#include <util/translation.h>
#include <matmul/matmul_v4.h>
#include <matmul/matmul_v4_bmx4.h>
#include <matmul/pow_v4.h>

// standalone-tool translation stub (mirrors matmul-v4-report.cpp)
const TranslateFn G_TRANSLATION_FUN{nullptr};

namespace bx = matmul::v4::bmx4;
namespace v4 = matmul::v4;

static uint256 U(const std::string& hex) { return uint256::FromHex(hex).value(); }
static void phex(const char* label, const uint256& h) { std::printf("%-10s %s\n", label, h.GetHex().c_str()); }
static void pi8(const char* label, const std::vector<int8_t>& v) {
    std::printf("%-10s", label);
    for (int i = 0; i < 8 && i < (int)v.size(); ++i) std::printf(" %d", (int)v[i]);
    std::printf("\n");
}

int main()
{
    const uint32_t n = 256;
    CBlockHeader h;
    h.nVersion = 0x20000004;
    h.hashPrevBlock  = U("5151515151515151515151515151515151515151515151515151515151515151");
    h.hashMerkleRoot = U("a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3");
    h.nTime = 1770000000u;
    h.nBits = 0x207fffff;
    h.nNonce64 = 1;
    h.nNonce = 1;
    h.matmul_dim = static_cast<uint16_t>(n);
    h.seed_a = U("1111111111111111111111111111111111111111111111111111111111111111");
    h.seed_b = U("2222222222222222222222222222222222222222222222222222222222222222");

    uint32_t m = 0;
    if (!bx::ValidateDimsBMX4C(n, v4::kTileB, m)) { std::printf("ValidateDimsBMX4C FAILED\n"); return 1; }
    std::printf("=== ENC-BMX4C reference golden (n=%u m=%u, nonce=1) ===\n", n, m);

    const uint256 sigma = v4::DeriveSigma(h);
    const uint256 seed_a = bx::DeriveOperandSeedBMX4C(h, v4::Operand::A);
    const uint256 seed_b = bx::DeriveOperandSeedBMX4C(h, v4::Operand::B);
    const auto [seed_u, seed_v] = bx::DeriveProjectorSeedsBMX4C(h);
    phex("sigma", sigma); phex("seed_a", seed_a); phex("seed_b", seed_b);
    phex("seed_u", seed_u); phex("seed_v", seed_v);

    const std::vector<int8_t> Ahat = bx::ExpandOperandA(seed_a, n);
    const std::vector<int8_t> Bhat = bx::ExpandOperandB(seed_b, n);
    const std::vector<int8_t> Um = bx::ExpandProjectorBMX4C(seed_u, m, n);
    const std::vector<int8_t> Vm = bx::ExpandProjectorBMX4C(seed_v, n, m);
    pi8("Ahat[0..7]", Ahat); pi8("Bhat[0..7]", Bhat);
    pi8("U[0..7]", Um); pi8("V[0..7]", Vm);

    const std::vector<int32_t> P = v4::ComputeProjectedLeft(Um, Ahat, n, m);
    const std::vector<int32_t> Q = v4::ComputeProjectedRight(Bhat, Vm, n, m);
    std::printf("P[0..3]    %d %d %d %d\n", P[0], P[1], P[2], P[3]);
    std::printf("Q[0..3]    %d %d %d %d\n", Q[0], Q[1], Q[2], Q[3]);

    const auto Chat = v4::ComputeCombineModQ(P, Q, n, m);
    std::printf("Chat[0..3] %llu %llu %llu %llu\n",
                (unsigned long long)Chat[0], (unsigned long long)Chat[1],
                (unsigned long long)Chat[2], (unsigned long long)Chat[3]);

    uint256 digest; std::vector<unsigned char> payload;
    if (!bx::ComputeDigestBMX4C(h, n, digest, payload)) { std::printf("ComputeDigestBMX4C FAILED\n"); return 1; }
    std::printf("payload_sz %zu (expect 8*m*m=%zu)\n", payload.size(), (size_t)8 * m * m);
    std::printf("payload[0..7]");
    for (int i = 0; i < 8 && i < (int)payload.size(); ++i) std::printf(" %02x", payload[i]);
    std::printf("\n");
    phex("DIGEST", digest);
    return 0;
}
