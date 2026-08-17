// Byte-exact gate for the v4.6 ENC_RC (Resident Curriculum) episode.
//
// Computes the ENC_RC_V1 episode digest for the frozen toy shape and asserts it equals btx
// PR#89's published golden. If this ever prints something else, our RC path has forked from
// consensus and any mined RC share would be rejected -- same contract digest_probe enforces for
// the base matmul chain.
//
// The sigma below is the v4 DeriveSigma output for btx's own test header
// (MakeRCHeader(42): version 0x20000004, prev 0x51.., merkle 0xa3.., time 1770000000,
// bits 0x207fffff, nonce 42, seed_a 0x11.., seed_b 0x22..). digest_probe already gates the
// base chain that produces sigma; this probe gates everything downstream of it.
//
// EXPECT: 5b1bff3c835b1c8e7816a2cccb181eb2fc30a99d97a971d73108c52a8238acd4 (v4.6 fused-FFN)

#include <matmul/matmul_v4_rc.h>
#include <uint256.h>

#include <cstdio>
#include <string_view>

static uint256 PU(std::string_view hex) { return uint256::FromHex(hex).value(); }

int main()
{
    const uint256 sigma =
        PU("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2");
    static constexpr const char* kGolden =
        "5b1bff3c835b1c8e7816a2cccb181eb2fc30a99d97a971d73108c52a8238acd4";

    const matmul::v4::rc::EpisodeParams toy{};   // ENC_RC_V1 frozen toy shape
    const uint256 digest = matmul::v4::rc::ComputeEpisodeDigest(sigma, toy);

    std::printf("sigma  = %s\n", sigma.GetHex().c_str());
    std::printf("digest = %s\n", digest.GetHex().c_str());
    std::printf("EXPECT = %s\n", kGolden);

    const bool ok = digest.GetHex() == kGolden;
    std::printf("=> ENC_RC episode %s\n", ok ? "BYTE-EXACT (5b1bff3c)" : "FORKED FROM CONSENSUS");
    return ok ? 0 : 1;
}
