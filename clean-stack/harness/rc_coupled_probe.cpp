// Byte-exact gate for the v4.6 ENC_RC coupled V3 puzzle.
//
// Computes the medium-V3 coupled digest for btx's own test header (MakeCoupHeader(42) @
// height 0) and asserts it equals PR#89's frozen golden. If this ever prints something else,
// our coupled path has forked from consensus -- same contract rc_probe enforces for the
// episode and digest_probe for the base chain.
//
// sigma and the bank-template hash below are recomputed values for that test header,
// cross-checked against the research oracle (rc_coupled_solver_v3.cpp) which derives them
// from the serialized header itself.
//
// EXPECT: a4bb0cc42e2b97631d126a0dcdae26ad83b2f287d885322392a564990a95bac4

#include <matmul/matmul_v4_rc_coupled.h>
#include <uint256.h>

#include <cstdio>
#include <string_view>

static uint256 PU(std::string_view hex) { return uint256::FromHex(hex).value(); }

int main()
{
    const uint256 sigma =
        PU("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2");
    const uint256 tmpl =
        PU("221c4a4edd1bdf4ecae55f67c7aad7691420a36405e7453bf24def8292ef1c1c");
    static constexpr const char* kGolden =
        "a4bb0cc42e2b97631d126a0dcdae26ad83b2f287d885322392a564990a95bac4";

    const auto params = matmul::v4::rc::MediumV3CoupParams();
    const uint256 digest =
        matmul::v4::rc::ComputeCoupledDigestV3(sigma, tmpl, /*height=*/0, params);

    // Split path (the solve loop's shape): bank built once, per-nonce leg separately. Must be
    // byte-identical to the one-shot oracle -- gates the BuildCoupledBankV3 /
    // ComputeCoupledDigestV3WithBank restructuring against the same golden.
    const auto bank = matmul::v4::rc::BuildCoupledBankV3(tmpl, /*height=*/0, params);
    const uint256 digest_split =
        matmul::v4::rc::ComputeCoupledDigestV3WithBank(sigma, bank, params);

    std::printf("sigma  = %s\n", sigma.GetHex().c_str());
    std::printf("digest = %s\n", digest.GetHex().c_str());
    std::printf("split  = %s\n", digest_split.GetHex().c_str());
    std::printf("EXPECT = %s\n", kGolden);

    const bool ok = digest.GetHex() == kGolden && digest_split.GetHex() == kGolden;
    std::printf("=> ENC_RC coupled V3 %s\n", ok ? "BYTE-EXACT (a4bb0cc4)" : "FORKED FROM CONSENSUS");
    return ok ? 0 : 1;
}
