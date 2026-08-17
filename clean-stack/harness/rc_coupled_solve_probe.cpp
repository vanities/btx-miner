// Unit gate for the ENC_RC COUPLED (V3) digest SOLVE LOOP (SolveMatMulRCCoupled).
//
// rc_coupled_probe gates the DIGEST byte-exactness (golden a4bb0cc4). This gates the SOLVE CONTROL
// FLOW around it: grind nonces, compare the coupled digest vs effective_target, set matmul_digest
// on a winner, advance/exhaust otherwise -- the coupled analogue of rc_solve_probe. The winner's
// digest must equal the standalone ComputeCoupledDigestV3 oracle recomputed from the SAME
// (DeriveSigma(winner), RCBankTemplateHash(winner)) -- so the solve loop cannot silently fork from
// the consensus digest. Medium-V3 dims (fast, the CI golden shape).

#include <pow.h>
#include <primitives/block.h>
#include <consensus/params.h>
#include <matmul/matmul_v4_rc_coupled.h>
#include <matmul/matmul_pow.h>
#include <uint256.h>
#include <arith_uint256.h>

#include <cstdio>
#include <optional>

int main()
{
    Consensus::Params params{};
    params.powLimit = uint256::FromHex(
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff").value();

    CBlockHeader base{};
    base.nVersion = 0x20000004;
    base.hashPrevBlock = uint256::FromHex(
        "5151515151515151515151515151515151515151515151515151515151515151").value();
    base.hashMerkleRoot = uint256::FromHex(
        "a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3").value();
    base.nTime = 1770000000u;
    base.nBits = 0x207fffffu;
    base.nNonce64 = 0;
    const int32_t height = 100;
    const matmul::v4::rc::CoupParamsV3 cp = matmul::v4::rc::MediumV3CoupParams();

    int fails = 0;

    // 1) Easy target: first nonce wins; winner digest == oracle recomputed from the winner header.
    {
        CBlockHeader b = base;
        uint64_t tries = 50;
        const arith_uint256 easy = ~arith_uint256(0);
        const bool won = SolveMatMulRCCoupled(b, params, tries, height, nullptr,
                                              std::optional<int64_t>{}, easy, cp);
        const uint256 sigma = matmul::DeriveSigma(b);
        const uint256 tmpl = matmul::v4::rc::RCBankTemplateHash(b);
        const uint256 oracle = matmul::v4::rc::ComputeCoupledDigestV3(
            sigma, tmpl, static_cast<uint32_t>(height), cp);
        const bool match = won && !b.matmul_digest.IsNull() && b.matmul_digest == oracle;
        std::printf("  easy-target  won=%d winner==oracle=%d (%s)\n", won, match,
                    b.matmul_digest.GetHex().substr(0, 16).c_str());
        if (!match) ++fails;
    }

    // 2) Impossible target (1): grinds all tries, returns false, consumes the window, advances.
    {
        CBlockHeader b = base;
        uint64_t tries = 3;
        const arith_uint256 hard = arith_uint256(1);
        const bool won = SolveMatMulRCCoupled(b, params, tries, height, nullptr,
                                              std::optional<int64_t>{}, hard, cp);
        std::printf("  hard-target  won=%d tries_left=%llu (expect won=0 tries=0)\n", won,
                    static_cast<unsigned long long>(tries));
        if (won || tries != 0) ++fails;
    }

    std::printf("=> RC coupled solve loop %s (%d failure%s)\n",
                fails ? "FAIL" : "OK", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
