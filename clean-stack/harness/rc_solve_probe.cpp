// Unit gate for the ENC_RC episode digest SOLVE LOOP (SolveMatMulRCEpisode).
//
// rc_probe gates the DIGEST (byte-exactness vs consensus golden 5b1bff3c). This gates the SOLVE
// CONTROL FLOW around it: grind nonces, compare digest vs effective_target, set matmul_digest on a
// winner, advance/exhaust otherwise -- the exact contract of consensus SolveMatMulV4RC. Runs at
// toy dims (fast), so it is CI-cheap. Does NOT exercise the profile-2 succinct carrier (a separate
// consensus requirement, not built -- see research/matmul-v4/RC-MINER-INTEGRATION.md).

#include <pow.h>
#include <primitives/block.h>
#include <consensus/params.h>
#include <matmul/matmul_v4_rc.h>
#include <matmul/matmul_pow.h>
#include <uint256.h>
#include <arith_uint256.h>

#include <cstdio>
#include <optional>

int main()
{
    Consensus::Params params{};                       // nonce-seed / mtp heights default INT32_MAX
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
    const int32_t height = 100;                        // finite; seeds derive from hashPrevBlock
    const matmul::v4::rc::EpisodeParams toy{};         // ENC_RC toy dims (fast)

    int fails = 0;

    // 1) Easy target (all-FF): the first nonce's digest is always <= target -> instant winner.
    {
        CBlockHeader b = base;
        uint64_t tries = 100;
        const arith_uint256 easy = ~arith_uint256(0);
        const bool won = SolveMatMulRCEpisode(b, params, tries, height, nullptr,
                                              std::optional<int64_t>{}, easy, toy);
        const bool digest_set = !b.matmul_digest.IsNull();
        std::printf("  easy-target      won=%d digest_set=%d\n", won, digest_set);
        if (!won || !digest_set) ++fails;
    }

    // 2) Impossible target (1): a 256-bit digest is ~never <= 1 -> grinds all tries, returns false,
    //    consumes max_tries, and advances the nonce.
    {
        CBlockHeader b = base;
        uint64_t tries = 3;
        const arith_uint256 hard = arith_uint256(1);
        const bool won = SolveMatMulRCEpisode(b, params, tries, height, nullptr,
                                              std::optional<int64_t>{}, hard, toy);
        std::printf("  hard-target      won=%d tries_left=%llu nonce=%llu (expect won=0 tries=0)\n",
                    won, static_cast<unsigned long long>(tries),
                    static_cast<unsigned long long>(b.nNonce64));
        if (won || tries != 0) ++fails;
    }

    // 3) Winner digest is byte-exact to the standalone oracle for the same (sigma, params) -- i.e.
    //    the solve loop sets exactly the consensus digest, no divergence.
    {
        CBlockHeader b = base;
        uint64_t tries = 1;
        const arith_uint256 easy = ~arith_uint256(0);
        (void)SolveMatMulRCEpisode(b, params, tries, height, nullptr,
                                   std::optional<int64_t>{}, easy, toy);
        const uint256 sigma = matmul::DeriveSigma(b);
        const uint256 oracle = matmul::v4::rc::ComputeEpisodeDigest(sigma, toy);
        const bool match = b.matmul_digest == oracle;
        std::printf("  winner==oracle   %d (%s)\n", match, b.matmul_digest.GetHex().substr(0, 16).c_str());
        if (!match) ++fails;
    }

    std::printf("=> RC episode solve loop %s (%d failure%s)\n",
                fails ? "FAIL" : "OK", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
