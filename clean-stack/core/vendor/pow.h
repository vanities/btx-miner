// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// Vendored from btx, then cut to the v4/ENC_RC surface the miner actually uses.
// The v3 declarations that lived here -- the Freivalds / product-committed /
// phase1+phase2 verify entry points, the sigma pre-hash gate, the peer
// verification budget, and the v3 pipeline telemetry -- went out with the v3
// solver. The node still defines them in libbitcoin_consensus; the miner no
// longer calls them.

#ifndef BITCOIN_POW_H
#define BITCOIN_POW_H

#include <consensus/params.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <stdint.h>
#include <string>
#include <vector>

class CBlockHeader;
class CBlock;
class CBlockIndex;
class uint256;
class arith_uint256;

/**
 * Convert nBits value to target.
 *
 * @param[in] nBits     compact representation of the target
 * @param[in] pow_limit PoW limit (consensus parameter)
 *
 * @return              the proof-of-work target or nullopt if the nBits value
 *                      is invalid (due to overflow or exceeding pow_limit)
 */
std::optional<arith_uint256> DeriveTarget(unsigned int nBits, const uint256 pow_limit);

struct MatMulSolvePipelineStats {
    //! ENC_RC episodes completed -- the RC path's proof of life. A saturated GPU
    //! that never finds a share looks identical to a wedged one without it.
    uint64_t rc_episodes{0};
    //! Solve windows entered (one per SolveMatMul call that reached a solver).
    uint64_t solve_windows{0};
};

struct MatMulAsertHalfLifeInfo {
    int64_t current_half_life_s{0};
    int32_t current_anchor_height{-1};
    bool upgrade_configured{false};
    bool upgrade_active{false};
    int32_t upgrade_height{-1};
    int64_t upgrade_half_life_s{0};
};

unsigned int GetNextWorkRequired(const CBlockIndex* pindexLast, const CBlockHeader *pblock, const Consensus::Params&);
unsigned int CalculateNextWorkRequired(const CBlockIndex* pindexLast, int64_t nFirstBlockTime, const Consensus::Params&);
bool EnforceTimewarpProtectionAtHeight(const Consensus::Params& params, int32_t block_height);

/** Check whether a block hash satisfies the proof-of-work requirement specified by nBits */
bool CheckProofOfWork(uint256 hash, unsigned int nBits, const Consensus::Params&);
bool CheckProofOfWorkImpl(uint256 hash, unsigned int nBits, const Consensus::Params&);
int64_t ExpectedDgwTimespan(int32_t height, const Consensus::Params& params);
uint256 DeterministicMatMulSeed(const uint256& prev_block_hash,
                                uint32_t height,
                                uint8_t which,
                                std::optional<uint64_t> nonce = std::nullopt);
uint256 DeterministicMatMulSeedV2(const CBlockHeader& block, uint32_t height, uint8_t which);
uint256 DeterministicMatMulSeedV3(const CBlockHeader& block, uint32_t height, int64_t parent_median_time_past, uint8_t which);
[[nodiscard]] bool SetDeterministicMatMulSeeds(
    CBlockHeader& block,
    const Consensus::Params& params,
    int32_t block_height,
    std::optional<int64_t> parent_median_time_past = std::nullopt);

MatMulSolvePipelineStats ProbeMatMulSolvePipelineStats();
void ResetMatMulSolvePipelineStats();

//! Height-gated matmul solve variants. ActiveMatMulSolveVariant() is the single generic
//! selector mapping a block height to the active matmul PoW solve path. RC = v4 ENC_RC
//! (Resident Curriculum) at and above nMatMulRCHeight; Coupled = the two-stage v4.6 leg,
//! which supersedes RC where both are active. Base names the retired v3 path: it has no
//! solver in this build and SolveMatMul fails closed on it.
enum class MatMulSolveVariant { Base, RC, Coupled };
MatMulSolveVariant ActiveMatMulSolveVariant(const Consensus::Params& params, int32_t block_height);

bool SolveMatMul(CBlockHeader& block, const Consensus::Params& params, uint64_t& max_tries,
                 int32_t block_height = -1,
                 const std::atomic<bool>* abort_flag = nullptr,
                 //! Optional pool/share mining target. When non-null, the solver returns as soon as it
                 //! finds a nonce whose digest is <= *share_target_override (typically an EASIER,
                 //! numerically larger target than the block target derived from nBits). A returned
                 //! candidate that also meets the block target is a fully consensus-valid block, so
                 //! every share is a genuine block candidate. Pass nullptr (default) for solo mining.
                 //! A zero target is rejected (returns false, loudly).
                 const uint256* share_target_override = nullptr,
                 std::optional<int64_t> parent_median_time_past = std::nullopt,
                 //! Optional pool-share sink (only meaningful with share_target_override). A candidate
                 //! whose digest meets the share target but NOT the block target is handed to the sink
                 //! instead of ending the solve: return true to keep mining on the SAME hot pipeline,
                 //! false to stop and receive that candidate as the solve result. Candidates meeting
                 //! the BLOCK target always return solved=true without invoking the sink.
                 const std::function<bool(const CBlockHeader&)>* share_sink = nullptr);

//! ENC_RC episode digest solve loop (exposed for rc_solve_probe). Grinds nonces through
//! ComputeEpisodeDigest vs effective_target, mirroring consensus SolveMatMulV4RC; a winner
//! (digest <= target) sets block.matmul_digest and returns true. At Epoch A / Profile 1 the
//! digest IS the whole proof (ExactReplay authority), so a winner is block-eligible as-is.
//! See research/matmul-v4/RC-MINER-INTEGRATION.md.
namespace matmul::v4::rc { struct EpisodeParams; }
bool SolveMatMulRCEpisode(CBlockHeader& block, const Consensus::Params& params,
                          uint64_t& max_tries, int32_t block_height,
                          const std::atomic<bool>* abort_flag,
                          std::optional<int64_t> parent_median_time_past,
                          const arith_uint256& effective_target,
                          const matmul::v4::rc::EpisodeParams& ep);

//! ENC_RC COUPLED (V3) digest solve loop (two-stage v4.6, coupled leg). Same contract as
//! SolveMatMulRCEpisode but grinds ComputeCoupledDigestV3 (byte-exact golden a4bb0cc4); the
//! nonce-independent bank_template_hash is computed once up front.
namespace matmul::v4::rc { struct CoupParamsV3; }
bool SolveMatMulRCCoupled(CBlockHeader& block, const Consensus::Params& params,
                          uint64_t& max_tries, int32_t block_height,
                          const std::atomic<bool>* abort_flag,
                          std::optional<int64_t> parent_median_time_past,
                          const arith_uint256& effective_target,
                          const matmul::v4::rc::CoupParamsV3& cp);

bool CheckKAWPOWProofOfWork(const CBlockHeader& block, uint32_t block_height, const Consensus::Params&);
bool SolveKAWPOW(CBlockHeader& block, uint32_t block_height, const Consensus::Params& params, uint64_t& max_tries);

/**
 * Return false if the proof-of-work requirement specified by new_nbits at a
 * given height is not possible, given the proof-of-work on the prior block as
 * specified by old_nbits.
 *
 * This function only checks that the new value is within a factor of 4 of the
 * old value for blocks at the difficulty adjustment interval, and otherwise
 * requires the values to be the same.
 *
 * Always returns true on networks where min difficulty blocks are allowed,
 * such as regtest/testnet.
 */
bool PermittedDifficultyTransition(const Consensus::Params& params, int64_t height, uint32_t old_nbits, uint32_t new_nbits);
MatMulAsertHalfLifeInfo GetMatMulAsertHalfLifeInfo(const CBlockIndex* pindexLast, const Consensus::Params& params);

#endif // BITCOIN_POW_H
