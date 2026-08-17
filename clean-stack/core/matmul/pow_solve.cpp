// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// ENC_RC (v4) solve path. The v3 matmul grind -- GPU pre-hash scan, batched
// digest pipeline, CPU scan-ahead, parallel solver -- and the v3-only consensus
// verify paths (Freivalds carrier, product-committed digest, phase1/phase2, the
// sigma pre-hash gate) were removed when v4 activated on mainnet; the chain no
// longer produces or accepts a v3 block, so the only solver here is the episode
// digest loop (plus the coupled-V3 loop, still gated off).
//
// What remains, and why:
//   - deterministic seeds: the RC episode preimage is built from them.
//   - DGW / ASERT retarget + DeriveTarget: solo mining and the difficulty
//     readout still need the target math.
//   - SolveMatMulRCEpisode / SolveMatMulRCCoupled: the v4 lottery.

#include <pow.h>

#include <arith_uint256.h>
#include <chain.h>
#include <crypto/sha256.h>
#include <hash.h>
#include <logging.h>
#include <matmul/matmul_pow.h>
#include <matmul/matmul_v4_rc.h>
#include <matmul/matmul_v4_rc_coupled.h>  // ComputeCoupledDigestV3 / RCBankTemplateHash / CoupParamsV3
#include <primitives/block.h>
#include <uint256.h>
#include <util/check.h>
#include <util/strencodings.h>
#include <util/string.h>
#include <util/time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

uint256 DeterministicMatMulSeed(const uint256& prev_block_hash, uint32_t height, uint8_t which,
                                std::optional<uint64_t> nonce)
{
    HashWriter hw;
    hw << prev_block_hash << height << which;
    // e1 fix (nonce-fold, flag-day gated): when a nonce is supplied, fold it into the seed so the
    // dense A*B product is nonce-DEPENDENT and cannot be precomputed once per tip and reused across
    // the whole nonce range (the ~12.8x amortization). The nonce is APPENDED so that the legacy
    // (no-nonce) derivation is byte-identical to before -- pre-activation blocks and genesis keep
    // their exact historical seeds. Callers pass the nonce only when IsMatMulNonceSeedActive(height).
    if (nonce.has_value()) {
        hw << *nonce;
    }
    return hw.GetSHA256();
}

uint256 DeterministicMatMulSeedV2(const CBlockHeader& block, uint32_t height, uint8_t which)
{
    HashWriter hw;
    hw << std::string{"BTX_MATMUL_SEED_V2"}
       << block.hashPrevBlock
       << height
       << block.nVersion
       << block.hashMerkleRoot
       << block.nTime
       << block.nBits
       << block.nNonce64
       << block.matmul_dim
       << which;
    return hw.GetSHA256();
}

uint256 DeterministicMatMulSeedV3(const CBlockHeader& block, uint32_t height, int64_t parent_median_time_past, uint8_t which)
{
    HashWriter hw;
    hw << std::string{"BTX_MATMUL_SEED_V3"}
       << block.hashPrevBlock
       << parent_median_time_past
       << height
       << block.nVersion
       << block.hashMerkleRoot
       << block.nTime
       << block.nBits
       << block.nNonce64
       << block.matmul_dim
       << which;
    return hw.GetSHA256();
}

bool SetDeterministicMatMulSeeds(
    CBlockHeader& block,
    const Consensus::Params& params,
    int32_t block_height,
    std::optional<int64_t> parent_median_time_past)
{
    if (block_height < 0) {
        block.seed_a.SetNull();
        block.seed_b.SetNull();
        return true;
    }
    if (params.IsMatMulParentMtpSeedActive(block_height)) {
        if (!parent_median_time_past.has_value()) {
            block.seed_a.SetNull();
            block.seed_b.SetNull();
            return false;
        }
        block.seed_a = DeterministicMatMulSeedV3(block, static_cast<uint32_t>(block_height), *parent_median_time_past, 0);
        block.seed_b = DeterministicMatMulSeedV3(block, static_cast<uint32_t>(block_height), *parent_median_time_past, 1);
        return true;
    }
    if (params.IsMatMulNonceSeedActive(block_height)) {
        block.seed_a = DeterministicMatMulSeedV2(block, static_cast<uint32_t>(block_height), 0);
        block.seed_b = DeterministicMatMulSeedV2(block, static_cast<uint32_t>(block_height), 1);
        return true;
    }

    block.seed_a = DeterministicMatMulSeed(block.hashPrevBlock, static_cast<uint32_t>(block_height), 0);
    block.seed_b = DeterministicMatMulSeed(block.hashPrevBlock, static_cast<uint32_t>(block_height), 1);
    return true;
}

namespace {
constexpr int64_t DGW_PAST_BLOCKS{180};
constexpr uint32_t DEFAULT_MINER_HEADER_TIME_REFRESH_ATTEMPTS{4'096U};
constexpr uint64_t MATMUL_V2_ABS_MAX_DIM{2048};
constexpr uint64_t MATMUL_V2_MAX_PAYLOAD_WORDS{MATMUL_V2_ABS_MAX_DIM * MATMUL_V2_ABS_MAX_DIM};
constexpr int64_t WARMUP_HARDENING_MIN_NUM{5};
constexpr int64_t WARMUP_HARDENING_MIN_DEN{6};
constexpr int64_t WARMUP_EASING_MAX_NUM{3};
constexpr int64_t WARMUP_EASING_MAX_DEN{1};
constexpr int64_t NORMAL_LEGACY_HARDENING_MIN_NUM{2};
constexpr int64_t NORMAL_LEGACY_HARDENING_MIN_DEN{3};
constexpr int64_t NORMAL_LEGACY_EASING_MAX_NUM{3};
constexpr int64_t NORMAL_LEGACY_EASING_MAX_DEN{2};
constexpr int64_t NORMAL_HARDENED_HARDENING_MIN_NUM{3};
constexpr int64_t NORMAL_HARDENED_HARDENING_MIN_DEN{4};
constexpr int64_t NORMAL_HARDENED_EASING_MAX_NUM{2};
constexpr int64_t NORMAL_HARDENED_EASING_MAX_DEN{1};
constexpr int64_t NORMAL_BOOSTED_EASING_MAX_NUM{3};
constexpr int64_t NORMAL_BOOSTED_EASING_MAX_DEN{1};
constexpr unsigned int NORMAL_SLEW_GUARD_SHIFT{2}; // 4x max change per block
constexpr uint8_t ASERT_RADIX_BITS{16};
// aserti3-2d fixed-point cubic approximation coefficients.
//
// For frac in [0, 2^16):
//   factor = 2^16 + ((C1*frac + C2*frac^2 + C3*frac^3 + 2^47) >> 48)
//
// This approximates 2^(frac / 2^16) deterministically with integer arithmetic.
// The constants match the BCH reference implementation and avoid floating point
// behavior in consensus code.
constexpr uint64_t ASERT_POLY_COEFF_1{195766423245049ULL};
constexpr uint64_t ASERT_POLY_COEFF_2{971821376ULL};
constexpr uint64_t ASERT_POLY_COEFF_3{5127ULL};
constexpr int64_t WARMUP_RESTART_GAP_THRESHOLD_MULTIPLIER{2};
constexpr int64_t WARMUP_RESTART_GAP_DAMPING_DIVISOR{2};

arith_uint256 SaturatingLeftShift256(const arith_uint256& val, unsigned int shift)
{
    if (shift == 0 || val == arith_uint256(0)) return val;
    if (shift >= 256) return (val == arith_uint256(0)) ? arith_uint256(0) : ~arith_uint256(0);
    arith_uint256 mask = ~arith_uint256(0);
    mask >>= shift;
    if (val > mask) return ~arith_uint256(0);  // saturate
    return val << shift;
}

arith_uint256 ClampRetargetResult(arith_uint256 target, const arith_uint256& pow_limit)
{
    // Never emit an unencodable/invalid compact target.
    if (target == 0) {
        target = arith_uint256{1};
    }
    if (target > pow_limit) {
        target = pow_limit;
    }
    return target;
}

arith_uint256 SaturatingMultiplyByUint32(const arith_uint256& value, uint32_t factor)
{
    if (value == 0 || factor == 0) {
        return arith_uint256{0};
    }
    const arith_uint256 max_uint{~arith_uint256{}};
    if (value > (max_uint / factor)) {
        return max_uint;
    }
    return value * factor;
}

arith_uint256 ScaleTargetByTimespan(const arith_uint256& target, int64_t actual_timespan, int64_t target_timespan)
{
    if (actual_timespan <= 0) {
        LogWarning("ScaleTargetByTimespan: actual_timespan=%lld is non-positive, clamping to 1\n",
                   static_cast<long long>(actual_timespan));
        actual_timespan = 1;
    }
    if (target_timespan <= 0) {
        LogWarning("ScaleTargetByTimespan: target_timespan=%lld is non-positive, clamping to 1\n",
                   static_cast<long long>(target_timespan));
        target_timespan = 1;
    }
    if (actual_timespan > std::numeric_limits<uint32_t>::max()) {
        LogWarning("ScaleTargetByTimespan: actual_timespan=%lld exceeds uint32_t max, clamping\n",
                   static_cast<long long>(actual_timespan));
        actual_timespan = std::numeric_limits<uint32_t>::max();
    }
    if (target_timespan > std::numeric_limits<uint32_t>::max()) {
        LogWarning("ScaleTargetByTimespan: target_timespan=%lld exceeds uint32_t max, clamping\n",
                   static_cast<long long>(target_timespan));
        target_timespan = std::numeric_limits<uint32_t>::max();
    }

    const uint32_t actual_u{static_cast<uint32_t>(actual_timespan)};
    const uint32_t target_u{static_cast<uint32_t>(target_timespan)};

    // Compute floor(target * actual / target_timespan) without intermediate
    // overflow in the 256-bit multiply step.
    const arith_uint256 max_uint{~arith_uint256{}};
    arith_uint256 quotient{target};
    quotient /= target_u;

    arith_uint256 remainder{target - (quotient * target_u)};
    if (quotient > (max_uint / actual_u)) {
        return max_uint;
    }

    arith_uint256 scaled{quotient * actual_u};
    remainder *= actual_u;
    remainder /= target_u;

    if (scaled > (max_uint - remainder)) {
        return max_uint;
    }
    scaled += remainder;
    return scaled;
}

arith_uint256 ApplyDgwSlewGuard(
    arith_uint256 candidate_target,
    const arith_uint256& parent_target,
    int32_t next_height,
    const Consensus::Params& params)
{
    if (next_height < params.nDgwSlewGuardHeight) {
        return candidate_target;
    }

    // Limit easing: next target cannot become more than 4x easier than parent.
    const arith_uint256 max_ease_target = SaturatingLeftShift256(parent_target, NORMAL_SLEW_GUARD_SHIFT);
    if (candidate_target > max_ease_target) {
        candidate_target = max_ease_target;
    }

    // Limit hardening: next target cannot become more than 4x harder than parent.
    arith_uint256 min_harden_target = parent_target;
    min_harden_target >>= NORMAL_SLEW_GUARD_SHIFT;
    if (min_harden_target == 0) {
        min_harden_target = arith_uint256{1};
    }
    if (candidate_target < min_harden_target) {
        candidate_target = min_harden_target;
    }

    return candidate_target;
}

bool IsDisabledHeight(int32_t h)
{
    return h == std::numeric_limits<int32_t>::max();
}

bool IsMatMulAsertHalfLifeUpgradeConfigured(const Consensus::Params& params)
{
    return !IsDisabledHeight(params.nMatMulAsertHalfLifeUpgradeHeight);
}

int32_t LatestMatMulAsertPreUpgradeAnchorHeight(const CBlockIndex* pindexLast, const Consensus::Params& params)
{
    int32_t anchor_height = params.nMatMulAsertHeight;
    if (pindexLast == nullptr) {
        return anchor_height;
    }
    if (params.nMatMulAsertRetune2Height >= params.nMatMulAsertHeight &&
        pindexLast->nHeight >= params.nMatMulAsertRetune2Height) {
        anchor_height = params.nMatMulAsertRetune2Height;
    } else if (params.nMatMulAsertRetuneHeight >= params.nMatMulAsertHeight &&
               pindexLast->nHeight >= params.nMatMulAsertRetuneHeight) {
        anchor_height = params.nMatMulAsertRetuneHeight;
    }
    return anchor_height;
}

MatMulAsertHalfLifeInfo ResolveMatMulAsertHalfLifeInfo(
    const CBlockIndex* pindexLast,
    const Consensus::Params& params)
{
    MatMulAsertHalfLifeInfo info;
    info.current_half_life_s = params.nMatMulAsertHalfLife;
    info.current_anchor_height = LatestMatMulAsertPreUpgradeAnchorHeight(pindexLast, params);
    info.upgrade_configured = IsMatMulAsertHalfLifeUpgradeConfigured(params);
    info.upgrade_height = info.upgrade_configured ? params.nMatMulAsertHalfLifeUpgradeHeight : -1;
    info.upgrade_half_life_s = info.upgrade_configured ? params.nMatMulAsertHalfLifeUpgrade : params.nMatMulAsertHalfLife;

    if (info.upgrade_configured &&
        pindexLast != nullptr &&
        pindexLast->nHeight >= params.nMatMulAsertHalfLifeUpgradeHeight) {
        info.upgrade_active = true;
        info.current_half_life_s = params.nMatMulAsertHalfLifeUpgrade;
        info.current_anchor_height = params.nMatMulAsertHalfLifeUpgradeHeight;
    }

    return info;
}

bool ValidateMatMulAsertParams(const Consensus::Params& params, int32_t next_height)
{
    if (params.nMatMulAsertHalfLife <= 0) {
        LogWarning("MatMulAsert: invalid half-life=%lld at height %d, failing closed to powLimit\n",
                   static_cast<long long>(params.nMatMulAsertHalfLife), next_height);
        return false;
    }
    if (params.nPowTargetSpacing <= 0) {
        LogWarning("MatMulAsert: invalid target spacing=%lld at height %d, failing closed to powLimit\n",
                   static_cast<long long>(params.nPowTargetSpacing), next_height);
        return false;
    }
    if (params.nMatMulAsertBootstrapFactor == 0) {
        LogWarning("MatMulAsert: bootstrap factor is zero at height %d, failing closed to powLimit\n",
                   next_height);
        return false;
    }
    if (params.nMatMulAsertRetuneHardeningFactor == 0) {
        LogWarning("MatMulAsert: retune hardening factor is zero at height %d, failing closed to powLimit\n",
                   next_height);
        return false;
    }
    if (params.nMatMulAsertRetune2TargetNum == 0 || params.nMatMulAsertRetune2TargetDen == 0) {
        LogWarning("MatMulAsert: retune2 ratio is invalid (num=%u den=%u) at height %d, failing closed to powLimit\n",
                   params.nMatMulAsertRetune2TargetNum, params.nMatMulAsertRetune2TargetDen, next_height);
        return false;
    }

    const bool retune_enabled = !IsDisabledHeight(params.nMatMulAsertRetuneHeight);
    const bool retune2_enabled = !IsDisabledHeight(params.nMatMulAsertRetune2Height);
    if (retune_enabled && params.nMatMulAsertRetuneHeight < params.nMatMulAsertHeight) {
        LogWarning("MatMulAsert: retune height=%d is below ASERT activation=%d at height %d, failing closed to powLimit\n",
                   params.nMatMulAsertRetuneHeight, params.nMatMulAsertHeight, next_height);
        return false;
    }
    if (retune2_enabled && params.nMatMulAsertRetune2Height < params.nMatMulAsertHeight) {
        LogWarning("MatMulAsert: retune2 height=%d is below ASERT activation=%d at height %d, failing closed to powLimit\n",
                   params.nMatMulAsertRetune2Height, params.nMatMulAsertHeight, next_height);
        return false;
    }
    if (retune_enabled && retune2_enabled &&
        params.nMatMulAsertRetune2Height < params.nMatMulAsertRetuneHeight) {
        LogWarning("MatMulAsert: retune2 height=%d is below retune height=%d at height %d, failing closed to powLimit\n",
                   params.nMatMulAsertRetune2Height, params.nMatMulAsertRetuneHeight, next_height);
        return false;
    }
    if (IsMatMulAsertHalfLifeUpgradeConfigured(params)) {
        if (params.nMatMulAsertHalfLifeUpgrade <= 0) {
            LogWarning("MatMulAsert: half-life upgrade value=%lld is invalid at height %d, failing closed to powLimit\n",
                       static_cast<long long>(params.nMatMulAsertHalfLifeUpgrade), next_height);
            return false;
        }

        int32_t latest_pre_upgrade_anchor = params.nMatMulAsertHeight;
        if (retune_enabled) {
            latest_pre_upgrade_anchor = std::max(latest_pre_upgrade_anchor, params.nMatMulAsertRetuneHeight);
        }
        if (retune2_enabled) {
            latest_pre_upgrade_anchor = std::max(latest_pre_upgrade_anchor, params.nMatMulAsertRetune2Height);
        }
        if (params.nMatMulAsertHalfLifeUpgradeHeight <= latest_pre_upgrade_anchor) {
            LogWarning("MatMulAsert: half-life upgrade height=%d must be above latest prior anchor=%d at height %d, failing closed to powLimit\n",
                       params.nMatMulAsertHalfLifeUpgradeHeight, latest_pre_upgrade_anchor, next_height);
            return false;
        }
    }
    return true;
}

const CBlockIndex* FindGenesisBlockIndex(const CBlockIndex* tip)
{
    if (tip == nullptr || tip->nHeight < 0) {
        return nullptr;
    }

    // Avoid GetAncestor(0) on malformed/unlinked index chains. Header-sync
    // side branches can contain inconsistent pointers while being validated.
    const CBlockIndex* cursor = tip;
    int remaining_steps = tip->nHeight;
    while (cursor != nullptr && cursor->nHeight > 0) {
        const CBlockIndex* prev = cursor->pprev;
        if (prev == nullptr) {
            return nullptr;
        }
        if (prev->nHeight >= cursor->nHeight) {
            return nullptr;
        }
        cursor = prev;
        if (--remaining_steps < 0) {
            return nullptr;
        }
    }

    if (cursor == nullptr) return nullptr;
    if (cursor->nHeight != 0) return nullptr;
    if (cursor->pprev != nullptr) return nullptr;
    return cursor;
}

uint32_t FastMineBootstrapBits(const CBlockIndex* genesis, const Consensus::Params& params)
{
    assert(genesis != nullptr);

    const arith_uint256 pow_limit = UintToArith256(params.powLimit);
    arith_uint256 bootstrap_target;
    bootstrap_target.SetCompact(genesis->nBits);

    const uint32_t scale = std::max<uint32_t>(params.nFastMineDifficultyScale, 1U);
    if (scale > 1) {
        const arith_uint256 max_without_overflow = pow_limit / scale;
        if (bootstrap_target > max_without_overflow) {
            bootstrap_target = pow_limit;
        } else {
            bootstrap_target *= scale;
        }
    }

    bootstrap_target = ClampRetargetResult(bootstrap_target, pow_limit);
    return bootstrap_target.GetCompact();
}

int64_t DampenWarmupTimespanForRestartGap(int64_t observed_timespan, int64_t target_timespan)
{
    assert(target_timespan > 0);
    assert(WARMUP_RESTART_GAP_THRESHOLD_MULTIPLIER > 0);
    assert(WARMUP_RESTART_GAP_DAMPING_DIVISOR > 0);

    const int64_t threshold = target_timespan > std::numeric_limits<int64_t>::max() / WARMUP_RESTART_GAP_THRESHOLD_MULTIPLIER
        ? std::numeric_limits<int64_t>::max()
        : target_timespan * WARMUP_RESTART_GAP_THRESHOLD_MULTIPLIER;
    if (observed_timespan <= threshold) {
        return observed_timespan;
    }

    const int64_t excess = observed_timespan - target_timespan;
    return target_timespan + (excess / WARMUP_RESTART_GAP_DAMPING_DIVISOR);
}

arith_uint256 CalculateMatMulAsertTarget(
    const arith_uint256& anchor_target,
    int64_t time_diff,
    int64_t height_diff,
    int64_t half_life,
    const Consensus::Params& params)
{
    const arith_uint256 pow_limit{UintToArith256(params.powLimit)};
    if (anchor_target == 0 || anchor_target > pow_limit) {
        return pow_limit;
    }
    if (height_diff < 0) {
        LogWarning("CalculateMatMulAsertTarget: height_diff=%lld is negative, failing closed to powLimit\n",
                   static_cast<long long>(height_diff));
        return pow_limit;
    }
    if (half_life <= 0 || params.nPowTargetSpacing <= 0) {
        LogWarning("CalculateMatMulAsertTarget: invalid parameters (half_life=%lld target_spacing=%lld), failing closed to powLimit\n",
                   static_cast<long long>(half_life),
                   static_cast<long long>(params.nPowTargetSpacing));
        return pow_limit;
    }

    const int64_t target_spacing = params.nPowTargetSpacing;

    // aserti3-2d exponent:
    //   exponent = ((time_diff - target_spacing * (height_diff + 1)) * 2^16) / half_life
    const __int128 ideal_delta = static_cast<__int128>(target_spacing) *
        static_cast<__int128>(height_diff + 1);
    const __int128 exponent_input = static_cast<__int128>(time_diff) - ideal_delta;
    const __int128 exponent_scaled = exponent_input << ASERT_RADIX_BITS;
    const __int128 exponent_q = exponent_scaled / static_cast<__int128>(half_life);
    int64_t exponent;
    if (exponent_q > std::numeric_limits<int64_t>::max()) {
        exponent = std::numeric_limits<int64_t>::max();
    } else if (exponent_q < std::numeric_limits<int64_t>::min()) {
        exponent = std::numeric_limits<int64_t>::min();
    } else {
        exponent = static_cast<int64_t>(exponent_q);
    }

    const int64_t shifts = exponent >> ASERT_RADIX_BITS;
    const uint32_t frac = static_cast<uint32_t>(exponent) & ((1U << ASERT_RADIX_BITS) - 1U);

    const __int128 poly = static_cast<__int128>(ASERT_POLY_COEFF_1) * frac
        + static_cast<__int128>(ASERT_POLY_COEFF_2) * frac * frac
        + static_cast<__int128>(ASERT_POLY_COEFF_3) * frac * frac * frac
        + (static_cast<__int128>(1) << 47);
    const uint32_t factor = (1U << ASERT_RADIX_BITS) + static_cast<uint32_t>(poly >> 48);

    const int64_t net_shift = shifts - ASERT_RADIX_BITS;
    arith_uint256 next_target{};
    if (net_shift <= -256) {
        next_target = arith_uint256{0};
    } else if (net_shift < 0) {
        const unsigned int right_shift = static_cast<unsigned int>(-net_shift);
        const arith_uint256 max_uint{~arith_uint256{}};
        if (anchor_target > (max_uint / factor)) {
            // Near powLimit, anchor_target*factor can overflow 256 bits even
            // though the final right-shifted value is representable. Shift
            // first in that case to avoid saturation artifacts.
            arith_uint256 shifted_anchor{anchor_target};
            shifted_anchor >>= right_shift;
            next_target = SaturatingMultiplyByUint32(shifted_anchor, factor);
        } else {
            next_target = SaturatingMultiplyByUint32(anchor_target, factor);
            next_target >>= right_shift;
        }
    } else if (net_shift >= 256) {
        next_target = ~arith_uint256{};
    } else {
        next_target = SaturatingMultiplyByUint32(anchor_target, factor);
        if (net_shift > 0) {
            next_target = SaturatingLeftShift256(next_target, static_cast<unsigned int>(net_shift));
        }
    }

    if (next_target == 0) {
        next_target = arith_uint256{1};
    }
    return ClampRetargetResult(next_target, pow_limit);
}

unsigned int DarkGravityWaveLegacy(const CBlockIndex* pindexLast, const Consensus::Params& params)
{
    assert(pindexLast != nullptr);

    const arith_uint256 bnPowLimit = UintToArith256(params.powLimit);

    if (pindexLast->nHeight < DGW_PAST_BLOCKS) {
        return bnPowLimit.GetCompact();
    }

    const CBlockIndex* pindex = pindexLast;
    arith_uint256 bnPastTargetAvg;

    for (unsigned int nCountBlocks = 1; nCountBlocks <= DGW_PAST_BLOCKS; ++nCountBlocks) {
        if (pindex == nullptr) {
            return bnPowLimit.GetCompact();
        }

        const arith_uint256 bnTarget = arith_uint256{}.SetCompact(pindex->nBits);
        if (nCountBlocks == 1) {
            bnPastTargetAvg = bnTarget;
        } else {
            bnPastTargetAvg = (bnPastTargetAvg * nCountBlocks + bnTarget) / (nCountBlocks + 1);
        }

        if (nCountBlocks != DGW_PAST_BLOCKS) {
            if (pindex->pprev == nullptr) {
                return bnPowLimit.GetCompact();
            }
            pindex = pindex->pprev;
        }
    }

    arith_uint256 bnNew{bnPastTargetAvg};

    int64_t nActualTimespan = pindexLast->GetBlockTime() - pindex->GetBlockTime();
    const int64_t nTargetTimespan = DGW_PAST_BLOCKS * params.nPowTargetSpacing;

    if (nTargetTimespan <= 0) {
        LogWarning("DarkGravityWaveLegacy: nTargetTimespan=%lld is non-positive (nPowTargetSpacing=%lld), returning powLimit\n",
                   static_cast<long long>(nTargetTimespan), static_cast<long long>(params.nPowTargetSpacing));
        return bnPowLimit.GetCompact();
    }

    if (nActualTimespan < nTargetTimespan / 3) nActualTimespan = nTargetTimespan / 3;
    if (nActualTimespan > nTargetTimespan * 3) nActualTimespan = nTargetTimespan * 3;

    bnNew = ScaleTargetByTimespan(bnNew, nActualTimespan, nTargetTimespan);
    bnNew = ClampRetargetResult(bnNew, bnPowLimit);
    return bnNew.GetCompact();
}

[[maybe_unused]] unsigned int DarkGravityWaveMatMul(const CBlockIndex* pindexLast, const Consensus::Params& params)
{
    assert(pindexLast != nullptr);

    const arith_uint256 bnPowLimit = UintToArith256(params.powLimit);

    const CBlockIndex* genesis = FindGenesisBlockIndex(pindexLast);
    if (genesis == nullptr) {
        return bnPowLimit.GetCompact();
    }
    const int64_t next_height64 = static_cast<int64_t>(pindexLast->nHeight) + 1;
    if (next_height64 < 0 || next_height64 > std::numeric_limits<int32_t>::max()) {
        return bnPowLimit.GetCompact();
    }
    const int32_t next_height = static_cast<int32_t>(next_height64);

    const uint32_t bootstrap_bits = FastMineBootstrapBits(genesis, params);

    // Fast-mining bootstrap intentionally runs at fixed bootstrap difficulty.
    // DGW retargeting begins once the network enters normal spacing.
    if (next_height < params.nFastMineHeight) {
        return bootstrap_bits;
    }

    // Fresh-genesis MatMul networks hold bootstrap difficulty for heights 1..180.
    if (pindexLast->nHeight < DGW_PAST_BLOCKS) {
        return bootstrap_bits;
    }

    // Transition warmup: retarget from the immediate parent so difficulty can
    // converge quickly from fast bootstrap cadence toward the 90s normal target.
    if (next_height >= params.nFastMineHeight &&
        next_height < params.nFastMineHeight + 2 * DGW_PAST_BLOCKS) {
        if (next_height == params.nFastMineHeight) {
            return pindexLast->nBits;
        }

        arith_uint256 bnNew = arith_uint256{}.SetCompact(pindexLast->nBits);
        int64_t nActualTimespan = pindexLast->pprev
            ? pindexLast->GetBlockTime() - pindexLast->pprev->GetBlockTime()
            : params.nPowTargetSpacingNormal;
        const int64_t nTargetTimespan = params.nPowTargetSpacingNormal;
        const int64_t min_timespan = std::max<int64_t>(
            1,
            (nTargetTimespan * WARMUP_HARDENING_MIN_NUM) / WARMUP_HARDENING_MIN_DEN);
        const int64_t max_timespan = std::max<int64_t>(
            min_timespan,
            (nTargetTimespan * WARMUP_EASING_MAX_NUM) / WARMUP_EASING_MAX_DEN);

        // Damp parent-gap shocks (common after miner/node downtime) before
        // clamping to warmup bounds.
        nActualTimespan = DampenWarmupTimespanForRestartGap(nActualTimespan, nTargetTimespan);

        // Asymmetric warmup clamps: harden more slowly on fast blocks, but
        // ease faster on slow blocks so post-restart recovery does not stall.
        if (nActualTimespan < min_timespan) nActualTimespan = min_timespan;
        if (nActualTimespan > max_timespan) nActualTimespan = max_timespan;

        bnNew = ScaleTargetByTimespan(bnNew, nActualTimespan, nTargetTimespan);
        // Never allow warmup retargeting to become easier than the fast-phase
        // bootstrap target.
        arith_uint256 warmup_floor{};
        warmup_floor.SetCompact(bootstrap_bits);
        if (bnNew > warmup_floor) {
            bnNew = warmup_floor;
        }
        bnNew = ClampRetargetResult(bnNew, bnPowLimit);
        return bnNew.GetCompact();
    }

    const CBlockIndex* pindex = pindexLast;
    arith_uint256 bnPastTargetAvg;

    for (unsigned int nCountBlocks = 1; nCountBlocks <= DGW_PAST_BLOCKS; ++nCountBlocks) {
        if (pindex == nullptr) {
            return bnPowLimit.GetCompact();
        }

        const arith_uint256 bnTarget = arith_uint256{}.SetCompact(pindex->nBits);
        if (nCountBlocks == 1) {
            bnPastTargetAvg = bnTarget;
        } else {
            bnPastTargetAvg = (bnPastTargetAvg * nCountBlocks + bnTarget) / (nCountBlocks + 1);
        }

        if (nCountBlocks != DGW_PAST_BLOCKS) {
            if (pindex->pprev == nullptr) {
                return bnPowLimit.GetCompact();
            }
            pindex = pindex->pprev;
        }
    }

    arith_uint256 bnNew{bnPastTargetAvg};

    int64_t nActualTimespan = pindexLast->GetBlockTime() - pindex->GetBlockTime();
    const int64_t nTargetTimespan = ExpectedDgwTimespan(next_height, params);
    if (nTargetTimespan <= 0) {
        LogWarning("DarkGravityWaveMatMul: nTargetTimespan=%lld is non-positive at height %d, returning powLimit\n",
                   static_cast<long long>(nTargetTimespan), next_height);
        return bnPowLimit.GetCompact();
    }

    // Normal-phase DGW clamp profile:
    // - legacy: 2/3..3/2 (historic behavior)
    // - hardened v1: 3/4..2/1
    // - hardened v2: 3/4..3/1 (easing boost to reduce long slow tails after
    //   hashrate shock departures while preserving hardening floor).
    int64_t min_num = NORMAL_LEGACY_HARDENING_MIN_NUM;
    int64_t min_den = NORMAL_LEGACY_HARDENING_MIN_DEN;
    int64_t max_num = NORMAL_LEGACY_EASING_MAX_NUM;
    int64_t max_den = NORMAL_LEGACY_EASING_MAX_DEN;
    if (next_height >= params.nDgwAsymmetricClampHeight) {
        min_num = NORMAL_HARDENED_HARDENING_MIN_NUM;
        min_den = NORMAL_HARDENED_HARDENING_MIN_DEN;
        max_num = NORMAL_HARDENED_EASING_MAX_NUM;
        max_den = NORMAL_HARDENED_EASING_MAX_DEN;
        if (next_height >= params.nDgwEasingBoostHeight) {
            max_num = NORMAL_BOOSTED_EASING_MAX_NUM;
            max_den = NORMAL_BOOSTED_EASING_MAX_DEN;
        }
    }
    const int64_t min_timespan = std::max<int64_t>(1, (nTargetTimespan * min_num) / min_den);
    const int64_t max_timespan = std::max<int64_t>(min_timespan, (nTargetTimespan * max_num) / max_den);
    if (nActualTimespan < min_timespan) nActualTimespan = min_timespan;
    if (nActualTimespan > max_timespan) nActualTimespan = max_timespan;

    bnNew = ScaleTargetByTimespan(bnNew, nActualTimespan, nTargetTimespan);
    const arith_uint256 parent_target = arith_uint256{}.SetCompact(pindexLast->nBits);
    bnNew = ApplyDgwSlewGuard(bnNew, parent_target, next_height, params);
    bnNew = ClampRetargetResult(bnNew, bnPowLimit);
    return bnNew.GetCompact();
}

// DESIGN INVARIANT: MatMul networks use ASERT exclusively for difficulty
// adjustment. DarkGravityWave (DGW) must NOT be used for MatMul mining.
// The fast-mining bootstrap phase (blocks 0..nFastMineHeight-1) uses a fixed
// genesis-derived difficulty. From nFastMineHeight (== nMatMulAsertHeight)
// onward, ASERT governs all retargeting. This design was chosen because
// ASERT's stateless, path-independent algorithm avoids the convergence,
// oscillation, and warmup issues inherent to DGW. Do not modify this
// algorithm selection without explicit project approval.
unsigned int MatMulAsert(const CBlockIndex* pindexLast, const Consensus::Params& params)
{
    assert(pindexLast != nullptr);
    const arith_uint256 pow_limit{UintToArith256(params.powLimit)};

    const int64_t next_height64 = static_cast<int64_t>(pindexLast->nHeight) + 1;
    if (next_height64 < 0 || next_height64 > std::numeric_limits<int32_t>::max()) {
        return pow_limit.GetCompact();
    }
    const int32_t next_height = static_cast<int32_t>(next_height64);

    // Fast-mining bootstrap phase: hold fixed genesis-derived difficulty.
    // This replaces the former DGW-based warmup/transition logic.
    if (next_height < params.nMatMulAsertHeight) {
        const CBlockIndex* genesis = FindGenesisBlockIndex(pindexLast);
        if (genesis == nullptr) {
            return pow_limit.GetCompact();
        }
        return FastMineBootstrapBits(genesis, params);
    }

    if (!ValidateMatMulAsertParams(params, next_height)) {
        return pow_limit.GetCompact();
    }

    const uint32_t bootstrap_factor = params.nMatMulAsertBootstrapFactor;
    if (next_height == params.nMatMulAsertHeight) {
        arith_uint256 parent_target{};
        parent_target.SetCompact(pindexLast->nBits);
        arith_uint256 bootstrap_target{parent_target};
        if (bootstrap_factor > 1) {
            bootstrap_target = SaturatingMultiplyByUint32(bootstrap_target, bootstrap_factor);
        }
        bootstrap_target = ClampRetargetResult(bootstrap_target, pow_limit);
        return bootstrap_target.GetCompact();
    }

    const uint32_t retune_hardening_factor = params.nMatMulAsertRetuneHardeningFactor;
    if (next_height == params.nMatMulAsertRetuneHeight) {
        arith_uint256 parent_target{};
        parent_target.SetCompact(pindexLast->nBits);
        arith_uint256 retune_target{parent_target};
        if (retune_hardening_factor > 1) {
            retune_target /= retune_hardening_factor;
        }
        retune_target = ClampRetargetResult(retune_target, pow_limit);
        return retune_target.GetCompact();
    }

    const uint32_t retune2_num = params.nMatMulAsertRetune2TargetNum;
    const uint32_t retune2_den = params.nMatMulAsertRetune2TargetDen;
    if (next_height == params.nMatMulAsertRetune2Height) {
        arith_uint256 parent_target{};
        parent_target.SetCompact(pindexLast->nBits);
        arith_uint256 retune2_target = ScaleTargetByTimespan(
            parent_target,
            static_cast<int64_t>(retune2_num),
            static_cast<int64_t>(retune2_den));
        retune2_target = ClampRetargetResult(retune2_target, pow_limit);
        return retune2_target.GetCompact();
    }

    if (next_height == params.nMatMulAsertHalfLifeUpgradeHeight) {
        arith_uint256 parent_target{};
        parent_target.SetCompact(pindexLast->nBits);
        parent_target = ClampRetargetResult(parent_target, pow_limit);
        return parent_target.GetCompact();
    }

    // ASERT anchor:
    // - base anchor is first ASERT block (activation block itself)
    // - after optional target retunes, re-anchor on the latest retune block to
    //   preserve one-time adjustments as the ASERT baseline
    // - after the optional half-life upgrade, re-anchor on the upgrade block so
    //   the new half-life applies prospectively instead of retroactively.
    const MatMulAsertHalfLifeInfo half_life_info = ResolveMatMulAsertHalfLifeInfo(pindexLast, params);
    const int32_t anchor_height = half_life_info.current_anchor_height;
    if (anchor_height < 0 || pindexLast->nHeight < anchor_height) {
        return pow_limit.GetCompact();
    }
    const CBlockIndex* anchor = pindexLast->GetAncestor(anchor_height);
    if (anchor == nullptr) {
        return pow_limit.GetCompact();
    }

    arith_uint256 anchor_target{};
    anchor_target.SetCompact(anchor->nBits);
    if (anchor_target == 0 || anchor_target > pow_limit) {
        anchor_target = pow_limit;
    }
    const int64_t time_diff = pindexLast->GetBlockTime() - anchor->GetBlockTime();
    const int64_t height_diff = static_cast<int64_t>(pindexLast->nHeight) - anchor->nHeight;
    const arith_uint256 next_target = CalculateMatMulAsertTarget(
        anchor_target,
        time_diff,
        height_diff,
        half_life_info.current_half_life_s,
        params);
    return next_target.GetCompact();
}
} // namespace

MatMulAsertHalfLifeInfo GetMatMulAsertHalfLifeInfo(const CBlockIndex* pindexLast, const Consensus::Params& params)
{
    return ResolveMatMulAsertHalfLifeInfo(pindexLast, params);
}


// ---- solve telemetry ------------------------------------------------------
// ENC_RC episodes completed. This is the RC path's proof of life: at full tilt
// a miner that never finds a share is otherwise indistinguishable from a wedged
// one (the v3 counters this replaced read zero at RC heights by construction).
std::atomic<uint64_t> g_matmul_rc_episodes{0};
// Solve windows entered (one per SolveMatMul call that reached a solver).
std::atomic<uint64_t> g_matmul_solve_windows{0};

MatMulSolvePipelineStats ProbeMatMulSolvePipelineStats()
{
    MatMulSolvePipelineStats stats;
    stats.rc_episodes = g_matmul_rc_episodes.load(std::memory_order_relaxed);
    stats.solve_windows = g_matmul_solve_windows.load(std::memory_order_relaxed);
    return stats;
}

void ResetMatMulSolvePipelineStats()
{
    g_matmul_rc_episodes.store(0, std::memory_order_relaxed);
    g_matmul_solve_windows.store(0, std::memory_order_relaxed);
}

int64_t ExpectedDgwTimespan(int32_t height, const Consensus::Params& params)
{
    const int64_t interval_count =
        (height >= params.nDgwWindowAlignmentHeight && DGW_PAST_BLOCKS > 1)
        ? (DGW_PAST_BLOCKS - 1)
        : DGW_PAST_BLOCKS;
    if (height < params.nFastMineHeight) {
        return (interval_count * params.nPowTargetSpacingFastMs) / 1000;
    }
    return interval_count * params.nPowTargetSpacingNormal;
}

bool EnforceTimewarpProtectionAtHeight(const Consensus::Params& params, int32_t block_height)
{
    if (!params.enforce_BIP94 || block_height <= 0) {
        return false;
    }

    // Per-block retargeting engines need per-block timestamp protection.
    if (!params.fPowNoRetargeting) {
        if (params.fMatMulPOW) {
            return true;
        }
        if (params.fKAWPOW && block_height >= params.nKAWPOWHeight) {
            return true;
        }
    }

    return block_height % params.DifficultyAdjustmentInterval() == 0;
}

unsigned int GetNextWorkRequired(const CBlockIndex* pindexLast, const CBlockHeader *pblock, const Consensus::Params& params)
{
    assert(pindexLast != nullptr);
    unsigned int nProofOfWorkLimit = UintToArith256(params.powLimit).GetCompact();
    const int64_t next_height = static_cast<int64_t>(pindexLast->nHeight) + 1;
    if (next_height < 0 || next_height > std::numeric_limits<int>::max()) {
        return nProofOfWorkLimit;
    }

    if (params.fPowNoRetargeting) {
        return pindexLast->nBits;
    }

    if (params.fMatMulPOW) {
        // DESIGN INVARIANT: MatMul networks use ASERT exclusively for all
        // difficulty adjustment after the fast-mining bootstrap phase.
        // DarkGravityWave (DGW) is NOT used for MatMul mining. Do not
        // reintroduce DGW routing here -- it was deliberately replaced by
        // ASERT to avoid convergence and oscillation issues inherent to DGW.
        return MatMulAsert(pindexLast, params);
    }

    if (params.fKAWPOW && next_height >= params.nKAWPOWHeight) {
        return DarkGravityWaveLegacy(pindexLast, params);
    }

    // Only change once per difficulty adjustment interval
    if (next_height % params.DifficultyAdjustmentInterval() != 0)
    {
        if (params.fPowAllowMinDifficultyBlocks)
        {
            // Special difficulty rule for testnet:
            // If the new block's timestamp is more than 2* 10 minutes
            // then allow mining of a min-difficulty block.
            if (pblock->GetBlockTime() > pindexLast->GetBlockTime() + params.nPowTargetSpacing*2)
                return nProofOfWorkLimit;
            else
            {
                // Return the last non-special-min-difficulty-rules-block
                const CBlockIndex* pindex = pindexLast;
                while (pindex->pprev && pindex->nHeight % params.DifficultyAdjustmentInterval() != 0 && pindex->nBits == nProofOfWorkLimit)
                    pindex = pindex->pprev;
                return pindex->nBits;
            }
        }
        return pindexLast->nBits;
    }

    // Go back by what we want to be 14 days worth of blocks
    int nHeightFirst = pindexLast->nHeight - (params.DifficultyAdjustmentInterval()-1);
    assert(nHeightFirst >= 0);
    const CBlockIndex* pindexFirst = pindexLast->GetAncestor(nHeightFirst);
    assert(pindexFirst);

    return CalculateNextWorkRequired(pindexLast, pindexFirst->GetBlockTime(), params);
}

unsigned int CalculateNextWorkRequired(const CBlockIndex* pindexLast, int64_t nFirstBlockTime, const Consensus::Params& params)
{
    if (params.fPowNoRetargeting)
        return pindexLast->nBits;

    // Limit adjustment step
    int64_t nActualTimespan = pindexLast->GetBlockTime() - nFirstBlockTime;
    if (nActualTimespan < params.nPowTargetTimespan/4)
        nActualTimespan = params.nPowTargetTimespan/4;
    if (nActualTimespan > params.nPowTargetTimespan*4)
        nActualTimespan = params.nPowTargetTimespan*4;

    // Retarget
    const arith_uint256 bnPowLimit = UintToArith256(params.powLimit);
    arith_uint256 bnNew;

    // Special difficulty rule for Testnet4
    if (params.enforce_BIP94) {
        // Here we use the first block of the difficulty period. This way
        // the real difficulty is always preserved in the first block as
        // it is not allowed to use the min-difficulty exception.
        int nHeightFirst = pindexLast->nHeight - (params.DifficultyAdjustmentInterval()-1);
        const CBlockIndex* pindexFirst = nHeightFirst >= 0 ? pindexLast->GetAncestor(nHeightFirst) : nullptr;
        bnNew.SetCompact((pindexFirst != nullptr ? pindexFirst : pindexLast)->nBits);
    } else {
        bnNew.SetCompact(pindexLast->nBits);
    }

    bnNew *= nActualTimespan;
    bnNew /= params.nPowTargetTimespan;

    bnNew = ClampRetargetResult(bnNew, bnPowLimit);

    return bnNew.GetCompact();
}

// Check that on difficulty adjustments, the new difficulty does not increase
// or decrease beyond the permitted limits.
bool PermittedDifficultyTransition(const Consensus::Params& params, int64_t height, uint32_t old_nbits, uint32_t new_nbits)
{
    if (params.fMatMulPOW) {
        auto old_target = DeriveTarget(old_nbits, params.powLimit);
        auto new_target = DeriveTarget(new_nbits, params.powLimit);
        if (!old_target || !new_target) return false;

        // Presync sanity bounds for ASERT headers: do not allow per-block jumps
        // beyond 4x in either direction.
        const arith_uint256 pow_limit = UintToArith256(params.powLimit);

        arith_uint256 easier_bound{*old_target};
        if (easier_bound > (pow_limit / 4)) {
            easier_bound = pow_limit;
        } else {
            easier_bound *= 4;
        }
        // Compare against the compact-rounded bound because headers encode
        // difficulty via compact nBits.
        arith_uint256 max_new_target;
        max_new_target.SetCompact(easier_bound.GetCompact());
        if (*new_target > max_new_target) return false;

        arith_uint256 harder_bound{*old_target};
        harder_bound /= 4;
        if (harder_bound == 0) harder_bound = arith_uint256{1};
        arith_uint256 min_new_target;
        min_new_target.SetCompact(harder_bound.GetCompact());
        if (*new_target < min_new_target) return false;

        return true;
    }

    if (params.fPowAllowMinDifficultyBlocks) return true;

    if (height % params.DifficultyAdjustmentInterval() == 0) {
        int64_t smallest_timespan = params.nPowTargetTimespan/4;
        int64_t largest_timespan = params.nPowTargetTimespan*4;

        const arith_uint256 pow_limit = UintToArith256(params.powLimit);
        arith_uint256 observed_new_target;
        observed_new_target.SetCompact(new_nbits);

        // Calculate the largest difficulty value possible:
        arith_uint256 largest_difficulty_target;
        largest_difficulty_target.SetCompact(old_nbits);
        largest_difficulty_target *= largest_timespan;
        largest_difficulty_target /= params.nPowTargetTimespan;

        if (largest_difficulty_target > pow_limit) {
            largest_difficulty_target = pow_limit;
        }

        // Round and then compare this new calculated value to what is
        // observed.
        arith_uint256 maximum_new_target;
        maximum_new_target.SetCompact(largest_difficulty_target.GetCompact());
        if (maximum_new_target < observed_new_target) return false;

        // Calculate the smallest difficulty value possible:
        arith_uint256 smallest_difficulty_target;
        smallest_difficulty_target.SetCompact(old_nbits);
        smallest_difficulty_target *= smallest_timespan;
        smallest_difficulty_target /= params.nPowTargetTimespan;

        if (smallest_difficulty_target > pow_limit) {
            smallest_difficulty_target = pow_limit;
        }

        // Round and then compare this new calculated value to what is
        // observed.
        arith_uint256 minimum_new_target;
        minimum_new_target.SetCompact(smallest_difficulty_target.GetCompact());
        if (minimum_new_target > observed_new_target) return false;
    } else if (old_nbits != new_nbits) {
        return false;
    }
    return true;
}

// Bypasses the actual proof of work check during fuzz testing with a simplified validation checking whether
// the most significant bit of the last byte of the hash is set.
bool CheckProofOfWork(uint256 hash, unsigned int nBits, const Consensus::Params& params)
{
    if constexpr (G_FUZZING) return (hash.data()[31] & 0x80) == 0;
    return CheckProofOfWorkImpl(hash, nBits, params);
}

std::optional<arith_uint256> DeriveTarget(unsigned int nBits, const uint256 pow_limit)
{
    bool fNegative;
    bool fOverflow;
    arith_uint256 bnTarget;

    bnTarget.SetCompact(nBits, &fNegative, &fOverflow);

    // Check range
    if (fNegative || bnTarget == 0 || fOverflow || bnTarget > UintToArith256(pow_limit))
        return {};

    return bnTarget;
}

bool CheckProofOfWorkImpl(uint256 hash, unsigned int nBits, const Consensus::Params& params)
{
    auto bnTarget{DeriveTarget(nBits, params.powLimit)};
    if (!bnTarget) return false;

    // Check proof of work matches claimed amount
    if (UintToArith256(hash) > bnTarget)
        return false;

    return true;
}

// Generic height -> active matmul solve variant. The single place a block height maps to a solve
// path; future height-forked axes slot in here so no caller hardcodes a per-variant height test.
MatMulSolveVariant ActiveMatMulSolveVariant(const Consensus::Params& params, int32_t block_height)
{
    // Coupled binds an episode leg, so it is the strictest fork and supersedes bare RC where both
    // are active; check it first (its activation height is >= the RC height by construction).
    if (params.IsMatMulRCCoupledActive(block_height)) return MatMulSolveVariant::Coupled;
    if (params.IsMatMulRCActive(block_height)) return MatMulSolveVariant::RC;
    return MatMulSolveVariant::Base;
}

extern std::atomic<uint64_t> g_matmul_rc_episodes;   // defined near ProbeMatMulSolvePipelineStats

// ENC_RC episode digest solve loop. Mirrors the consensus reference SolveMatMulV4RC
// (btx PR#89 src/pow.cpp): per nonce, pin seeds -> DeriveSigma -> ComputeEpisodeDigest ->
// compare vs effective_target. Winner (digest <= target) sets block.matmul_digest, returns true;
// otherwise advances the nonce and decrements max_tries until exhausted.
//
// This is the DIGEST contract ONLY. It is byte-exact to RecomputeResidentCurriculumReference
// (our ComputeEpisodeDigest is that reference), so no separate reseal is needed. The datacenter
// profile (mainnet default) additionally requires the winner to emit the succinct Freivalds
// sampled carrier (ProveWinnerEpisodeV7) or the block fails closed at verify -- that prover is an
// EXTERNAL blocker (see research/matmul-v4/RC-MINER-INTEGRATION.md) and is NOT built here. The
// caller gates whether to run this at all.
bool SolveMatMulRCEpisode(CBlockHeader& block, const Consensus::Params& params,
                          uint64_t& max_tries, int32_t block_height,
                          const std::atomic<bool>* abort_flag,
                          std::optional<int64_t> parent_median_time_past,
                          const arith_uint256& effective_target,
                          const matmul::v4::rc::EpisodeParams& ep)
{
    // Digest backend: prefer the GPU (~200x at datacenter dims) when the CUDA episode backend is
    // linked and a device is present; the digest is byte-exact to the CPU oracle (gated by
    // rc_gpu_accel_probe: GPU==CPU==golden). BTX_RC_EPISODE_CPU=1 forces the CPU oracle (debug/
    // determinism cross-check). No CUDA build -> always CPU.
    bool use_gpu = false;
#ifdef MATADOR_ENABLE_CUDA
    use_gpu = std::getenv("BTX_RC_EPISODE_CPU") == nullptr && matmul::v4::rc::RCEpisodeGpuAvailable();
#endif
    // RC-DIAG (2026-08-09): live rigs latch, peg the GPU and complete hundreds of episodes while
    // submitting ZERO shares. The target arithmetic says otherwise, so print the ACTUAL numbers:
    // the target we compare against and the best (lowest) digest we have actually produced. If the
    // best digest hovers near 2^256 the digests are wrong; if it is well under the target and we
    // still never return, the compare/plumbing is wrong. Off unless BTX_RC_DIAG=1.
    const bool rc_diag = std::getenv("BTX_RC_DIAG") != nullptr;
    arith_uint256 best_digest = ~arith_uint256();   // all ones = "no digest seen yet"
    uint64_t diag_n = 0;
    if (rc_diag) {
        fprintf(stderr, "RC-DIAG height=%d target=%s gpu=%d\n",
                block_height, ArithToUint256(effective_target).GetHex().c_str(), use_gpu ? 1 : 0);
        fflush(stderr);
    }
    while (max_tries > 0) {
        if (abort_flag != nullptr && abort_flag->load(std::memory_order_relaxed)) return false;
        if (!SetDeterministicMatMulSeeds(block, params, block_height, parent_median_time_past)) {
            return false;
        }
        const uint256 sigma = matmul::DeriveSigma(block);
        uint256 mined;
#ifdef MATADOR_ENABLE_CUDA
        if (use_gpu) mined = matmul::v4::rc::ComputeEpisodeDigestGPU(sigma, ep);
        else
#endif
            mined = matmul::v4::rc::ComputeEpisodeDigest(sigma, ep);
        // Proof of life for the RC path: the v3 telemetry counters stay flat here, so without
        // this an RC miner at full tilt is indistinguishable from a wedged one.
        g_matmul_rc_episodes.fetch_add(1, std::memory_order_relaxed);
        if (mined.IsNull()) return false;               // malformed dims -> reject, never grind blind
        if (rc_diag) {
            const arith_uint256 m = UintToArith256(mined);
            if (m < best_digest) best_digest = m;
            if ((++diag_n % 1) == 0) {   // RC-DIAG: every episode (~1/s, cheap)
                fprintf(stderr, "RC-DIAG n=%llu nonce=%llu last=%s best=%s target=%s le=%d\n",
                          static_cast<unsigned long long>(diag_n),
                          static_cast<unsigned long long>(block.nNonce64),
                          mined.GetHex().c_str(),
                          ArithToUint256(best_digest).GetHex().c_str(),
                          ArithToUint256(effective_target).GetHex().c_str(),
                          (m <= effective_target) ? 1 : 0);
                fflush(stderr);
            }
        }
        if (UintToArith256(mined) <= effective_target) {
            block.matmul_digest = mined;                // winner (digest is already the reseal)
            return true;
        }
        --max_tries;
        if (block.nNonce64 == std::numeric_limits<uint64_t>::max()) return false;
        ++block.nNonce64;
        block.nNonce = static_cast<uint32_t>(block.nNonce64);
    }
    return false;
}

// ENC_RC COUPLED (V3) digest solve loop. Mirrors SolveMatMulRCEpisode but grinds the coupled-V3
// digest. The bank (pages + commitment root) keys off RCBankTemplateHash(block) which is
// nonce-independent, so it is BUILT ONCE per template via BuildCoupledBankV3 (~96 GiB dequant
// retention at production dims) and every nonce only pays the per-nonce leg
// (ComputeCoupledDigestV3WithBank) -- the old one-shot ComputeCoupledDigestV3 re-derived the
// whole bank per nonce, which was the dominant "minutes/nonce" cost. Byte-exact to the golden
// a4bb0cc4 oracle, which is the CPU reference the future GPU coupled solver will diff against.
// CPU-only today: GPU acceleration is a separate build for mainnet throughput.
bool SolveMatMulRCCoupled(CBlockHeader& block, const Consensus::Params& params,
                          uint64_t& max_tries, int32_t block_height,
                          const std::atomic<bool>* abort_flag,
                          std::optional<int64_t> parent_median_time_past,
                          const arith_uint256& effective_target,
                          const matmul::v4::rc::CoupParamsV3& cp)
{
    const uint256 bank_tmpl = matmul::v4::rc::RCBankTemplateHash(block);   // nonce-independent
    // Backend: prefer the GPU coupled backend (device-derived bank, template-cached in VRAM when
    // it fits; byte-exact to the CPU oracle, gated by rc_gpu_coupled_probe). BTX_RC_COUPLED_CPU=1
    // forces the CPU oracle (debug/determinism cross-check). No CUDA build -> always CPU. The
    // CPU bank (~96 GiB dequant at production dims) is only built on the CPU path.
    bool coup_use_gpu = false;
#ifdef MATADOR_ENABLE_CUDA
    coup_use_gpu =
        std::getenv("BTX_RC_COUPLED_CPU") == nullptr && matmul::v4::rc::RCCoupledGpuAvailable();
#endif
    const auto bank_t0 = std::chrono::steady_clock::now();
    matmul::v4::rc::CoupledBankV3 bank;
    uint256 bank_root;
#ifdef MATADOR_ENABLE_CUDA
    if (coup_use_gpu) {
        bank_root = matmul::v4::rc::ComputeCoupledBankRootGPU(
            bank_tmpl, static_cast<uint32_t>(block_height), cp);
    } else
#endif
    {
        bank = matmul::v4::rc::BuildCoupledBankV3(
            bank_tmpl, static_cast<uint32_t>(block_height), cp);
        bank_root = bank.bank_root;
    }
    const auto bank_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - bank_t0)
                             .count();
    LogPrintf("[rc-coupled] bank root ready (backend=%s): pages=%u page_bytes=%u in %lld ms\n",
              coup_use_gpu ? "gpu" : "cpu", cp.bank_pages, cp.lobe_width * cp.lobe_width,
              static_cast<long long>(bank_ms));
    while (max_tries > 0) {
        if (abort_flag != nullptr && abort_flag->load(std::memory_order_relaxed)) return false;
        if (!SetDeterministicMatMulSeeds(block, params, block_height, parent_median_time_past)) {
            return false;
        }
        const uint256 sigma = matmul::DeriveSigma(block);
        uint256 mined;
#ifdef MATADOR_ENABLE_CUDA
        if (coup_use_gpu)
            mined = matmul::v4::rc::ComputeCoupledDigestV3GPU(
                sigma, bank_tmpl, static_cast<uint32_t>(block_height), bank_root, cp);
        else
#endif
            mined = matmul::v4::rc::ComputeCoupledDigestV3WithBank(sigma, bank, cp);
        if (mined.IsNull()) return false;               // malformed dims -> reject, never grind blind
        if (UintToArith256(mined) <= effective_target) {
            block.matmul_digest = mined;                // winner (digest is already the reseal)
            return true;
        }
        --max_tries;
        if (block.nNonce64 == std::numeric_limits<uint64_t>::max()) return false;
        ++block.nNonce64;
        block.nNonce = static_cast<uint32_t>(block.nNonce64);
    }
    return false;
}

// Gate for the RC episode digest solve loop. DEFAULT ON since v0.9.2: the v4.7 Epoch A
// activation tuple is PROFILE 1 with ExactReplay authority (PR#97) -- digest <= target IS the
// whole proof (the pre-hash gate is retired at v4 heights and no succinct carrier is required),
// so a mined episode digest is fully block-eligible and the old fail-closed posture would idle
// the fleet at activation. BTX_RC_ENABLE_EPISODE_SOLVE=0 force-disables (restores the pre-0.9.2
// behavior); unset or any other value = enabled. Note this branch is only reachable at RC-active
// heights (vendored params default nMatMulRCHeight=INT32_MAX; see BTX_MATMUL_RC_HEIGHT latch).
static bool RCEpisodeSolveEnabled()
{
    const char* v = std::getenv("BTX_RC_ENABLE_EPISODE_SOLVE");
    return v == nullptr || !(v[0] == '0' && v[1] == '\0');
}

// Gate for the RC COUPLED (V3) digest solve loop. Same rationale as RCEpisodeSolveEnabled: OFF by
// default so a finite coupled height on a public net does not silently grind a ~48 GiB bank puzzle
// that cannot yet produce an acceptable block. Set BTX_RC_ENABLE_COUPLED_SOLVE=1 for regtest
// end-to-end / once the coupled carrier + pool format exist.
static bool RCCoupledSolveEnabled()
{
    const char* v = std::getenv("BTX_RC_ENABLE_COUPLED_SOLVE");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

bool SolveMatMul(CBlockHeader& block, const Consensus::Params& params, uint64_t& max_tries,
                 int32_t block_height,
                 const std::atomic<bool>* abort_flag,
                 const uint256* share_target_override,
                 std::optional<int64_t> parent_median_time_past,
                 const std::function<bool(const CBlockHeader&)>* share_sink)
{
    if (!params.fMatMulPOW) return false;
    if (max_tries == 0) return false;
    g_matmul_solve_windows.fetch_add(1, std::memory_order_relaxed);
    // Height-gated matmul switch. ENC_RC (Resident Curriculum) is the live consensus solver at and
    // above nMatMulRCHeight; Coupled supersedes it where both are active. Below the RC height there
    // is no solver left to fall through to -- see the fail-closed arm at the end.
    const MatMulSolveVariant solve_variant = ActiveMatMulSolveVariant(params, block_height);
    if (solve_variant == MatMulSolveVariant::Coupled) {
        // Coupled (V3) digest solve loop -- BUILT and unit-tested (rc_coupled_solve_probe), GATED OFF
        // by default. Coupled supersedes bare RC where both heights are active; it binds an episode
        // leg to the ~48 GiB bank puzzle. Same fail-closed posture as the episode branch: on a public
        // net a mined coupled digest still needs the carrier the pool/consensus format defines, so we
        // do not grind unless explicitly enabled. See research/matmul-v4/RC-MINER-INTEGRATION.md.
        if (RCCoupledSolveEnabled()) {
            auto coup_bnTarget = DeriveTarget(block.nBits, params.powLimit);
            if (!coup_bnTarget) return false;
            arith_uint256 coup_target = *coup_bnTarget;
            if (share_target_override != nullptr) {
                coup_target = UintToArith256(*share_target_override);
                if (coup_target == 0) return false;
            }
            // Mainnet activation shape. A regtest/profile override slots in here once the profile is
            // carried in the vendored consensus params (currently only the height is).
            const auto cp = matmul::v4::rc::ProductionV3CoupParams();
            if (SolveMatMulRCCoupled(block, params, max_tries, block_height, abort_flag,
                                     parent_median_time_past, coup_target, cp)) {
                LogPrintf("SolveMatMulRCCoupled: coupled digest winner at height %d nonce=%llu -- "
                          "digest valid, but public-net acceptance ALSO needs the carrier/pool RC "
                          "format (not built); block fails closed at verify without it.\n",
                          block_height, static_cast<unsigned long long>(block.nNonce64));
                return true;
            }
            return false;
        }
        static std::atomic<int64_t> s_last_coup_warn_us{0};
        const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
        int64_t prev = s_last_coup_warn_us.load(std::memory_order_relaxed);
        if (now_us - prev >= 60'000'000 &&
            s_last_coup_warn_us.compare_exchange_strong(prev, now_us, std::memory_order_relaxed)) {
            LogPrintf("MATMUL RC COUPLED ACTIVE at height %d but coupled solve is GATED OFF -- mining "
                      "is HALTED (fail-closed). Set BTX_RC_ENABLE_COUPLED_SOLVE=1 once a carrier / "
                      "pool RC format is available. ZERO shares until then.\n",
                      block_height);
        }
        return false;
    }
    if (solve_variant == MatMulSolveVariant::RC) {
        // Episode digest solve loop -- BUILT, unit-tested (rc_solve_probe), and DEFAULT ON since
        // v0.9.2. Epoch A (v4.7 / PR#97) fixes the activation tuple to Profile 1 with ExactReplay
        // authority: digest <= target is the entire lottery, so a mined digest is block-eligible
        // as-is. See research/matmul-v4/RC-MINER-INTEGRATION.md.
        if (RCEpisodeSolveEnabled()) {
            auto rc_bnTarget = DeriveTarget(block.nBits, params.powLimit);
            if (!rc_bnTarget) return false;
            arith_uint256 rc_target = *rc_bnTarget;
            if (share_target_override != nullptr) {
                rc_target = UintToArith256(*share_target_override);
                if (rc_target == 0) {
                    // A zero share target silently produced an idle GPU, no allocation and no
                    // submissions -- indistinguishable from a broken dispatcher, and it cost a
                    // partner a day of misdiagnosis. Never fail quiet here.
                    static std::atomic<int64_t> s_last_zero_target_us{0};
                    const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                               std::chrono::steady_clock::now().time_since_epoch()).count();
                    int64_t prev = s_last_zero_target_us.load(std::memory_order_relaxed);
                    if (now_us - prev >= 60'000'000 &&
                        s_last_zero_target_us.compare_exchange_strong(prev, now_us, std::memory_order_relaxed)) {
                        LogPrintf("MATMUL RC at height %d: the pool's share target is ZERO, so no digest can "
                                  "ever satisfy it and the GPU stays idle. This is a JOB problem, not a solver "
                                  "problem -- check the target field the pool sends on its RC jobs.\n",
                                  block_height);
                    }
                    return false;
                }
            }
            // ACTIVE launch profile: PR#97 (v4.7) fixes the Epoch A activation tuple to PROFILE 1
            // (ProductionEpisodeParams, ExactReplay authority) -- the old profile-2 default here
            // would have ground the WRONG, ~6x more expensive shape at Phase-1 activation.
            // BTX_MATMUL_RC_PROFILE=2 selects the datacenter shape for the later Epoch D.
            const auto ep = matmul::v4::rc::ActiveProfileEpisodeParams();
            if (SolveMatMulRCEpisode(block, params, max_tries, block_height, abort_flag,
                                     parent_median_time_past, rc_target, ep)) {
                LogPrintf("SolveMatMulRCEpisode: episode digest winner at height %d nonce=%llu "
                          "(Epoch A / Profile 1: digest <= target is the full proof, "
                          "ExactReplay-verifiable).\n",
                          block_height, static_cast<unsigned long long>(block.nNonce64));
                return true;
            }
            return false;
        }
        // Default: FAIL CLOSED, but LOUDLY. A silent return here would idle the rig the instant an RC
        // height goes finite on a public net, and we would only notice via acc/hr drifting to 0.
        // Throttle to once per ~60 s so the log is unmissable without flooding the solve loop.
        static std::atomic<int64_t> s_last_warn_us{0};
        const int64_t now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
        int64_t prev = s_last_warn_us.load(std::memory_order_relaxed);
        if (now_us - prev >= 60'000'000 &&
            s_last_warn_us.compare_exchange_strong(prev, now_us, std::memory_order_relaxed)) {
            LogPrintf("MATMUL RC ACTIVE at height %d but RC episode solve is DISABLED by "
                      "BTX_RC_ENABLE_EPISODE_SOLVE=0 -- mining is HALTED (fail-closed). Unset it "
                      "(default ON since v0.9.2) to mine RC. ZERO shares until then.\n",
                      block_height);
        }
        return false;
    }
    // Base (v3) is GONE. Mainnet activated ENC_RC, so a v3 digest is unmineable
    // and unacceptable -- the grind pipeline and its verify paths were removed
    // rather than left as dead weight. Reaching here means the height resolved
    // below the RC activation, which on a live chain only happens if an operator
    // pinned a bogus --rc-height or a pool announced a pre-RC job. Fail closed
    // and say so, rather than silently producing nothing.
    static std::atomic<int64_t> s_last_base_warn_us{0};
    const int64_t base_now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
    int64_t base_prev = s_last_base_warn_us.load(std::memory_order_relaxed);
    if (base_now_us - base_prev >= 60'000'000 &&
        s_last_base_warn_us.compare_exchange_strong(base_prev, base_now_us, std::memory_order_relaxed)) {
        LogPrintf("SolveMatMul: height %d resolves to the pre-RC (v3) variant, which this build no "
                  "longer implements -- v4/ENC_RC is the only solver. Mining is HALTED. Check the "
                  "RC activation height (--rc-height / BTX_MATMUL_RC_HEIGHT) and the pool's job "
                  "profile. ZERO shares until then.\n",
                  block_height);
    }
    return false;
}
