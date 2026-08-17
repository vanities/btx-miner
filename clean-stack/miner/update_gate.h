// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.
#ifndef MATADOR_MINER_UPDATE_GATE_H
#define MATADOR_MINER_UPDATE_GATE_H

#include "version_compare.h"

#include <cstdint>
#include <string>

// Outcome of the auto-update adopt gate. RunUpdateCheck() turns each into a log line; only Adopt
// proceeds to download/verify/swap. Order of the guards below is the order RunUpdateCheck applies.
enum class UpdateDecision {
    UpToDate,       // candidate is not strictly newer than current (or empty tag) -> nothing to do
    AutoUpdateOff,  // newer exists, but auto_update is disabled -> notify only
    NoAsset,        // newer exists, but no binary asset for this platform -> notify only
    Baking,         // newer exists, but younger than min_version_age_s -> hold, adopt on a later check
    StampMismatch,  // already re-exec'd into this tag yet still report current -> stop (bad version stamp)
    Adopt,          // newer, allowed, has asset, baked, not a re-exec loop -> adopt it
};

// Pure decision for whether to auto-adopt `candidate_tag` over `current_tag`. No I/O, no clock, no
// getenv -- all inputs are passed in (the caller reads published-at age + MATADOR_UPDATED_TO). This
// is the exact gate that shipped the v0.6.10->v0.6.9 lexical-downgrade bug, so it is unit-tested
// (clean-stack/harness/update_gate_test.cpp). `candidate_age_s` < 0 means "age unknown" (never blocks
// on bake-time). `already_updated_to` is the MATADOR_UPDATED_TO env value, or nullptr if unset.
inline UpdateDecision DecideUpdate(const std::string& candidate_tag,
                                   const std::string& current_tag,
                                   bool auto_update,
                                   bool has_platform_asset,
                                   int64_t candidate_age_s,
                                   int64_t min_version_age_s,
                                   const char* already_updated_to)
{
    if (candidate_tag.empty() || !VersionGreater(candidate_tag, current_tag))
        return UpdateDecision::UpToDate;
    if (!auto_update)        return UpdateDecision::AutoUpdateOff;
    if (!has_platform_asset) return UpdateDecision::NoAsset;
    if (min_version_age_s > 0 && candidate_age_s >= 0 && candidate_age_s < min_version_age_s)
        return UpdateDecision::Baking;
    if (already_updated_to != nullptr && candidate_tag == already_updated_to)
        return UpdateDecision::StampMismatch;
    return UpdateDecision::Adopt;
}

#endif  // MATADOR_MINER_UPDATE_GATE_H
