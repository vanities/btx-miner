// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.
#ifndef MATADOR_MINER_DEVFEE_WINDOW_H
#define MATADOR_MINER_DEVFEE_WINDOW_H

#include <cstdint>

// Time-based dev-fee gate (like solo / Claymore / ethminer): for the first `devfee_pct`% of each
// `period_sec` window, found shares are submitted on the dev session (credit the dev); the rest
// credit the user. Returns true iff `uptime_sec` falls inside the current period's dev window.
//
// The fraction of wall-clock time this returns true is exactly devfee_pct/100, so this IS the
// fee rate -- a bug here over- or under-charges every miner running the wrapper. Pure + unit-tested
// (clean-stack/harness/devfee_window_test.cpp). devfee_pct<=0 (fee off) returns false always.
//
// CALLERS: pass a WALL-CLOCK phase (epoch seconds), never process uptime. Uptime restarts at 0,
// so every restart lands inside the leading dev slice -- which billed 4 of 5 solo blocks to the
// dev address on a restart-heavy night (2026-08-12) under a 1% fee. The argument keeps its old
// name only because the arithmetic is identical for any monotonic seconds source.
inline bool InDevFeeWindow(double uptime_sec, double period_sec, int devfee_pct)
{
    if (devfee_pct <= 0 || period_sec <= 0.0) return false;
    const double window = period_sec * (static_cast<double>(devfee_pct) / 100.0);
    // Position within the current period: uptime_sec mod period_sec (truncated, matches the miner).
    const double pos = uptime_sec - period_sec *
        static_cast<double>(static_cast<uint64_t>(uptime_sec / period_sec));
    return pos < window;
}

#endif  // MATADOR_MINER_DEVFEE_WINDOW_H
