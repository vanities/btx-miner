// matador-miner: pure difficulty / rate / duration formatters + difficulty math.
// No miner state -- pure functions, unit-tested in harness/miner_format_test.cpp.
// Extracted verbatim from matador-miner.cpp; included into the single miner TU.
#pragma once

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#include <arith_uint256.h>

// Network difficulty from compact nBits (Bitcoin GetDifficulty: relative to the
// 0x1d00ffff diff-1 anchor, the convention block explorers display). Pure.
static double DifficultyFromCompactBits(uint32_t nbits)
{
    const uint32_t mant = nbits & 0x00ffffffu;
    if (mant == 0) return 0.0;
    int nShift = (nbits >> 24) & 0xff;
    double dDiff = static_cast<double>(0x0000ffff) / static_cast<double>(mant);
    while (nShift < 29) { dDiff *= 256.0; ++nShift; }
    while (nShift > 29) { dDiff /= 256.0; --nShift; }
    return dDiff;
}

// Expected digest attempts to land one share at this 256-bit target = 2^256/target.
// Anchor-independent and in the SAME units as the live nonce/s counter, so
// shares*this/elapsed is directly comparable to the measured nonce/s.
static double AttemptsPerShare(const arith_uint256& target)
{
    const double t = target.getdouble();
    return t > 0.0 ? std::ldexp(1.0, 256) / t : 0.0;
}

// Share difficulty from the full per-share target (diff-1 target = 0xffff<<208,
// the same convention as DifficultyFromCompactBits). Every BTX pool conveys the
// share target it grades submits against in mining.notify[6] (or the login
// dialect's shareTarget), so pool-diff is derived here, from the target we
// actually solve. Some pools ALSO emit mining.set_difficulty (minebtx does), but
// that number is in different units and does not gate shares -- see the
// set_difficulty handler in stratum_client.h. Never derive pool-diff from it.
static double DifficultyFromTarget(const arith_uint256& target)
{
    const double t = target.getdouble();
    return t > 0.0 ? 65535.0 * std::ldexp(1.0, 208) / t : 0.0;
}

// Difficulty formatter that stays readable across the huge net-diff (M/k) and the
// tiny sub-1 pool share-diff alike.
static std::string FmtDiff(double v)
{
    std::ostringstream os;
    os << std::fixed;
    if (v >= 1e12)     os << std::setprecision(2) << v / 1e12 << 'T';
    else if (v >= 1e9) os << std::setprecision(2) << v / 1e9 << 'B';
    else if (v >= 1e6) os << std::setprecision(2) << v / 1e6 << 'M';
    else if (v >= 1e3) os << std::setprecision(2) << v / 1e3 << 'k';
    else if (v >= 1.0) os << std::setprecision(2) << v;
    else               os << std::setprecision(5) << v;
    return os.str();
}

// Rate formatter: flat integer below 1000, dynamic k/M/B/T above
// (e.g. 873, 36.32k, 1.20M, 1.05B, 2.30T) -- so scan/nonce rates roll units cleanly.
static std::string FmtRate(double v)
{
    std::ostringstream os;
    if (v >= 1e12)     os << std::fixed << std::setprecision(2) << v / 1e12 << 'T';
    else if (v >= 1e9) os << std::fixed << std::setprecision(2) << v / 1e9 << 'B';
    else if (v >= 1e6) os << std::fixed << std::setprecision(2) << v / 1e6 << 'M';
    else if (v >= 1e3) os << std::fixed << std::setprecision(2) << v / 1e3 << 'k';
    else               os << static_cast<uint64_t>(v < 0.0 ? 0.0 : v);
    return os.str();
}

// Duration formatter: dynamic units so uptime stays readable as it grows
// (e.g. 45s, 12m30s, 3h05m, 2d04h) instead of an ever-growing seconds count.
static std::string FmtDuration(uint64_t s)
{
    std::ostringstream os;
    const uint64_t d = s / 86400, h = (s % 86400) / 3600, m = (s % 3600) / 60, sec = s % 60;
    if (d > 0)      os << d << 'd' << std::setw(2) << std::setfill('0') << h << 'h';
    else if (h > 0) os << h << 'h' << std::setw(2) << std::setfill('0') << m << 'm';
    else if (m > 0) os << m << 'm' << std::setw(2) << std::setfill('0') << sec << 's';
    else            os << sec << 's';
    return os.str();
}
