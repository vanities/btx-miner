// Unit test for the pure difficulty/rate/duration formatters (clean-stack/miner/miner_format.h).
// These drive every human-facing rate/diff/uptime field in the logs and the status API, so this pins
// the unit rollover (k/M/B/T), the precision, and the difficulty math. Links matador_core only for
// arith_uint256 (no btx node, no sockets).
//   cmake --build <build> --target miner_format_test && ./<build>/miner_format_test   (exit 0 = pass)
// run-tests: -Icore/vendor core/vendor/arith_uint256.cpp core/vendor/uint256.cpp core/vendor/crypto/hex_base.cpp
#include "../miner/miner_format.h"

#include <cmath>
#include <cstdio>
#include <string>

static int g_fail = 0;

static void eq_str(const std::string& got, const char* want, const char* label)
{
    if (got == want) std::printf("  ok   %-34s -> \"%s\"\n", label, got.c_str());
    else { std::printf("  FAIL %-34s -> \"%s\" (want \"%s\")\n", label, got.c_str(), want); ++g_fail; }
}
static void approx(double got, double want, const char* label)
{
    const double tol = 1e-9 * (std::fabs(want) > 1.0 ? std::fabs(want) : 1.0);
    if (std::fabs(got - want) <= tol) std::printf("  ok   %-34s -> %.6g\n", label, got);
    else { std::printf("  FAIL %-34s -> %.10g (want %.10g)\n", label, got, want); ++g_fail; }
}

int main()
{
    std::printf("[miner_format_test]\n");

    // ---- FmtDiff: precision + k/M/B/T rollover; sub-1 keeps 5 decimals ----
    eq_str(FmtDiff(1.0),      "1.00",     "FmtDiff(1.0)");
    eq_str(FmtDiff(1500.0),   "1.50k",    "FmtDiff(1500)");
    eq_str(FmtDiff(2.5e6),    "2.50M",    "FmtDiff(2.5e6)");
    eq_str(FmtDiff(1.5e9),    "1.50B",    "FmtDiff(1.5e9)");
    eq_str(FmtDiff(3.0e12),   "3.00T",    "FmtDiff(3e12)");
    eq_str(FmtDiff(0.5),      "0.50000",  "FmtDiff(0.5)");

    // ---- FmtRate: flat integer below 1000, dynamic units above; negative clamps to 0 ----
    eq_str(FmtRate(873.0),    "873",      "FmtRate(873)");
    eq_str(FmtRate(36320.0),  "36.32k",   "FmtRate(36320)");
    eq_str(FmtRate(1.2e6),    "1.20M",    "FmtRate(1.2e6)");
    eq_str(FmtRate(1.05e9),   "1.05B",    "FmtRate(1.05e9)");
    eq_str(FmtRate(2.3e12),   "2.30T",    "FmtRate(2.3e12)");
    eq_str(FmtRate(-5.0),     "0",        "FmtRate(-5)");

    // ---- FmtDuration: dynamic units, zero-padded second field ----
    eq_str(FmtDuration(45),       "45s",     "FmtDuration(45s)");
    eq_str(FmtDuration(750),      "12m30s",  "FmtDuration(12m30s)");
    eq_str(FmtDuration(11100),    "3h05m",   "FmtDuration(3h05m)");
    eq_str(FmtDuration(187200),   "2d04h",   "FmtDuration(2d04h)");
    eq_str(FmtDuration(0),        "0s",      "FmtDuration(0)");

    // ---- DifficultyFromCompactBits: relative to the 0x1d00ffff diff-1 anchor ----
    approx(DifficultyFromCompactBits(0x1d00ffffu), 1.0,        "diff(0x1d00ffff)=1");
    approx(DifficultyFromCompactBits(0x1c00ffffu), 256.0,      "diff(0x1c00ffff)=256");
    approx(DifficultyFromCompactBits(0x1e00ffffu), 1.0/256.0,  "diff(0x1e00ffff)=1/256");
    approx(DifficultyFromCompactBits(0x1d000000u), 0.0,        "diff(mantissa=0)=0");

    // ---- AttemptsPerShare = 2^256 / target ----
    approx(AttemptsPerShare(arith_uint256(1) << 255), 2.0, "attempts(2^255)=2");
    approx(AttemptsPerShare(arith_uint256(1) << 254), 4.0, "attempts(2^254)=4");
    approx(AttemptsPerShare(arith_uint256(0)),        0.0, "attempts(0)=0");

    // ---- DifficultyFromTarget: diff-1 target (0xffff<<208) -> 1.0; half target -> 2.0 ----
    approx(DifficultyFromTarget(arith_uint256(0xffff) << 208), 1.0, "diffTarget(diff1)=1");
    approx(DifficultyFromTarget(arith_uint256(0xffff) << 207), 2.0, "diffTarget(diff1/2)=2");

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
