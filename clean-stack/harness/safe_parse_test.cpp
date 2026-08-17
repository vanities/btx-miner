// Unit test for the safe numeric parse helpers (clean-stack/miner/config_parse.h).
// Every env/CLI knob used to go straight through std::stoi/stoull/stod, which THROW on a
// typo ("--rpcport 1o0", MAXTRIES=1e9) -- an uncaught throw during arg parsing is
// std::terminate AT STARTUP, i.e. a systemd crash-loop on an unattended rig. Worse,
// stoi("1o0") without a full-consume check silently parses as 1 and half-configures the
// miner. This pins the replacement helpers: parse the WHOLE string or report failure,
// never throw. Standalone: MATADOR_CONFIG_PARSE_HELPERS_ONLY compiles just the pure
// helper block of config_parse.h (no Config/UniValue/mlog deps).
#define MATADOR_CONFIG_PARSE_HELPERS_ONLY
#include "../miner/config_parse.h"

#include <cstdint>
#include <cstdio>
#include <string>

static int g_fail = 0;

static void ok(bool cond, const char* label)
{
    if (cond) std::printf("  ok   %s\n", label);
    else { std::printf("  FAIL %s\n", label); ++g_fail; }
}

static void int_accepts(const char* s, int want, const char* label)
{
    int out = -777;
    const bool r = SafeParseInt(s, out);
    if (r && out == want) std::printf("  ok   %-40s -> %d\n", label, out);
    else { std::printf("  FAIL %-40s -> ok=%d out=%d (want %d)\n", label, r ? 1 : 0, out, want); ++g_fail; }
}

static void int_rejects(const char* s, const char* label)
{
    int out = -777;
    const bool r = SafeParseInt(s, out);
    if (!r) std::printf("  ok   %-40s -> rejected\n", label);
    else { std::printf("  FAIL %-40s -> accepted out=%d (want reject)\n", label, out); ++g_fail; }
}

static void u64_accepts(const char* s, uint64_t want, const char* label)
{
    uint64_t out = 777;
    const bool r = SafeParseUint64(s, out);
    if (r && out == want) std::printf("  ok   %-40s -> %llu\n", label, static_cast<unsigned long long>(out));
    else { std::printf("  FAIL %-40s -> ok=%d out=%llu (want %llu)\n", label, r ? 1 : 0,
                       static_cast<unsigned long long>(out), static_cast<unsigned long long>(want)); ++g_fail; }
}

static void u64_rejects(const char* s, const char* label)
{
    uint64_t out = 777;
    const bool r = SafeParseUint64(s, out);
    if (!r) std::printf("  ok   %-40s -> rejected\n", label);
    else { std::printf("  FAIL %-40s -> accepted out=%llu (want reject)\n", label,
                       static_cast<unsigned long long>(out)); ++g_fail; }
}

static void dbl_accepts(const char* s, double want, const char* label)
{
    double out = -777.0;
    const bool r = SafeParseDouble(s, out);
    if (r && out == want) std::printf("  ok   %-40s -> %g\n", label, out);
    else { std::printf("  FAIL %-40s -> ok=%d out=%g (want %g)\n", label, r ? 1 : 0, out, want); ++g_fail; }
}

static void dbl_rejects(const char* s, const char* label)
{
    double out = -777.0;
    const bool r = SafeParseDouble(s, out);
    if (!r) std::printf("  ok   %-40s -> rejected\n", label);
    else { std::printf("  FAIL %-40s -> accepted out=%g (want reject)\n", label, out); ++g_fail; }
}

int main()
{
    std::printf("[safe_parse_test]\n");

    // ---- SafeParseInt: happy paths ----
    int_accepts("0", 0, "int \"0\"");
    int_accepts("19334", 19334, "int \"19334\" (rpc port)");
    int_accepts("-265", -265, "int \"-265\" (clk offset style)");
    int_accepts("  42", 42, "int leading whitespace");
    int_accepts("42  ", 42, "int trailing whitespace");
    int_accepts("+7", 7, "int explicit +");

    // ---- SafeParseInt: the crash/typo class ----
    int_rejects("", "int empty (\"--rpcport\" as last arg)");
    int_rejects("abc", "int \"abc\"");
    int_rejects("1o0", "int \"1o0\" (stoi would return 1!)");
    int_rejects("42x", "int trailing garbage \"42x\"");
    int_rejects("4 2", "int embedded space \"4 2\"");
    int_rejects("1e3", "int \"1e3\" (no float notation)");
    int_rejects("12.5", "int \"12.5\"");
    int_rejects("0x10", "int \"0x10\" (hex not a port)");
    int_rejects("99999999999999999999", "int overflow");
    int_rejects("   ", "int whitespace only");

    // ---- SafeParseUint64 ----
    u64_accepts("0", 0, "u64 \"0\"");
    u64_accepts("1000000", 1000000ULL, "u64 \"1000000\" (maxtries)");
    u64_accepts("18446744073709551615", 18446744073709551615ULL, "u64 max");
    u64_rejects("", "u64 empty");
    u64_rejects("-5", "u64 \"-5\" (stoull would WRAP to 2^64-5)");
    u64_rejects("  -5", "u64 \" -5\" (wrap via leading space)");
    u64_rejects("1e9", "u64 \"1e9\" (MAXTRIES=1e9 typo)");
    u64_rejects("abc", "u64 \"abc\"");
    u64_rejects("99999999999999999999999", "u64 overflow");

    // ---- SafeParseDouble ----
    dbl_accepts("0", 0.0, "dbl \"0\"");
    dbl_accepts("550.5", 550.5, "dbl \"550.5\" (warn power watts)");
    dbl_accepts("1e3", 1000.0, "dbl \"1e3\"");
    dbl_accepts("-1.5", -1.5, "dbl \"-1.5\"");
    dbl_rejects("", "dbl empty");
    dbl_rejects("abc", "dbl \"abc\"");
    dbl_rejects("1.5w", "dbl trailing garbage \"1.5w\"");
    dbl_rejects("nan", "dbl \"nan\" (non-finite)");
    dbl_rejects("inf", "dbl \"inf\" (non-finite)");

    // out must be untouched on rejection (callers keep their defaults)
    {
        int out = 1234;
        (void)SafeParseInt("garbage", out);
        ok(out == 1234, "int out untouched on reject");
        uint64_t u = 9876;
        (void)SafeParseUint64("-1", u);
        ok(u == 9876, "u64 out untouched on reject");
        double d = 3.5;
        (void)SafeParseDouble("x", d);
        ok(d == 3.5, "dbl out untouched on reject");
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
