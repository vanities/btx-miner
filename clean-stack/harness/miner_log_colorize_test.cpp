// Byte-for-byte pinning test for mlog::Colorize (clean-stack/miner/miner_log.h).
// Colorize was refactored for cost (computed OUTSIDE the global log mutex, plus a
// cheap "no '=' -> skip the ~25 field-regex passes" fast path). The refactor must be
// OUTPUT-PRESERVING: these goldens were captured from the PRE-refactor Colorize and
// pin every field pass, the pass-order interactions (pool-nonce/s before nonce/s,
// temp/rej%% thresholds), and the fast-path lines (no '=') staying identical.
// Regenerate (only if the palette itself intentionally changes): print
// mlog::Colorize(in) for each input with '\033' escaped as "\033". Links uint256
// only because miner_log.h's Short(uint256) helper needs the type.
//   ./run-tests.sh   (exit 0 = pass)
// run-tests: -Icore/vendor core/vendor/uint256.cpp core/vendor/crypto/hex_base.cpp
#include "../miner/miner_log.h"

#include <cstdio>
#include <string>

static int g_fail = 0;

struct Golden { const char* in; const char* want; };

// Captured from the pre-refactor Colorize -- do NOT hand-edit (see the header
// comment for how to regenerate on an intentional palette change).
static const Golden kGoldens[] = {
    {"[stats] 60s: nonce/s=30.5k pool-nonce/s=31.00k digest-c/s=44.6k scan=118.3MN/s acc=12 rejected=0 rej%=0.0 stale=3 net-diff=112.5M pool-diff=1.2k spacing=28.1k digest-batch/s=15.2 height=123456 clean=yes v0.8.21",
     "\033[1;95m[stats]\033[0m 60s: \033[1;93mnonce/s=30.5k\033[0m \033[1;92mpool-nonce/s=31.00k\033[0m digest-c/s=44.6k \033[36mscan=118.3MN/s\033[0m \033[32macc=12\033[0m \033[92mrejected=0\033[0m rej%=\033[92m0.0\033[0m \033[33mstale=3\033[0m \033[1;96mnet-diff=112.5M\033[0m \033[94mpool-diff=1.2k\033[0m \033[2mspacing=28.1k\033[0m \033[96mdigest-batch/s=15.2\033[0m \033[2mheight=123456\033[0m \033[32mclean=yes\033[0m \033[1mv0.8.21\033[0m"},
    {"[share] ACCEPTED id=42 (accepted=13 rejected=1)",
     "\033[1;92m[share]\033[0m \033[1;92mACCEPTED\033[0m id=42 (\033[32maccepted=13\033[0m \033[1;91mrejected=1\033[0m)"},
    {"[gpu] temp=83C clk=3105 mem=14001 pow=552W fan=60% util=100% nonce/W=55",
     "\033[1;94m[gpu]\033[0m temp=\033[1;91m83C\033[0m \033[36mclk=3105\033[0m \033[36mmem=14001\033[0m \033[33mpow=552W\033[0m \033[2mfan=60%\033[0m \033[35mutil=100%\033[0m \033[1mnonce/W=55\033[0m"},
    {"[gpu] temp=76C then temp=60C rej%=2.5",
     "\033[1;94m[gpu]\033[0m temp=\033[33m76C\033[0m then temp=\033[92m60C\033[0m rej%=\033[1;91m2.5\033[0m"},
    {"nonce/s=1.2M after a job switch",
     "\033[1;93mnonce/s=1.2M\033[0m after a job switch"},
    {"[solve] done height=999 clean=no stale=0",
     "\033[34m[solve]\033[0m done \033[2mheight=999\033[0m \033[2mclean=no\033[0m stale=0"},
    // fast-path lines: NO '=' anywhere -- the field-pass skip must be a no-op
    {"[pool] connected; the cape flutters",
     "\033[36m[pool]\033[0m connected; the cape flutters"},
    {"[watchdog] enabled; actions include reconnect/failover",
     "\033[35m[watchdog]\033[0m enabled; actions include reconnect/failover"},
    {"ERROR WARN [update] [init] [stats] [share] v0.8.21 ACCEPTED",
     "\033[1;91mERROR\033[0m \033[1;33mWARN\033[0m \033[1;93m[update]\033[0m \033[1;97m[init]\033[0m \033[1;95m[stats]\033[0m \033[1;92m[share]\033[0m \033[1mv0.8.21\033[0m \033[1;92mACCEPTED\033[0m"},
    {"plain prose line with no tags and no fields",
     "plain prose line with no tags and no fields"},
    {"",
     ""},
};

// Print with escapes visible so a diff is readable in CI output.
static std::string visible(const std::string& s)
{
    std::string out;
    for (unsigned char c : s) {
        if (c == '\033') out += "\\033";
        else out += static_cast<char>(c);
    }
    return out;
}

int main()
{
    std::printf("[miner_log_colorize_test]\n");
    int idx = 0;
    for (const Golden& g : kGoldens) {
        const std::string got = mlog::Colorize(g.in);
        if (got == g.want) {
            std::printf("  ok   golden[%d] byte-identical (%zu bytes)\n", idx, got.size());
        } else {
            std::printf("  FAIL golden[%d]\n    in  : %s\n    want: %s\n    got : %s\n",
                        idx, g.in, visible(g.want).c_str(), visible(got).c_str());
            ++g_fail;
        }
        ++idx;
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
