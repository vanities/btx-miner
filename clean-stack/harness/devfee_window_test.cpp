// Unit test for InDevFeeWindow() -- the time-based dev-fee gate (clean-stack/miner/devfee_window.h).
// A bug here changes the actual fee rate every miner pays. Standalone: no btx, no core, no sockets.
//   cmake --build <build> --target devfee_window_test && ./<build>/devfee_window_test   (exit 0 = pass)
#include "../miner/devfee_window.h"

#include <cstdio>
#include <initializer_list>
#include <string>

static int g_fail = 0;
static void check(bool cond, const char* expr)
{
    std::printf(cond ? "  ok   %s\n" : "  FAIL %s\n", expr);
    if (!cond) ++g_fail;
}
#define CHECK(x) check((x), #x)

int main()
{
    std::printf("[devfee_window_test]\n");
    const double P = 3600.0;   // kPoolDevPeriodSec (1 hour); devfee=1 -> 36s window

    // Boundaries of the window at the start of a period.
    CHECK(InDevFeeWindow(0.0,   P, 1));    // pos=0  < 36
    CHECK(InDevFeeWindow(35.9,  P, 1));    // pos=35.9 < 36
    CHECK(!InDevFeeWindow(36.0, P, 1));    // pos=36 NOT < 36
    CHECK(!InDevFeeWindow(100.0, P, 1));   // pos=100 outside

    // The window repeats every period (mod), not just the first hour.
    CHECK(InDevFeeWindow(P + 0.0,   P, 1));    // pos=0 next period
    CHECK(InDevFeeWindow(P + 35.9,  P, 1));
    CHECK(!InDevFeeWindow(P + 36.0, P, 1));
    CHECK(InDevFeeWindow(100 * P + 10.0, P, 1));   // still works far out

    // devfee off -> never in window (no fee charged).
    CHECK(!InDevFeeWindow(0.0, P, 0));
    CHECK(!InDevFeeWindow(10.0, P, 0));

    // Bigger fee widens the window proportionally (5% -> 180s).
    CHECK(InDevFeeWindow(179.0, P, 5));
    CHECK(!InDevFeeWindow(181.0, P, 5));

    // Fee-RATE property: fraction of wall-clock in the window == devfee% (the whole point).
    // Sample one second at a time across a full period and count.
    for (int pct : {1, 5, 10}) {
        int in = 0;
        for (int s = 0; s < 3600; ++s) if (InDevFeeWindow(static_cast<double>(s), P, pct)) ++in;
        const int expect = 36 * pct;   // pct% of 3600
        // exact here because window=36*pct is an integer number of 1s samples
        check(in == expect, (std::string("fee-rate ") + std::to_string(pct) + "% == "
                             + std::to_string(in) + "/3600").c_str());
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
