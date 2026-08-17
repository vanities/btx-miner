// Unit test for DecideUpdate() -- the auto-update adopt gate (clean-stack/miner/update_gate.h).
// This gate shipped the v0.6.10 -> v0.6.9 lexical-downgrade bug; the cases below lock that shut and
// cover every guard (version / auto_update / asset / bake-time / re-exec loop) plus their ordering.
// Standalone: no btx, no core, no network.
//   cmake --build <build> --target update_gate_test && ./<build>/update_gate_test   (exit 0 = pass)
#include "../miner/update_gate.h"

#include <cstdio>

static int g_fail = 0;

static const char* name(UpdateDecision d)
{
    switch (d) {
        case UpdateDecision::UpToDate:      return "UpToDate";
        case UpdateDecision::AutoUpdateOff: return "AutoUpdateOff";
        case UpdateDecision::NoAsset:       return "NoAsset";
        case UpdateDecision::Baking:        return "Baking";
        case UpdateDecision::StampMismatch: return "StampMismatch";
        case UpdateDecision::Adopt:         return "Adopt";
    }
    return "?";
}

static void check_eq(UpdateDecision got, UpdateDecision want, const char* what)
{
    const bool ok = (got == want);
    if (ok) std::printf("  ok   %s -> %s\n", what, name(got));
    else  { std::printf("  FAIL %s -> got %s, want %s\n", what, name(got), name(want)); ++g_fail; }
}
#define CHECK_EQ(call, want) check_eq((call), (want), #call)

int main()
{
    std::printf("[update_gate_test]\n");
    const char* NONE = nullptr;   // MATADOR_UPDATED_TO unset

    // ---- version gate: newer adopts, equal/older does NOT (the whole bug surface) ----
    CHECK_EQ(DecideUpdate("v0.7.0", "v0.7.0", true, true, -1, 0, NONE), UpdateDecision::UpToDate);   // equal
    CHECK_EQ(DecideUpdate("v0.6.9", "v0.7.0", true, true, -1, 0, NONE), UpdateDecision::UpToDate);   // older
    CHECK_EQ(DecideUpdate("",       "v0.7.0", true, true, -1, 0, NONE), UpdateDecision::UpToDate);   // empty
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true, -1, 0, NONE), UpdateDecision::Adopt);      // newer

    // The exact regression: 0.6.10 IS newer than 0.6.9 (and must never downgrade the other way).
    CHECK_EQ(DecideUpdate("v0.6.10", "v0.6.9",  true, true, -1, 0, NONE), UpdateDecision::Adopt);
    CHECK_EQ(DecideUpdate("v0.6.9",  "v0.6.10", true, true, -1, 0, NONE), UpdateDecision::UpToDate);
    // ...and 10.0.0 > 9.9.9 (the "esp if we get to 10.0.0" case).
    CHECK_EQ(DecideUpdate("v10.0.0", "v9.9.9",  true, true, -1, 0, NONE), UpdateDecision::Adopt);

    // ---- auto_update off: newer exists but we only notify ----
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", false, true, -1, 0, NONE), UpdateDecision::AutoUpdateOff);

    // ---- no platform asset: newer + auto on, but nothing to download ----
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, false, -1, 0, NONE), UpdateDecision::NoAsset);

    // ---- bake-time ----
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true,  100, 3600, NONE), UpdateDecision::Baking); // too young
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true, 3600, 3600, NONE), UpdateDecision::Adopt);  // age==min adopts
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true, 7200, 3600, NONE), UpdateDecision::Adopt);  // older than min
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true,   -1, 3600, NONE), UpdateDecision::Adopt);  // age unknown
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true,  100,    0, NONE), UpdateDecision::Adopt);  // bake off

    // ---- re-exec loop guard ----
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true, -1, 0, "v0.8.0"), UpdateDecision::StampMismatch); // same tag
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true, -1, 0, "v0.7.5"), UpdateDecision::Adopt);         // other tag

    // ---- guard ORDER: earlier guards win over later ones ----
    // auto-off beats missing-asset + baking + stamp-match
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", false, false, 0, 3600, "v0.8.0"), UpdateDecision::AutoUpdateOff);
    // baking beats stamp-match
    CHECK_EQ(DecideUpdate("v0.8.0", "v0.7.0", true, true, 0, 3600, "v0.8.0"), UpdateDecision::Baking);

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
