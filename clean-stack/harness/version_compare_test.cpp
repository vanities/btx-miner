// Unit test for VersionGreater() -- the auto-update version gate (clean-stack/miner/version_compare.h).
// Standalone: no btx, no core, no GPU. Build + run:
//   cmake --build <build> --target version_compare_test && ./<build>/version_compare_test
// Returns 0 if all pass, 1 otherwise.
#include "../miner/version_compare.h"

#include <cstdio>
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
    std::printf("[version_compare_test]\n");

    // The bug that started this: 0.6.10 must be NEWER than 0.6.9 (lexical sorted it below).
    CHECK(VersionGreater("v0.6.10", "v0.6.9"));
    CHECK(!VersionGreater("v0.6.9", "v0.6.10"));

    // Two-digit major: 10.0.0 > 9.9.9 (the "when we get to 10.0.0" case).
    CHECK(VersionGreater("v10.0.0", "v9.9.9"));
    CHECK(!VersionGreater("v9.9.9", "v10.0.0"));

    // Normal forward progression.
    CHECK(VersionGreater("v0.7.0", "v0.6.9"));
    CHECK(!VersionGreater("v0.6.9", "v0.7.0"));
    CHECK(VersionGreater("v0.7.1", "v0.7.0"));
    CHECK(VersionGreater("v1.0.0", "v0.99.99"));

    // Equal -> NOT newer (so a binary never re-adopts its own published version = no loop).
    CHECK(!VersionGreater("v0.7.0", "v0.7.0"));

    // 'v' prefix optional on either side.
    CHECK(!VersionGreater("0.7.0", "v0.7.0"));
    CHECK(VersionGreater("0.7.1", "v0.7.0"));

    // -suffix (prerelease tag like -obs/-dbg) ignored: parsed to the numeric part.
    CHECK(VersionGreater("v0.7.0", "v0.6.9-obs"));
    CHECK(!VersionGreater("v0.6.9-dbg", "v0.7.0"));
    CHECK(!VersionGreater("v0.6.9", "v0.6.9-obs"));   // equal numeric -> not newer

    // Per-component integer boundaries (the whole point).
    CHECK(VersionGreater("v0.6.100", "v0.6.99"));
    CHECK(VersionGreater("v0.10.0", "v0.9.99"));
    CHECK(!VersionGreater("v0.9.99", "v0.10.0"));

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
