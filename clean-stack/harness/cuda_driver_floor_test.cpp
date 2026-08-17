// Unit test for the CUDA driver-floor version math (clean-stack/miner/cuda_driver_floor.h).
// Standalone: no btx, no core, no CUDA, no GPU. Build + run:
//   ./run-tests.sh          (or: c++ -std=c++20 harness/cuda_driver_floor_test.cpp -o /dev/null)
// Returns 0 if all pass, 1 otherwise.
#include "../miner/cuda_driver_floor.h"

#include <cstdio>

static int g_fail = 0;
static void check(bool cond, const char* expr)
{
    std::printf(cond ? "  ok   %s\n" : "  FAIL %s\n", expr);
    if (!cond) ++g_fail;
}
#define CHECK(x) check((x), #x)

// Decode `packed` and report whether a driver advertising it clears the given build floor.
// Mirrors CheckCudaDriverFloor()'s two-step: DecodeCudaVersion -> CudaDriverMeetsFloor.
static bool DriverPasses(int packed, int req_major, int req_minor)
{
    int major = 0, minor = 0;
    if (!DecodeCudaVersion(packed, major, minor)) return true;  // unknown -> conservative pass
    return CudaDriverMeetsFloor(major, minor, req_major, req_minor);
}

int main()
{
    std::printf("[cuda_driver_floor_test]\n");

    // ---- DecodeCudaVersion: CUDA packs 1000*major + 10*minor ----
    int major = 0, minor = 0;
    CHECK(DecodeCudaVersion(13030, major, minor) && major == 13 && minor == 3);  // 13.3
    CHECK(DecodeCudaVersion(13000, major, minor) && major == 13 && minor == 0);  // 13.0
    CHECK(DecodeCudaVersion(12040, major, minor) && major == 12 && minor == 4);  // 12.4, green-ctx floor
    CHECK(DecodeCudaVersion(12080, major, minor) && major == 12 && minor == 8);  // 12.8, legacy toolkit
    CHECK(DecodeCudaVersion(11080, major, minor) && major == 11 && minor == 8);  // 11.8

    // "No driver installed" arrives as 0 -- must decode as unusable, not as CUDA 0.0.
    CHECK(!DecodeCudaVersion(0, major, minor));
    CHECK(!DecodeCudaVersion(-1, major, minor));
    CHECK(!DecodeCudaVersion(999, major, minor));  // sub-1.0 garbage -> major 0

    // A failed decode must not clobber the caller's values (CheckCudaDriverFloor logs them).
    major = 7; minor = 7;
    CHECK(!DecodeCudaVersion(0, major, minor) && major == 7 && minor == 7);

    // ---- CudaDriverMeetsFloor: exact-equality and both sides of each boundary ----
    CHECK(CudaDriverMeetsFloor(13, 0, 13, 0));    // exactly at floor
    CHECK(CudaDriverMeetsFloor(13, 3, 13, 0));    // newer minor, same major
    CHECK(CudaDriverMeetsFloor(14, 0, 13, 0));    // newer major
    CHECK(!CudaDriverMeetsFloor(12, 9, 13, 0));   // older major, higher minor -> still too old
    CHECK(!CudaDriverMeetsFloor(12, 0, 13, 0));

    // Legacy build floor: CUDA 12.0, the MAJOR floor -- not 12.8, the toolkit minor it links.
    CHECK(CudaDriverMeetsFloor(12, 0, 12, 0));
    CHECK(CudaDriverMeetsFloor(12, 2, 12, 0));
    CHECK(CudaDriverMeetsFloor(12, 8, 12, 0));
    CHECK(CudaDriverMeetsFloor(13, 0, 12, 0));    // a CUDA-13 driver runs the legacy build
    CHECK(!CudaDriverMeetsFloor(11, 8, 12, 0));   // CUDA 11 driver cannot

    // Unknown driver version PASSES: this gate aborts the miner, so a false block (refusing to
    // start on a fine machine) is worse than a missed catch (backend probe still warns).
    CHECK(CudaDriverMeetsFloor(0, 0, 13, 0));
    CHECK(CudaDriverMeetsFloor(-1, 0, 13, 0));

    // ---- End-to-end: the real driver/build pairings this gate exists to judge ----
    // Main build (links CUDA 13.x, needs a driver supporting >= 13.0).
    CHECK(DriverPasses(13030, 13, 0));   // r595 -- pc's deployed driver
    CHECK(DriverPasses(13000, 13, 0));   // r580 -- exactly the documented minimum
    CHECK(!DriverPasses(12080, 13, 0));  // r570
    CHECK(!DriverPasses(12040, 13, 0));  // r550 -- has green-ctx symbols, still cannot init CUDA 13

    // The reported user: driver too old to export cuGreenCtxStreamCreate (< CUDA 12.4 / r550).
    // Their <=v0.8.32 binary died in ld.so before this gate could run; on v0.8.33+ it survives
    // load and this gate is what must explain the failure.
    CHECK(!DriverPasses(12000, 13, 0));  // r525
    CHECK(!DriverPasses(11080, 13, 0));  // r520

    // Legacy build (links CUDA 12.8, floor = CUDA 12.0 / r525 by minor-version compatibility).
    // The two 12.x rows below are not theory: with the gate bypassed, the -legacy binary
    // initialized CUDA and mined clean on exactly these drivers (2026-07-09, Vast).
    CHECK(DriverPasses(12020, 12, 0));   // r535.309.01, GTX 1080 Ti -- measured 1.31k nonce/s rej=0
    CHECK(DriverPasses(12040, 12, 0));   // r550.144.03, RTX 2080 Ti -- measured 3.68k nonce/s rej=0
    CHECK(DriverPasses(12000, 12, 0));   // r525, the documented CUDA 12.0 minimum
    CHECK(DriverPasses(12080, 12, 0));   // r570, the old (over-strict) floor still passes
    CHECK(DriverPasses(13030, 12, 0));   // a modern driver runs legacy fine
    CHECK(!DriverPasses(11080, 12, 0));  // CUDA 11.8 driver -- genuinely below the major floor

    // No driver at all (Apple, no libcuda, nvidia-smi absent) -> never block.
    CHECK(DriverPasses(0, 13, 0));
    CHECK(DriverPasses(0, 12, 8));

    if (g_fail == 0) {
        std::printf("[cuda_driver_floor_test] all passed\n");
        return 0;
    }
    std::printf("[cuda_driver_floor_test] %d FAILED\n", g_fail);
    return 1;
}
