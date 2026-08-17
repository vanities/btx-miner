// Minimal NodeClock::now() so matador_core stays btx-free (digest_probe links without the btx
// node archives). btx's full mockable version is in util/time.cpp; the standalone core never
// mocks time, so this is just the system-clock path. The miner links btx too (--allow-multiple-
// definition); matador_core's copy wins by group order, which is fine in production. (#13)
#include <util/time.h>

#include <chrono>
#include <cassert>

NodeClock::time_point NodeClock::now() noexcept
{
    using namespace std::chrono_literals;
    const auto ret{std::chrono::system_clock::now().time_since_epoch()};
    assert(ret > 0s);
    return time_point{ret};
}
