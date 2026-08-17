// Regression test for the pool-socket options (clean-stack/miner/pool_socket_opts.h).
// A dropped TCP_NODELAY silently re-enables Nagle -> tiny share submits can stall ~40ms
// (stale-share risk); a dropped SO_KEEPALIVE re-opens the half-open-socket hang. This
// pins both by applying the options to a real socket and reading them back. It does NOT
// measure latency -- Nagle only fires with unacked data outstanding, so a loopback
// request/response shows no delay either way; the value here is guarding the config.
// Standalone: no btx, no pool, no connect (setsockopt/getsockopt need only an fd).
//   cmake --build <build> --target pool_socket_opts_test && ./<build>/pool_socket_opts_test
#include "../miner/pool_socket_opts.h"

#include <cstdio>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>

static int g_fail = 0;

static void expect_enabled(int fd, int level, int optname, const char* label)
{
    int val = 0;
    socklen_t len = sizeof(val);
    if (::getsockopt(fd, level, optname, &val, &len) == 0 && val != 0)
        std::printf("  ok   %-32s = %d (enabled)\n", label, val);
    else { std::printf("  FAIL %-32s = %d (want enabled)\n", label, val); ++g_fail; }
}

int main()
{
    std::printf("[pool_socket_opts_test]\n");
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { std::printf("  FAIL could not create socket\n"); return 1; }

    ApplyPoolSocketOpts(fd);

    // The load-bearing latency guard: Nagle MUST be off on the pool socket.
    expect_enabled(fd, IPPROTO_TCP, TCP_NODELAY, "TCP_NODELAY");
    // The liveness guard: keepalive MUST be on so a dead peer is torn down.
    expect_enabled(fd, SOL_SOCKET, SO_KEEPALIVE, "SO_KEEPALIVE");

    ::close(fd);
    if (g_fail == 0) std::printf("ALL PASS\n");
    else std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
