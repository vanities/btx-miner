#pragma once
// Pool-socket options, extracted as a free function so they are unit-testable
// (harness/pool_socket_opts_test.cpp) instead of being buried inline in Connect().
//
//   - SO_KEEPALIVE + tuned probes: tear down a silently half-open peer (pool hangs
//     or a NAT drops the flow with no RST/FIN) so the reader thread never blocks
//     forever on a dead socket.
//   - TCP_NODELAY: disable Nagle so tiny, latency-critical share submits are not
//     held ~40ms waiting to coalesce with more data or a pending ACK. Submits are
//     one-shot request/response, so there is no throughput cost to disabling it.
//
// Best-effort: setsockopt failures are non-fatal (the socket still works). Kept
// dependency-free (only POSIX socket headers) so the test links without btx/UniValue.
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

inline void ApplyPoolSocketOpts(int fd)
{
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
#ifdef TCP_NODELAY
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
#ifdef TCP_KEEPIDLE
    int idle = 30, intvl = 10, cnt = 3;   // dead after ~30 + 3*10 = 60s
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE,  &idle,  sizeof(idle));
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT,   &cnt,   sizeof(cnt));
#endif
}
