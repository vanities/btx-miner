// Unit test for the host:port parsers (clean-stack/miner/endpoint_parse.h). A wrong parse points the
// miner at the wrong pool/proxy or refuses to connect, so this pins the contract AND the quirks
// (lenient std::stoi, trailing-slash handling, the minebtx alias). Standalone: no btx, no sockets.
//   cmake --build <build> --target endpoint_parse_test && ./<build>/endpoint_parse_test   (exit 0 = pass)
#include "../miner/endpoint_parse.h"

#include <cstdio>
#include <string>

static int g_fail = 0;

static void ok_case(bool got, const std::string& host, int port,
                    const char* label, const char* want_host, int want_port)
{
    const bool pass = got && host == want_host && port == want_port;
    if (pass) std::printf("  ok   %-40s -> %s:%d\n", label, host.c_str(), port);
    else { std::printf("  FAIL %-40s -> ok=%d host=%s port=%d (want %s:%d)\n",
                       label, got, host.c_str(), port, want_host, want_port); ++g_fail; }
}
static void bad_case(bool got, const std::string& host, int port, const char* label)
{
    if (!got) std::printf("  ok   %-40s -> rejected\n", label);
    else { std::printf("  FAIL %-40s -> accepted host=%s port=%d (want reject)\n",
                       label, host.c_str(), port); ++g_fail; }
}

// NB: every macro below MUST sequence the parse call before reading h/p/t.
// The original form -- ok_case(ParsePoolEndpoint(in, h, p), h, p, ...) -- relies on function
// ARGUMENT EVALUATION ORDER, which C++ leaves unspecified. GCC evaluates right-to-left, so `p`
// was copied by value (still 0) BEFORE the parse ran, while `host` bound by reference and read
// the updated value. Result: every case reported ok=1, correct host, port=0, and the suite had
// never actually exercised the parser on GCC. The parser itself was always correct.
#define POOL_OK(in, eh, ep) do { std::string h; int p = 0; \
    const bool r_ = ParsePoolEndpoint(in, h, p); ok_case(r_, h, p, "pool " in, eh, ep); } while (0)
#define POOL_BAD(in)        do { std::string h; int p = 0; \
    const bool r_ = ParsePoolEndpoint(in, h, p); bad_case(r_, h, p, "pool " in); } while (0)

static void tls_case(bool got, const std::string& host, int port, bool tls,
                     const char* label, const char* want_host, int want_port, bool want_tls)
{
    const bool pass = got && host == want_host && port == want_port && tls == want_tls;
    if (pass) std::printf("  ok   %-40s -> %s:%d tls=%d\n", label, host.c_str(), port, tls);
    else { std::printf("  FAIL %-40s -> ok=%d host=%s port=%d tls=%d (want %s:%d tls=%d)\n",
                       label, got, host.c_str(), port, tls, want_host, want_port, want_tls); ++g_fail; }
}
#define POOL_TLS(in, eh, ep, etls) do { std::string h; int p = 0; bool t = false; \
    const bool r_ = ParsePoolEndpoint(in, h, p, &t); tls_case(r_, h, p, t, "pool-tls " in, eh, ep, etls); } while (0)
#define SHP_OK(in, eh, ep)  do { std::string h; int p = 0; \
    const bool r_ = SplitHostPort(in, h, p); ok_case(r_, h, p, "split " in, eh, ep); } while (0)
#define SHP_BAD(in)         do { std::string h; int p = 0; \
    const bool r_ = SplitHostPort(in, h, p); bad_case(r_, h, p, "split " in); } while (0)

int main()
{
    std::printf("[endpoint_parse_test]\n");

    // ---- ParsePoolEndpoint: the happy paths ----
    POOL_OK("minebtx", "stratum.minebtx.com", 3333);          // bare alias expands
    POOL_OK("host:3333", "host", 3333);
    POOL_OK("stratum.minebtx.com:3333", "stratum.minebtx.com", 3333);
    POOL_OK("stratum+tcp://host:3333", "host", 3333);         // scheme stripped
    POOL_OK("stratum://pool.example:9999", "pool.example", 9999);
    POOL_OK("host:1", "host", 1);                             // low port boundary
    POOL_OK("host:65535", "host", 65535);                    // high port boundary

    // ---- ParsePoolEndpoint: rejects ----
    POOL_BAD("");          // empty
    POOL_BAD("host");      // no :port
    POOL_BAD(":3333");     // empty host
    POOL_BAD("host:");     // no port digits
    POOL_BAD("host:abc");  // non-numeric port
    POOL_BAD("host:0");    // port 0
    POOL_BAD("host:-1");   // negative
    POOL_BAD("host:65536");// above max
    POOL_BAD("host:99999");// above max

    // ---- ParsePoolEndpoint: QUIRKS pinned (current behavior, not necessarily ideal) ----
    POOL_OK("host:3333abc", "host", 3333);   // std::stoi stops at first non-digit
    POOL_OK("host:3333/", "host", 3333);     // no slash-strip here, but stoi ignores the trailing '/'

    // ---- ParsePoolEndpoint: TLS scheme detection (use_tls out-param) ----
    POOL_TLS("ssl://ninjaraider.com:44921", "ninjaraider.com", 44921, true);   // real flightsheet example
    POOL_TLS("tls://host:3333", "host", 3333, true);
    POOL_TLS("stratum+ssl://host:3333", "host", 3333, true);
    POOL_TLS("stratum+tls://host:3333", "host", 3333, true);
    POOL_TLS("SSL://host:3333", "host", 3333, true);          // scheme match is case-insensitive
    POOL_TLS("host:3333", "host", 3333, false);                // bare host:port -> no TLS
    POOL_TLS("stratum+tcp://host:3333", "host", 3333, false);
    POOL_TLS("stratum://host:3333", "host", 3333, false);
    POOL_TLS("minebtx", "stratum.minebtx.com", 3333, false);   // alias expands, still plaintext

    // ---- SplitHostPort (proxy): happy paths ----
    SHP_OK("host:1080", "host", 1080);
    SHP_OK("socks5://host:1080", "host", 1080);   // scheme stripped
    SHP_OK("host:1080/", "host", 1080);           // trailing slash stripped
    SHP_OK("host:1080///", "host", 1080);         // all trailing slashes stripped
    SHP_OK("127.0.0.1:1080", "127.0.0.1", 1080);
    SHP_OK("host:1", "host", 1);
    SHP_OK("host:65535", "host", 65535);

    // ---- SplitHostPort: rejects ----
    SHP_BAD("");
    SHP_BAD("host");
    SHP_BAD(":1080");
    SHP_BAD("host:");
    SHP_BAD("host:abc");
    SHP_BAD("host:0");
    SHP_BAD("host:65536");

    // ---- divergence from the pool parser: SplitHostPort has NO "minebtx" alias ----
    SHP_BAD("minebtx");

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
