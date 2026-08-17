// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.
#ifndef MATADOR_MINER_ENDPOINT_PARSE_H
#define MATADOR_MINER_ENDPOINT_PARSE_H

#include <algorithm>
#include <cctype>
#include <string>

// host:port parsing for the stratum pool and the SOCKS5 proxy. A wrong parse here means the miner
// connects to the wrong place or not at all, so both are pure (std::string/int only) and unit-tested
// (clean-stack/harness/endpoint_parse_test.cpp). NOTE: both use rfind(':'), so a bare IPv6 literal
// (multiple colons) is NOT supported -- use [brackets] or a hostname.

// Parse a pool endpoint: "host:port" or a scheme-prefixed form. The bare alias "minebtx" expands to
// the public pool. Recognized schemes: stratum+tcp:// / stratum:// (plaintext, stripped) and
// ssl:// / tls:// / stratum+ssl:// / stratum+tls:// (TLS -- sets use_tls when non-null; ninjaraider's
// flightsheet uses "ssl://ninjaraider.com:44921"). An unrecognized scheme is stripped like a
// plaintext one (matches the old behavior) rather than rejected. Returns false if there is no
// ':port' or the port is invalid.
inline bool ParsePoolEndpoint(const std::string& raw, std::string& host, int& port, bool* use_tls = nullptr)
{
    std::string s = raw;
    if (s == "minebtx") s = "stratum.minebtx.com:3333";
    bool tls = false;
    const auto scheme = s.find("://");
    if (scheme != std::string::npos) {
        std::string sch = s.substr(0, scheme);
        std::transform(sch.begin(), sch.end(), sch.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        tls = (sch == "ssl" || sch == "tls" || sch == "stratum+ssl" || sch == "stratum+tls");
        s = s.substr(scheme + 3);
    }
    if (use_tls != nullptr) *use_tls = tls;
    const auto colon = s.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= s.size()) return false;
    host = s.substr(0, colon);
    try {
        port = std::stoi(s.substr(colon + 1));
    } catch (...) {
        return false;
    }
    return port > 0 && port <= 65535 && !host.empty();
}

// "host:port" (optionally scheme-prefixed, e.g. socks5://host:port or ALL_PROXY style) -> (host,
// port). Tolerates a trailing slash. Returns false if malformed. Shared by --socks5, the
// SOCKS5/ALL_PROXY env, and config. (Unlike ParsePoolEndpoint: no "minebtx" alias, strips a
// trailing '/'.)
inline bool SplitHostPort(std::string s, std::string& host, int& port)
{
    const auto scheme = s.find("://");
    if (scheme != std::string::npos) s = s.substr(scheme + 3);
    while (!s.empty() && s.back() == '/') s.pop_back();
    const auto pos = s.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= s.size()) return false;
    host = s.substr(0, pos);
    try { port = std::stoi(s.substr(pos + 1)); } catch (...) { return false; }
    return port > 0 && port < 65536;
}

#endif  // MATADOR_MINER_ENDPOINT_PARSE_H
