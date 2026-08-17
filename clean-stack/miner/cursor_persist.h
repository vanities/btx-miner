// matador-miner: nonce-cursor persistence (resume scan past a reconnect/restart).
// Extracted verbatim from matador-miner.cpp; included into the single miner
// translation unit at the point it is used (just before RunPoolLoop).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

// ---- nonce-cursor persistence -------------------------------------------------------
// Resume scanning past where we left off before a reconnect or process restart, instead of
// restarting at the range base -- which re-scans ground already covered and re-submits nonces
// we already sent, drawing duplicate-share rejects (worst right after a crash/restart, when the
// in-memory submitted-key dedup is gone too). Pure effective-hashrate: the PoW/solution and the
// digest math are untouched. State is a tiny text file ("range cursor" per line) under
// XDG_DATA_HOME (or ~/.local/share)/matador-miner/nonce-cursors.
namespace {
using CursorRanges = std::vector<std::pair<uint64_t, uint64_t>>;  // (range_base, max_cursor)
constexpr size_t kCursorRanges = 8;                               // keep the most-recent few blocks

std::string HomeDir()
{
    if (const char* h = std::getenv("HOME"); h && *h) return h;
    // systemd system services usually run with HOME UNSET -- fall back to the passwd entry so the
    // state dir still resolves (root service -> /root, a user service -> /home/<user>).
    if (const struct passwd* pw = ::getpwuid(::getuid()); pw && pw->pw_dir && *pw->pw_dir) return pw->pw_dir;
    return {};
}

std::string CursorStateDir()
{
    if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) return std::string(x) + "/matador-miner";
    const std::string h = HomeDir();
    return h.empty() ? std::string{} : h + "/.local/share/matador-miner";
}

CursorRanges LoadCursorRanges()
{
    CursorRanges out;
    const std::string dir = CursorStateDir();
    if (dir.empty()) return out;
    std::ifstream f(dir + "/nonce-cursors");
    uint64_t range = 0, cursor = 0;
    while (f >> range >> cursor) {
        out.emplace_back(range, cursor);
        if (out.size() > kCursorRanges) out.erase(out.begin());
    }
    return out;
}

void SaveCursorRanges(const CursorRanges& r)
{
    const std::string dir = CursorStateDir();
    if (dir.empty()) return;
    if (const std::string h = HomeDir(); !h.empty()) {         // best-effort nested mkdir
        ::mkdir((h + "/.local").c_str(), 0755);
        ::mkdir((h + "/.local/share").c_str(), 0755);
    }
    ::mkdir(dir.c_str(), 0755);                                 // ignore EEXIST
    const std::string tmp = dir + "/nonce-cursors.tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return;
        for (const auto& [range, cursor] : r) f << range << ' ' << cursor << '\n';
    }
    std::rename(tmp.c_str(), (dir + "/nonce-cursors").c_str());  // atomic replace
}

uint64_t ResumeCursor(const CursorRanges& r, uint64_t range_base)
{
    for (const auto& [rb, cur] : r)
        if (rb == range_base && cur > range_base) return cur;
    return range_base;
}

void RecordCursor(CursorRanges& r, uint64_t range_base, uint64_t cursor)
{
    for (auto& [rb, cur] : r)
        if (rb == range_base) { if (cursor > cur) cur = cursor; return; }
    r.emplace_back(range_base, cursor);
    if (r.size() > kCursorRanges) r.erase(r.begin());
}
} // namespace
