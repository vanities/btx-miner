// matador-miner: structured logging (mlog) + small log helpers.
// Extracted verbatim from matador-miner.cpp; included into the single miner
// translation unit. Self-contained so it could also back a future unit test.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>

#include <uint256.h>

// ===========================================================================
// 0. Structured, scope-tagged, timed logging (log-everything rule).
//    Levels gated by LOG_LEVEL env (debug<info<warn<error). NEVER logs the
//    rpc cookie/password (the payout address is public and safe to log).
// ===========================================================================
namespace mlog {

enum class Level { Debug = 0, Info = 1, Warn = 2, Error = 3 };

static Level g_level = Level::Info;
static std::mutex g_mu;
static bool  g_color = false;   // set in Init(): true on a TTY or under systemd's journal (and not NO_COLOR)

// ANSI SGR codes. Kept tiny + local so the logger has no extra deps.
namespace ansi {
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* DIM   = "\033[2m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW= "\033[33m";
    static constexpr const char* BRED  = "\033[1;31m";   // bold red (errors)
}

static Level ParseLevel(const char* s)
{
    if (s == nullptr) return Level::Info;
    std::string v{s};
    for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (v == "debug" || v == "trace") return Level::Debug;
    if (v == "info") return Level::Info;
    if (v == "warn" || v == "warning") return Level::Warn;
    if (v == "error") return Level::Error;
    return Level::Info;
}

static void Init()
{
    g_level = ParseLevel(std::getenv("LOG_LEVEL"));
    // Color when writing to a terminal. Honor NO_COLOR (https://no-color.org) and
    // FORCE_COLOR (e.g. for `docker logs -t` or piping into a color-aware pager).
    const char* nc = std::getenv("NO_COLOR");
    const char* fc = std::getenv("FORCE_COLOR");
    // Also colorize when systemd wired stderr to the journal (it exports
    // JOURNAL_STREAM): journald stores the ANSI bytes and `journalctl` replays
    // them, so `journalctl -u matador-miner` is colorized with no client filter.
    const bool journal = std::getenv("JOURNAL_STREAM") != nullptr;
    g_color = (fc && *fc) ? true : ((nc == nullptr) && (isatty(STDERR_FILENO) || journal));
}

static const char* LevelColor(Level l)
{
    switch (l) {
        case Level::Debug: return ansi::DIM;
        case Level::Info:  return ansi::GREEN;
        case Level::Warn:  return ansi::YELLOW;
        case Level::Error: return ansi::BRED;
    }
    return "";
}

static const char* LevelTag(Level l)
{
    switch (l) {
        case Level::Debug: return "DEBUG";
        case Level::Info:  return "INFO ";
        case Level::Warn:  return "WARN ";
        case Level::Error: return "ERROR";
    }
    return "?????";
}

static std::string NowStamp()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t = system_clock::to_time_t(now);
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm_buf{};
    gmtime_r(&t, &tm_buf);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                  tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, static_cast<int>(ms.count()));
    return std::string(buf);
}

// Wrap every match of `re` in SGR `code` ... reset (perl: s{(pat)}{\e[CODEm$1\e[0m}g).
static void Sub(std::string& s, const std::regex& re, const char* code)
{
    s = std::regex_replace(s, re, std::string("\033[") + code + "m$&\033[0m");
}

// Field-level colorizer: an in-binary port of the `_matcolor` zsh filter, so a
// plain `journalctl -u matador-miner` (or any TTY) shows the same palette with no
// client-side pipe. Runs on the message body only -- a few lines per window, so
// not hot. Order mirrors _matcolor exactly: the passes interact (e.g. pool-nonce/s
// must color before nonce/s, via the (^|[^-]) guard), so do NOT reorder.
static std::string Colorize(std::string s)
{
    static const std::regex re_error   (R"(\bERROR\b)");
    static const std::regex re_warn    (R"(\bWARN\b)");
    static const std::regex re_stats   (R"(\[stats\])");
    static const std::regex re_share   (R"(\[share\])");
    static const std::regex re_gpu     (R"(\[gpu\])");
    static const std::regex re_pool    (R"(\[pool\])");
    static const std::regex re_solve   (R"(\[solve\])");
    static const std::regex re_update  (R"(\[update\])");
    static const std::regex re_watchdog(R"(\[watchdog\])");
    static const std::regex re_init    (R"(\[init\])");
    static const std::regex re_rej0    (R"((?:rejected|rej)=0\b)");
    static const std::regex re_rejn    (R"((?:rejected|rej)=[1-9]\d*)");
    static const std::regex re_rejpct  (R"(rej%=([\d.]+))");
    static const std::regex re_acc     (R"((?:accepted|acc)=\d+)");
    static const std::regex re_ACCEPTED(R"(\bACCEPTED\b)");
    static const std::regex re_poolnps (R"(pool-nonce/s=[\d.]+[kMGBT]?)");
    static const std::regex re_nps     (R"((^|[^-])(nonce/s=[\d.]+[kMGBT]?))");
    static const std::regex re_scan    (R"(scan=[\d.]+[kMGBT]?N/s)");
    static const std::regex re_spacing (R"(spacing=[\d.]+[kMGBT]?)");
    static const std::regex re_netdiff (R"(net-diff=[\d.]+[kMGBT]?)");
    static const std::regex re_pooldiff(R"(pool-diff=[\d.]+[kMGBT]?)");
    static const std::regex re_dbatch  (R"(digest-batch/s=[\d.]+[kMGBT]?)");
    static const std::regex re_stale   (R"(stale=[1-9]\d*)");
    static const std::regex re_temp    (R"(temp=(\d+)C)");
    static const std::regex re_clk     (R"(clk=\d+)");
    static const std::regex re_mem     (R"(\bmem=\d+)");
    static const std::regex re_pow     (R"(pow=\d+W)");
    static const std::regex re_fan     (R"(fan=\d+%)");
    static const std::regex re_util    (R"(util=\d+%)");
    static const std::regex re_noncew  (R"(nonce/W=\d+)");
    static const std::regex re_cleany  (R"(clean=yes)");
    static const std::regex re_cleann  (R"(clean=no)");
    static const std::regex re_height  (R"(height=\d+)");
    static const std::regex re_version (R"(\bv\d+\.\d+\.\d+[\w.-]*)");

    // Cheap fast path: every FIELD pass below matches a literal "key=value", so a
    // line with no '=' can only ever hit the scope-tag/word passes. Skipping the
    // ~25 field-regex passes on such lines is output-identical BY CONSTRUCTION (a
    // regex requiring '=' cannot match a '='-free line; pinned byte-for-byte by
    // harness/miner_log_colorize_test.cpp) and keeps the regex cost off plain
    // prose lines. Tag/word passes (ERROR/[pool]/ACCEPTED/vX.Y.Z) still run, in
    // the exact original order.
    const bool has_fields = s.find('=') != std::string::npos;

    Sub(s, re_error, "1;91");  Sub(s, re_warn, "1;33");
    Sub(s, re_stats, "1;95");  Sub(s, re_share, "1;92");
    Sub(s, re_gpu, "1;94");
    Sub(s, re_pool, "36");     Sub(s, re_solve, "34");
    Sub(s, re_update, "1;93"); Sub(s, re_watchdog, "35"); Sub(s, re_init, "1;97");
    if (has_fields) {
        Sub(s, re_rej0, "92");     Sub(s, re_rejn, "1;91");

        // rej%=NN : >0 bold-red else green (label stays default, like the perl)
        {
            std::string out; out.reserve(s.size());
            auto end = std::sregex_iterator();
            size_t last = 0;
            for (auto it = std::sregex_iterator(s.begin(), s.end(), re_rejpct); it != end; ++it) {
                const auto& m = *it;
                out.append(s, last, static_cast<size_t>(m.position()) - last);
                const char* c = (std::stod(m[1].str()) > 0) ? "1;91" : "92";
                out += "rej%=\033["; out += c; out += "m"; out += m[1].str(); out += "\033[0m";
                last = static_cast<size_t>(m.position()) + static_cast<size_t>(m.length());
            }
            out.append(s, last, std::string::npos);
            s.swap(out);
        }

        Sub(s, re_acc, "32");
    }
    Sub(s, re_ACCEPTED, "1;92");
    if (has_fields) {
        Sub(s, re_poolnps, "1;92");
        // nonce/s, but not the one inside pool-nonce/s (perl's (?<!-) lookbehind);
        // (^|[^-]) consumes+re-emits the guard char via $1.
        s = std::regex_replace(s, re_nps, std::string("$1\033[1;93m$2\033[0m"));
        Sub(s, re_scan, "36");     Sub(s, re_spacing, "2");
        Sub(s, re_netdiff, "1;96");Sub(s, re_pooldiff, "94");
        Sub(s, re_dbatch, "96");   Sub(s, re_stale, "33");

        // temp=NNC : >=82 bold-red, >=75 yellow, else green (label stays default)
        {
            std::string out; out.reserve(s.size());
            auto end = std::sregex_iterator();
            size_t last = 0;
            for (auto it = std::sregex_iterator(s.begin(), s.end(), re_temp); it != end; ++it) {
                const auto& m = *it;
                out.append(s, last, static_cast<size_t>(m.position()) - last);
                int t = std::stoi(m[1].str());
                const char* c = (t >= 82) ? "1;91" : (t >= 75) ? "33" : "92";
                out += "temp=\033["; out += c; out += "m"; out += m[1].str(); out += "C\033[0m";
                last = static_cast<size_t>(m.position()) + static_cast<size_t>(m.length());
            }
            out.append(s, last, std::string::npos);
            s.swap(out);
        }

        Sub(s, re_clk, "36");      Sub(s, re_mem, "36");
        Sub(s, re_pow, "33");      Sub(s, re_fan, "2");
        Sub(s, re_util, "35");     Sub(s, re_noncew, "1");
        Sub(s, re_cleany, "32");   Sub(s, re_cleann, "2");
        Sub(s, re_height, "2");
    }
    Sub(s, re_version, "1");
    return s;
}

static void Emit(Level l, const std::string& msg)
{
    if (l < g_level) return;
    if (!g_color) {
        std::lock_guard<std::mutex> lk(g_mu);
        std::clog << "[" << NowStamp() << "] " << LevelTag(l) << " " << msg << std::endl;
        return;
    }
    // Colorized: dim timestamp, level tag in its level color, then the body run
    // through Colorize() -- the in-binary port of the `_matcolor` filter -- so a
    // plain `journalctl -u matador-miner` shows the same field palette with no
    // client-side pipe. (The --log-file copy strips ANSI; see log_tee.h.)
    //
    // Colorize runs OUTSIDE g_mu: it is dozens of std::regex passes per line, and
    // holding the global log mutex through them serialized every logging thread
    // (reader, watchdog, heartbeat, solve loop) behind the slowest line. The
    // colorized body depends only on msg, so computing it lock-free is safe;
    // g_mu still orders the final stream write, and NowStamp() stays INSIDE the
    // lock so emitted timestamps remain monotone in output order.
    const std::string body = Colorize(msg);
    std::lock_guard<std::mutex> lk(g_mu);
    std::clog << ansi::DIM << "[" << NowStamp() << "]" << ansi::RESET << " "
              << LevelColor(l) << LevelTag(l) << ansi::RESET << " "
              << body << std::endl;
}

// Convenience: build a line from a string then emit.
#define LOGD(expr) do { if (::mlog::g_level <= ::mlog::Level::Debug) { std::ostringstream _o; _o << expr; ::mlog::Emit(::mlog::Level::Debug, _o.str()); } } while (0)
#define LOGI(expr) do { if (::mlog::g_level <= ::mlog::Level::Info)  { std::ostringstream _o; _o << expr; ::mlog::Emit(::mlog::Level::Info,  _o.str()); } } while (0)
#define LOGW(expr) do { if (::mlog::g_level <= ::mlog::Level::Warn)  { std::ostringstream _o; _o << expr; ::mlog::Emit(::mlog::Level::Warn,  _o.str()); } } while (0)
#define LOGE(expr) do { std::ostringstream _o; _o << expr; ::mlog::Emit(::mlog::Level::Error, _o.str()); } while (0)

} // namespace mlog

// Wall-clock span/timer helper (ms). Named Timer (not Span) to avoid colliding
// with btx's own Span<> template in src/span.h.
struct Timer {
    std::chrono::steady_clock::time_point t0{std::chrono::steady_clock::now()};
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }
};

// Short hash prefix for logs (first 12 hex chars).
static std::string Short(const std::string& hex) { return hex.substr(0, std::min<size_t>(12, hex.size())); }
static std::string Short(const uint256& h) { return Short(h.GetHex()); }
