// matador-miner: multi-GPU rig-level stats aggregation ([stats-all]).
// Each forked child drops a tiny key=value snapshot at its ~30s [stats] heartbeat;
// the supervisor sums the fresh snapshots and prints ONE rig-level line every 60s:
//
//   [stats-all] gpus=5/5 ep/s=6.05 acc=42 rej=0 pow=2980W maxtemp=71C
//               | per-gpu ep/s: 1.21 1.21 1.21 1.21 1.21
//
// The per-card [stats] lines are unchanged (needed to diagnose a sick card); this adds
// the rig total that HiveOS/pools compute but a journalctl/console user never sees, and
// makes a dead/respawning card visible at a glance (gpus=4/5 + "down" in the list).
// Snapshots live under (XDG_DATA_HOME | ~/.local/share)/matador-miner/mgpu-stats/ and
// are written atomically (tmp + rename); a snapshot older than kStaleS counts as down.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mgpu {

constexpr int kAggIntervalS = 60;   // supervisor print cadence
constexpr int kStaleS = 90;         // snapshot older than this = child considered down

inline std::string HomeDirMg()
{
    if (const char* h = std::getenv("HOME"); h && *h) return h;
    if (const struct passwd* pw = ::getpwuid(::getuid()); pw && pw->pw_dir && *pw->pw_dir) return pw->pw_dir;
    return {};
}

inline std::string StatsDir()
{
    std::string base;
    if (const char* x = std::getenv("XDG_DATA_HOME"); x && *x) base = std::string(x) + "/matador-miner";
    else {
        const std::string h = HomeDirMg();
        if (h.empty()) return {};
        base = h + "/.local/share/matador-miner";
    }
    return base + "/mgpu-stats";
}

inline bool IsChild() { return std::getenv("MATADOR_MULTI_GPU_CHILD") != nullptr; }

// Multi-GPU children stay QUIET on the console by default: their per-card
// [stats]/[stats-avg]/[gpu] lines are suppressed (the supervisor's [stats-all]/
// [stats-all-avg] roll-ups are the rig-level truth, and the per-card numbers stay
// available via the per-child status API and the [stats-all] per-gpu breakdown).
// MATADOR_MGPU_CHILD_STATS=1 restores the full per-card console lines for debugging.
inline bool ChildConsoleStatsEnabled()
{
    static const bool e = [] {
        const char* v = std::getenv("MATADOR_MGPU_CHILD_STATS");
        return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    return e;
}

// Per-child heartbeat snapshot: live rates + 1h/24h averages + GPU telemetry.
struct ChildSnap {
    bool fresh{false};
    // ENC_RC: episodes/s is the card's throughput. The v3 digest-candidate and
    // pre-hash-scan rates that used to ride along measured a pipeline that is gone.
    double ep_s{0};
    uint64_t acc{0}, rej{0};
    double a1_ep{0}, a1_acc_hr{0};
    double a24_ep{0}, a24_acc_hr{0};
    double pow_w{0};
    int temp_c{0};
};

// Child side: atomically publish this card's latest heartbeat numbers. Failure is
// silent-by-design (stats are best-effort; never let telemetry interfere with mining).
inline void WriteChildSnapshot(const ChildSnap& s)
{
    const char* idx = std::getenv("MATADOR_MULTI_GPU_CHILD_INDEX");
    const std::string dir = StatsDir();
    if (!idx || dir.empty()) return;
    ::mkdir(dir.substr(0, dir.rfind('/')).c_str(), 0755);
    ::mkdir(dir.c_str(), 0755);
    const std::string path = dir + "/child-" + std::string(idx) + ".stats";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return;
        f << "ts=" << static_cast<int64_t>(::time(nullptr)) << "\n"
          << "ep_s=" << s.ep_s << "\n"
          << "acc=" << s.acc << "\n"
          << "rej=" << s.rej << "\n"
          << "a1_ep=" << s.a1_ep << "\n"
          << "a1_acc_hr=" << s.a1_acc_hr << "\n"
          << "a24_ep=" << s.a24_ep << "\n"
          << "a24_acc_hr=" << s.a24_acc_hr << "\n"
          << "pow_w=" << s.pow_w << "\n"
          << "temp_c=" << s.temp_c << "\n";
    }
    ::rename(tmp.c_str(), path.c_str());
}

inline ChildSnap ReadChildSnapshot(const std::string& dir, size_t index)
{
    ChildSnap s;
    std::ifstream f(dir + "/child-" + std::to_string(index) + ".stats");
    if (!f) return s;
    int64_t ts = 0;
    std::string line;
    while (std::getline(f, line)) {
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq);
        const char* v = line.c_str() + eq + 1;
        if (k == "ts") ts = std::strtoll(v, nullptr, 10);
        else if (k == "ep_s") s.ep_s = std::strtod(v, nullptr);
        else if (k == "acc") s.acc = std::strtoull(v, nullptr, 10);
        else if (k == "rej") s.rej = std::strtoull(v, nullptr, 10);
        else if (k == "a1_ep") s.a1_ep = std::strtod(v, nullptr);
        else if (k == "a1_acc_hr") s.a1_acc_hr = std::strtod(v, nullptr);
        else if (k == "a24_ep") s.a24_ep = std::strtod(v, nullptr);
        else if (k == "a24_acc_hr") s.a24_acc_hr = std::strtod(v, nullptr);
        else if (k == "pow_w") s.pow_w = std::strtod(v, nullptr);
        else if (k == "temp_c") s.temp_c = static_cast<int>(std::strtol(v, nullptr, 10));
    }
    s.fresh = ts > 0 && (::time(nullptr) - ts) <= kStaleS;
    return s;
}

// Supervisor side: wipe stale snapshots from a previous run so gpus=N/M starts honest.
inline void ClearSnapshots(size_t total)
{
    const std::string dir = StatsDir();
    if (dir.empty()) return;
    for (size_t i = 0; i < total; ++i) {
        ::unlink((dir + "/child-" + std::to_string(i) + ".stats").c_str());
    }
}

// Supervisor aggregation loop (runs on its own thread; waitpid owns the main one).
// FmtRate comes from miner_format.h, included ahead of this header in the miner TU.
inline void SupervisorStatsLoop(std::atomic<bool>& stop, size_t total)
{
    const std::string dir = StatsDir();
    if (dir.empty() || total == 0) return;
    while (!stop.load()) {
        for (int i = 0; i < kAggIntervalS * 2 && !stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (stop.load()) break;
        double eps = 0, pow = 0;
        double a1e = 0, a1h = 0;
        double a24e = 0, a24h = 0;
        uint64_t acc = 0, rej = 0;
        int maxtemp = 0;
        size_t alive = 0;
        std::string per_gpu;
        for (size_t i = 0; i < total; ++i) {
            const ChildSnap s = ReadChildSnapshot(dir, i);
            if (!per_gpu.empty()) per_gpu += " ";
            if (s.fresh) {
                ++alive;
                eps += s.ep_s;
                acc += s.acc; rej += s.rej;
                a1h += s.a1_acc_hr; a1e += s.a1_ep;
                a24h += s.a24_acc_hr; a24e += s.a24_ep;
                pow += s.pow_w;
                if (s.temp_c > maxtemp) maxtemp = s.temp_c;
                per_gpu += FmtRate(s.ep_s);
            } else {
                per_gpu += "down";
            }
        }
        if (alive == 0) continue;   // children not heartbeating yet (first ~30s)
        LOGI("[stats-all] gpus=" << alive << "/" << total
             << " ep/s=" << FmtRate(eps)
             << " acc=" << acc << " rej=" << rej
             << (pow > 0 ? (" pow=" + std::to_string(static_cast<int>(pow + 0.5)) + "W") : "")
             << (maxtemp > 0 ? (" maxtemp=" + std::to_string(maxtemp) + "C") : "")
             << " | per-gpu ep/s: " << per_gpu);
        // Averages roll-up (mirrors the per-card [stats-avg] shape). Skipped until
        // at least one child has non-zero averages (solo mode reports none).
        if (a1e > 0 || a24e > 0) {
            LOGI("[stats-all-avg]"
                 << " 1h: ep/s=" << FmtRate(a1e)
                 << " acc/hr=" << static_cast<uint64_t>(a1h + 0.5)
                 << " | 24h: ep/s=" << FmtRate(a24e)
                 << " acc/hr=" << static_cast<uint64_t>(a24h + 0.5));
        }
    }
}

// RAII wrapper so every supervisor exit path stops + joins the aggregator thread.
struct SupervisorStats {
    std::atomic<bool> stop{false};
    std::thread t;
    explicit SupervisorStats(size_t total)
    {
        ClearSnapshots(total);
        t = std::thread([this, total] { SupervisorStatsLoop(stop, total); });
    }
    ~SupervisorStats()
    {
        stop.store(true);
        if (t.joinable()) t.join();
    }
};

}  // namespace mgpu
