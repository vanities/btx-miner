// matador-miner: live GPU telemetry for the [gpu] heartbeat line (NVML via
// lazily-dlopen'd libnvidia-ml, no fork). Telemetry only -- never touches the
// solve/digest. Extracted verbatim from matador-miner.cpp; included into the
// single miner translation unit.
#pragma once

#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unistd.h>   // getpid: tell OUR compute process apart from a foreign tenant

// Live GPU telemetry for the [gpu] heartbeat line. Read via NVML getters through a
// lazily-opened libnvidia-ml handle -- NO fork (unlike nvidia-smi), microseconds per
// query, safe alongside the live CUDA contexts. Every symbol is null-guarded; if the
// driver/lib is missing or a symbol fails to resolve, ok stays false and the [gpu]
// line is simply skipped. Telemetry only -- never touches the solve/digest.
struct GpuTelemetry {
    bool ok{false};
    unsigned temp_c{0}, sm_mhz{0}, mem_mhz{0}, pow_w{0}, fan_pct{0}, util_pct{0};
    // WHY the card is running at the clock it is. Without this a slow rig is a guessing
    // game: a 2026-08-21 session spent an afternoon on a thermal hypothesis for a board
    // that was reporting SwPowerCap the whole time, and never once asserted a thermal
    // slowdown. One NVML call answers it. 0 = no reason asserted (not throttled).
    unsigned long long throttle{0};
    bool has_throttle{false};
    // Foreign compute tenants: OTHER processes holding this GPU. A desktop AI job or a
    // second miner silently halves ep/s, and every symptom points at the miner instead.
    // Excludes our own pid. has_procs=false means NVML could not answer (do not report 0).
    bool has_procs{false};
    unsigned foreign_procs{0};
    unsigned long long foreign_mib{0};
};

// NVML clocksThrottleReasons bits -> short greppable tags for the [gpu] line.
// Bit values are ABI-stable in nvml.h (nvmlClocksThrottleReasonGpuIdle = 1, ...).
static std::string ThrottleReasonsStr(unsigned long long m)
{
    if (m == 0) return "none";
    std::string s;
    auto add = [&](unsigned long long bit, const char* tag) {
        if ((m & bit) == 0) return;
        if (!s.empty()) s += ",";
        s += tag;
        m &= ~bit;
    };
    add(0x1ULL,  "idle");
    add(0x2ULL,  "app-clocks");
    add(0x4ULL,  "sw-power-cap");     // the common one: board is at its power limit
    add(0x8ULL,  "hw-slowdown");
    add(0x10ULL, "sync-boost");
    add(0x20ULL, "sw-thermal");
    add(0x40ULL, "hw-thermal");
    add(0x80ULL, "hw-power-brake");
    add(0x100ULL,"display-clocks");
    if (m != 0) { if (!s.empty()) s += ","; s += "other"; }
    return s;
}
static GpuTelemetry QueryGpuTelemetry()
{
    struct Nvml {
        void* dev{nullptr};
        int (*GetTemp)(void*, int, unsigned*){nullptr};   // nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU=0, &c)
        int (*GetPower)(void*, unsigned*){nullptr};        // nvmlDeviceGetPowerUsage(dev, &mW)
        int (*GetClock)(void*, int, unsigned*){nullptr};   // nvmlDeviceGetClockInfo(dev, type, &MHz); SM=1, MEM=2
        int (*GetFan)(void*, unsigned*){nullptr};          // nvmlDeviceGetFanSpeed(dev, &pct)
        int (*GetUtil)(void*, void*){nullptr};             // nvmlDeviceGetUtilizationRates(dev, &nvmlUtilization_t)
        int (*GetThrottle)(void*, unsigned long long*){nullptr};  // nvmlDeviceGetCurrentClocksThrottleReasons
        int (*GetProcs)(void*, unsigned*, void*){nullptr};        // nvmlDeviceGetComputeRunningProcesses_v3/_v2
        bool inited{false};
    };
    static Nvml n;
    static std::once_flag once;
    std::call_once(once, [] {
        void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) return;
        auto Init   = reinterpret_cast<int (*)()>(dlsym(lib, "nvmlInit_v2"));
        auto Handle = reinterpret_cast<int (*)(unsigned, void**)>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
        if (Init == nullptr || Handle == nullptr || Init() != 0) return;
        void* h = nullptr;
        if (Handle(0, &h) != 0 || h == nullptr) return;
        n.dev     = h;
        n.GetTemp  = reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(lib, "nvmlDeviceGetTemperature"));
        n.GetPower = reinterpret_cast<int (*)(void*, unsigned*)>(dlsym(lib, "nvmlDeviceGetPowerUsage"));
        n.GetClock = reinterpret_cast<int (*)(void*, int, unsigned*)>(dlsym(lib, "nvmlDeviceGetClockInfo"));
        n.GetFan   = reinterpret_cast<int (*)(void*, unsigned*)>(dlsym(lib, "nvmlDeviceGetFanSpeed"));
        n.GetUtil  = reinterpret_cast<int (*)(void*, void*)>(dlsym(lib, "nvmlDeviceGetUtilizationRates"));
        n.GetThrottle = reinterpret_cast<int (*)(void*, unsigned long long*)>(
            dlsym(lib, "nvmlDeviceGetCurrentClocksThrottleReasons"));
        // _v3 first (current nvmlProcessInfo_t), then _v2, then the unversioned symbol.
        // All three write the same leading {pid, usedGpuMemory} pair, which is all we read,
        // so the extra trailing fields in the newer struct are harmless slack.
        n.GetProcs = reinterpret_cast<int (*)(void*, unsigned*, void*)>(
            dlsym(lib, "nvmlDeviceGetComputeRunningProcesses_v3"));
        if (n.GetProcs == nullptr)
            n.GetProcs = reinterpret_cast<int (*)(void*, unsigned*, void*)>(
                dlsym(lib, "nvmlDeviceGetComputeRunningProcesses_v2"));
        if (n.GetProcs == nullptr)
            n.GetProcs = reinterpret_cast<int (*)(void*, unsigned*, void*)>(
                dlsym(lib, "nvmlDeviceGetComputeRunningProcesses"));
        n.inited   = true;
    });
    GpuTelemetry t;
    if (!n.inited || n.dev == nullptr) return t;
    unsigned v = 0;
    if (n.GetTemp  && n.GetTemp(n.dev, 0, &v) == 0) t.temp_c = v;
    if (n.GetPower && n.GetPower(n.dev, &v) == 0)    t.pow_w  = v / 1000;   // mW -> W
    if (n.GetClock && n.GetClock(n.dev, 1, &v) == 0) t.sm_mhz = v;         // NVML_CLOCK_SM
    if (n.GetClock && n.GetClock(n.dev, 2, &v) == 0) t.mem_mhz = v;        // NVML_CLOCK_MEM
    if (n.GetFan   && n.GetFan(n.dev, &v) == 0)      t.fan_pct = v;
    struct { unsigned gpu; unsigned memory; } util{0, 0};                  // == nvmlUtilization_t
    if (n.GetUtil && n.GetUtil(n.dev, &util) == 0)   t.util_pct = util.gpu;
    unsigned long long mask = 0;
    if (n.GetThrottle && n.GetThrottle(n.dev, &mask) == 0) { t.throttle = mask; t.has_throttle = true; }
    if (n.GetProcs) {
        // Widest nvmlProcessInfo_t (v3): the older versions write a prefix of this, so a
        // v3-sized slot is always big enough and never under-reads.
        struct ProcInfo {
            unsigned pid;
            unsigned long long used_mem;   // bytes; NVML_VALUE_NOT_AVAILABLE when unreadable
            unsigned gi_id;
            unsigned ci_id;
        };
        ProcInfo procs[32];
        unsigned count = 32;
        const int rc = n.GetProcs(n.dev, &count, procs);
        // 0 = ok. 7 = INSUFFICIENT_SIZE: more than 32 tenants, count holds the real total;
        // the array still holds the first 32, so report what we can rather than nothing.
        if (rc == 0 || rc == 7) {
            const unsigned mine = static_cast<unsigned>(::getpid());
            const unsigned n_read = (rc == 0) ? count : 32u;
            for (unsigned i = 0; i < n_read; ++i) {
                if (procs[i].pid == mine) continue;
                ++t.foreign_procs;
                // NVML reports an unreadable footprint as ~0ULL; do not add that to a sum.
                if (procs[i].used_mem != ~0ULL) t.foreign_mib += procs[i].used_mem / (1024 * 1024);
            }
            t.has_procs = true;
        }
    }
    t.ok = true;
    return t;
}
