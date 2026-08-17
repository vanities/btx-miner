// matador-miner: live GPU telemetry for the [gpu] heartbeat line (NVML via
// lazily-dlopen'd libnvidia-ml, no fork). Telemetry only -- never touches the
// solve/digest. Extracted verbatim from matador-miner.cpp; included into the
// single miner translation unit.
#pragma once

#include <dlfcn.h>
#include <mutex>

// Live GPU telemetry for the [gpu] heartbeat line. Read via NVML getters through a
// lazily-opened libnvidia-ml handle -- NO fork (unlike nvidia-smi), microseconds per
// query, safe alongside the live CUDA contexts. Every symbol is null-guarded; if the
// driver/lib is missing or a symbol fails to resolve, ok stays false and the [gpu]
// line is simply skipped. Telemetry only -- never touches the solve/digest.
struct GpuTelemetry {
    bool ok{false};
    unsigned temp_c{0}, sm_mhz{0}, mem_mhz{0}, pow_w{0}, fan_pct{0}, util_pct{0};
};
static GpuTelemetry QueryGpuTelemetry()
{
    struct Nvml {
        void* dev{nullptr};
        int (*GetTemp)(void*, int, unsigned*){nullptr};   // nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU=0, &c)
        int (*GetPower)(void*, unsigned*){nullptr};        // nvmlDeviceGetPowerUsage(dev, &mW)
        int (*GetClock)(void*, int, unsigned*){nullptr};   // nvmlDeviceGetClockInfo(dev, type, &MHz); SM=1, MEM=2
        int (*GetFan)(void*, unsigned*){nullptr};          // nvmlDeviceGetFanSpeed(dev, &pct)
        int (*GetUtil)(void*, void*){nullptr};             // nvmlDeviceGetUtilizationRates(dev, &nvmlUtilization_t)
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
    t.ok = true;
    return t;
}
