// matador-miner: optional NVIDIA GPU tuning (clock offset / power limit / locked
// clocks / fan duty / mem-clock offset) via NVML loaded at RUNTIME with dlopen,
// so the release binary keeps ZERO link-time NVML dependency. Includes the
// shutdown-revert watcher that restores stock on exit/signal.
// SECTION of the single miner translation unit (uses mlog + the g_tune state);
// #included at the point those are in scope. Extracted verbatim.
#pragma once

// Apply optional NVIDIA GPU tuning (clock offset / power limit / locked clocks / fan duty)
// to every NVIDIA GPU via NVML, loaded at RUNTIME with dlopen so the release binary keeps
// ZERO link-time NVML dependency (libnvidia-ml is only touched when a knob is set). On
// a power-capped card a positive clock offset shifts the V/F curve up -> the same watts
// buy more clock -> the (compute-bound) digest runs faster -> more nonces/s; pinning the
// fan high holds the thermal margin so the boost can't step down (flatter nps in a warm
// room). Same NVML calls LACT/lolMiner make; no third-party code. Requires root (NVML
// control is privileged). NEVER fatal: any failure logs and mining continues.
//
// Everything applied here is REVERTED to stock on shutdown: ApplyGpuTuning keeps NVML
// open and records what it changed in g_tune; a process-global SIGTERM/SIGINT handler
// (TuneTermHandler) reverts then re-raises, and std::atexit covers clean exits. So a
// `systemctl stop`/`restart` leaves the card on its stock fan curve, clocks and power -
// not pinned.
//   clk_offset    MHz GPC V/F offset           (0 = leave stock)
//   power_limit_w board power limit in W        (0 = leave as-is)
//   lock_mhz      pin core clock to this MHz    (0 = unlocked; stability/consistency)
//   fan_pct       pin fan duty in %            (0 = leave on the driver's auto curve)
namespace {
struct GpuTuneState {
    void* lib{nullptr};                            // kept open for the process lifetime (revert)
    int (*SetOff)(void*, int){nullptr};            // nvmlDeviceSetGpcClkVfOffset
    int (*SetMemOff)(void*, int){nullptr};         // nvmlDeviceSetMemClkVfOffset
    int (*SetPower)(void*, unsigned){nullptr};     // nvmlDeviceSetPowerManagementLimit
    int (*ResetLock)(void*){nullptr};              // nvmlDeviceResetGpuLockedClocks
    int (*ResetMemLock)(void*){nullptr};           // nvmlDeviceResetMemoryLockedClocks
    int (*SetDefFan)(void*, unsigned){nullptr};    // nvmlDeviceSetDefaultFanSpeed_v2
    const char* (*ErrStr)(int){nullptr};           // nvmlErrorString
    struct Dev { void* h{nullptr}; unsigned def_power_mw{0}; unsigned nfans{0}; std::string name; };
    std::vector<Dev> devs;
    bool did_offset{false}, did_mem_offset{false}, did_power{false}, did_lock{false}, did_mem_lock{false}, did_fan{false};
    std::atomic<bool> reverted{false};
};
GpuTuneState g_tune;

// Restore stock fan curve / clock offset / locked clocks / power limit for every device we
// tuned. Idempotent (runs at most once). Reverts only knobs that were applied. From the
// signal handler (from_signal=true) it avoids iostream and uses async-signal-safe write()
// for the confirmation line; the atexit (clean-exit) path logs normally.
void RevertGpuTuning(bool from_signal)
{
    if (g_tune.reverted.exchange(true)) return;
    for (auto& d : g_tune.devs) {
        if (g_tune.did_fan && g_tune.SetDefFan)
            for (unsigned f = 0; f < d.nfans; ++f) g_tune.SetDefFan(d.h, f);
        if (g_tune.did_lock && g_tune.ResetLock) g_tune.ResetLock(d.h);
        if (g_tune.did_mem_lock && g_tune.ResetMemLock) g_tune.ResetMemLock(d.h);
        if (g_tune.did_offset && g_tune.SetOff) g_tune.SetOff(d.h, 0);
        if (g_tune.did_mem_offset && g_tune.SetMemOff) g_tune.SetMemOff(d.h, 0);
        if (g_tune.did_power && g_tune.SetPower && d.def_power_mw) g_tune.SetPower(d.h, d.def_power_mw);
    }
    if (from_signal) {
        static const char m[] = "[gpu-tune] reverted GPU tuning to stock on signal\n";
        ssize_t rc = write(STDERR_FILENO, m, sizeof(m) - 1); (void)rc;
    } else {
        LOGI("[gpu-tune] reverted GPU tuning to stock (fan/offset/lock/power) on exit");
    }
}

// SIGTERM/SIGINT -> revert, then re-raise to exit with the signal's status. A process-global
// sigaction handler (NOT a sigwait thread) is used deliberately: systemd's SIGTERM can be
// delivered to ANY thread, and a sigwait/blocked-mask scheme only reverts if EVERY thread
// blocks the signal -- a library/logger thread spawned before us that doesn't would take the
// default action and kill the process first (observed). A handler fires regardless of which
// thread gets the signal. NVML setters from a handler aren't strictly async-signal-safe, but
// this is teardown and the calls are simple; reliability wins over the theoretical hazard.
void TuneTermHandler(int sig)
{
    RevertGpuTuning(true);
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

// Install the shutdown-revert: sigaction for SIGTERM/SIGINT + atexit for clean exits.
void StartTuneRevertWatcher()
{
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return;
    std::atexit(+[] { RevertGpuTuning(false); });
    struct sigaction sa{};
    sa.sa_handler = TuneTermHandler;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, SIGINT);
    sigaddset(&sa.sa_mask, SIGTERM);   // hold the other term signal off while reverting
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    LOGI("[gpu-tune] shutdown-revert installed (SIGTERM/SIGINT/exit -> restore stock)");
}
} // namespace

static void ApplyGpuTuning(int clk_offset, int power_limit_w, int lock_mhz, int fan_pct, int mem_clk_offset,
                           int lock_mem_mhz)
{
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        LOGW("[gpu-tune] libnvidia-ml not found; cannot apply GPU tuning (NVIDIA driver installed?)");
        return;
    }
    using fn_ret = int (*)();
    using fn_count = int (*)(unsigned*);
    using fn_handle = int (*)(unsigned, void**);
    using fn_minmax = int (*)(void*, int*, int*);
    using fn_set = int (*)(void*, int);
    using fn_name = int (*)(void*, char*, unsigned);
    using fn_err = const char* (*)(int);
    using fn_setu = int (*)(void*, unsigned);
    using fn_constr = int (*)(void*, unsigned*, unsigned*);
    using fn_lock = int (*)(void*, unsigned, unsigned);
    using fn_dev = int (*)(void*);
    using fn_u_get = int (*)(void*, unsigned*);
    using fn_iget = int (*)(void*, int*);
    using fn_fan2 = int (*)(void*, unsigned, unsigned);
    auto Init = reinterpret_cast<fn_ret>(dlsym(lib, "nvmlInit_v2"));
    auto Count = reinterpret_cast<fn_count>(dlsym(lib, "nvmlDeviceGetCount_v2"));
    auto Handle = reinterpret_cast<fn_handle>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
    auto MinMax = reinterpret_cast<fn_minmax>(dlsym(lib, "nvmlDeviceGetGpcClkMinMaxVfOffset"));
    auto SetOff = reinterpret_cast<fn_set>(dlsym(lib, "nvmlDeviceSetGpcClkVfOffset"));
    auto MemMinMax = reinterpret_cast<fn_minmax>(dlsym(lib, "nvmlDeviceGetMemClkMinMaxVfOffset"));
    auto SetMemOff = reinterpret_cast<fn_set>(dlsym(lib, "nvmlDeviceSetMemClkVfOffset"));
    auto GetOff = reinterpret_cast<fn_iget>(dlsym(lib, "nvmlDeviceGetGpcClkVfOffset"));
    auto GetMemOff = reinterpret_cast<fn_iget>(dlsym(lib, "nvmlDeviceGetMemClkVfOffset"));
    auto GetName = reinterpret_cast<fn_name>(dlsym(lib, "nvmlDeviceGetName"));
    auto ErrStr = reinterpret_cast<fn_err>(dlsym(lib, "nvmlErrorString"));
    auto Shutdown = reinterpret_cast<fn_ret>(dlsym(lib, "nvmlShutdown"));
    auto SetPower = reinterpret_cast<fn_setu>(dlsym(lib, "nvmlDeviceSetPowerManagementLimit"));
    auto PowerC = reinterpret_cast<fn_constr>(dlsym(lib, "nvmlDeviceGetPowerManagementLimitConstraints"));
    auto LockClk = reinterpret_cast<fn_lock>(dlsym(lib, "nvmlDeviceSetGpuLockedClocks"));
    auto DefPower = reinterpret_cast<fn_u_get>(dlsym(lib, "nvmlDeviceGetPowerManagementDefaultLimit"));
    auto ResetLock = reinterpret_cast<fn_dev>(dlsym(lib, "nvmlDeviceResetGpuLockedClocks"));
    auto LockMemClk = reinterpret_cast<fn_lock>(dlsym(lib, "nvmlDeviceSetMemoryLockedClocks"));
    auto ResetMemLock = reinterpret_cast<fn_dev>(dlsym(lib, "nvmlDeviceResetMemoryLockedClocks"));
    auto GetNumFans = reinterpret_cast<fn_u_get>(dlsym(lib, "nvmlDeviceGetNumFans"));
    auto SetFan = reinterpret_cast<fn_fan2>(dlsym(lib, "nvmlDeviceSetFanSpeed_v2"));
    auto SetDefFan = reinterpret_cast<fn_setu>(dlsym(lib, "nvmlDeviceSetDefaultFanSpeed_v2"));
    auto estr = [&](int rr) { return ErrStr ? ErrStr(rr) : "?"; };
    // Verify an applied V/F offset actually stuck, and re-apply once if not. Rapid
    // back-to-back restarts can race the prior process's revert-to-0 against this apply:
    // NVML returns success but the offset ends up 0 and the clock sits at stock (observed
    // 2026-06-25: log said +1000 MHz, GPU stayed at 13801 / ~4% slow). Read it back; on
    // mismatch retry once, then warn loud so it shows in the journal instead of silently
    // mining slow. Never touches the hash; no-op if the GET symbol is missing (older driver).
    auto verify_offset = [&](void* hh, const std::string& tg, const char* what, fn_set setfn, fn_iget getfn, int want) {
        if (getfn == nullptr) return;
        int got = 0;
        if (getfn(hh, &got) != 0 || got == want) return;  // read failed, or already correct
        LOGW(tg << " " << what << " readback " << got << " != " << want
             << " MHz (rapid-restart race?) -- re-applying");
        if (setfn) setfn(hh, want);
        got = 0;
        if (getfn(hh, &got) == 0 && got != want)
            LOGW(tg << " " << what << " STILL " << got << " != " << want
                 << " MHz -- GPU NOT tuned; check `nvidia-smi --query-gpu=clocks.mem,clocks.sm`");
        else
            LOGI(tg << " " << what << " re-applied OK (" << want << " MHz)");
    };
    if (!Init || !Count || !Handle) {
        LOGW("[gpu-tune] NVML core symbols missing (driver too old?); skipping");
        dlclose(lib);
        return;
    }
    int r = Init();
    if (r != 0) { LOGW("[gpu-tune] nvmlInit failed: " << estr(r)); dlclose(lib); return; }
    unsigned ngpu = 0;
    if (Count(&ngpu) != 0 || ngpu == 0) {
        LOGW("[gpu-tune] no NVIDIA GPUs visible to NVML");
        if (Shutdown) Shutdown();
        dlclose(lib);
        return;
    }
    for (unsigned d = 0; d < ngpu; ++d) {
        void* h = nullptr;
        if (Handle(d, &h) != 0 || h == nullptr) continue;
        char nm[96] = {0};
        if (GetName) GetName(h, nm, sizeof(nm));
        const std::string dname = "GPU" + std::to_string(d) + " (" + nm + ")";
        const std::string tag = "[gpu-tune] " + dname;
        GpuTuneState::Dev dev;
        dev.h = h;
        dev.name = dname;
        // board power limit (W) -- set first so the offset has the full budget; clamped to
        // range. Capture the stock DEFAULT limit first so shutdown can restore it.
        if (power_limit_w > 0 && SetPower) {
            if (DefPower) { unsigned dm = 0; if (DefPower(h, &dm) == 0) dev.def_power_mw = dm; }
            unsigned want_mw = static_cast<unsigned>(power_limit_w) * 1000u, lo = 0, hi = 0;
            if (PowerC && PowerC(h, &lo, &hi) == 0) {
                if (want_mw < lo) want_mw = lo;
                if (want_mw > hi) want_mw = hi;
            }
            int sr = SetPower(h, want_mw);
            if (sr == 0) { LOGI(tag << " power limit = " << (want_mw / 1000u) << " W"); g_tune.did_power = true; }
            else LOGW(tag << " set power limit FAILED: " << estr(sr) << " -- run the miner as root");
        }
        // GPC clock V/F offset (MHz) -- clamped to the driver's allowed range
        if (clk_offset != 0 && SetOff) {
            int want = clk_offset, lo = -1000, hi = 1000;
            if (MinMax && MinMax(h, &lo, &hi) == 0) {
                if (want < lo) want = lo;
                if (want > hi) want = hi;
            }
            int sr = SetOff(h, want);
            if (sr == 0) { LOGI(tag << " GPC clock offset = " << want << " MHz"
                              << (want != clk_offset ? (" (clamped, range " + std::to_string(lo) +
                                                        ".." + std::to_string(hi) + ")") : std::string()));
                           g_tune.did_offset = true;
                           verify_offset(h, tag, "GPC clock offset", SetOff, GetOff, want); }
            else LOGW(tag << " set clock offset " << want << " FAILED: " << estr(sr) << " -- run the miner as root");
        }
        // memory (GDDR7) clock V/F offset (MHz) -- clamped to the driver's allowed range. On a
        // power-capped, HBM-traffic-bound digest, faster memory buys more effective bandwidth at the
        // same board watts. CAUTION: a too-high mem offset corrupts digests silently -> pool rejects.
        if (mem_clk_offset != 0 && SetMemOff) {
            int want = mem_clk_offset, lo = -2000, hi = 4000;
            if (MemMinMax && MemMinMax(h, &lo, &hi) == 0) {
                if (want < lo) want = lo;
                if (want > hi) want = hi;
            }
            int sr = SetMemOff(h, want);
            if (sr == 0) { LOGI(tag << " memory clock offset = " << want << " MHz"
                              << (want != mem_clk_offset ? (" (clamped, range " + std::to_string(lo) +
                                                            ".." + std::to_string(hi) + ")") : std::string()));
                           g_tune.did_mem_offset = true;
                           verify_offset(h, tag, "memory clock offset", SetMemOff, GetMemOff, want); }
            else LOGW(tag << " set memory clock offset " << want << " FAILED: " << estr(sr) << " -- run the miner as root");
        }
        // locked core clock (MHz) -- pin min=max for stability/consistency / to cap the boost
        if (lock_mhz > 0 && LockClk) {
            unsigned v = static_cast<unsigned>(lock_mhz);
            int sr = LockClk(h, v, v);
            if (sr == 0) { LOGI(tag << " core clock locked = " << v << " MHz"); g_tune.did_lock = true; }
            else LOGW(tag << " lock core clock " << v << " FAILED: " << estr(sr) << " -- run the miner as root");
        }
        // locked memory clock (MHz) -- the inverse power lever of mem_clk_offset: pinning GDDR LOW
        // frees board power at the cap and the core boost takes it. Snaps to supported points
        // (5090: 14551/7001/810/405). MEASURED DEAD on the 5090 at ndiff~6.5 (2026-07-08): the
        // digest's bandwidth loss costs more than the core boost pays (see miner_config.h). Kept
        // for per-card offline tuning. Guard = batch fill + rej. Overrides mem_clk_offset.
        if (lock_mem_mhz > 0 && LockMemClk) {
            unsigned v = static_cast<unsigned>(lock_mem_mhz);
            int sr = LockMemClk(h, v, v);
            if (sr == 0) { LOGI(tag << " memory clock locked = " << v << " MHz (supported points may snap)"); g_tune.did_mem_lock = true; }
            else LOGW(tag << " lock memory clock " << v << " FAILED: " << estr(sr) << " -- run the miner as root");
        }
        // fan duty (%) -- pin EVERY fan so the thermal margin can't shrink and step the boost
        // down. Reverted to the driver's auto curve on shutdown.
        if (fan_pct > 0 && SetFan) {
            unsigned pct = static_cast<unsigned>(fan_pct > 100 ? 100 : fan_pct);
            unsigned nf = 0;
            dev.nfans = (GetNumFans && GetNumFans(h, &nf) == 0 && nf > 0) ? nf : 1u;
            int ok = 0, fail = 0, lastr = 0;
            for (unsigned f = 0; f < dev.nfans; ++f) {
                int sr = SetFan(h, f, pct);
                if (sr == 0) ++ok; else { ++fail; lastr = sr; }
            }
            if (ok > 0) { LOGI(tag << " fan = " << pct << "% (" << ok << "/" << dev.nfans << " fan(s))"); g_tune.did_fan = true; }
            if (fail > 0) LOGW(tag << " set fan FAILED on " << fail << " fan(s): " << estr(lastr) << " -- run the miner as root");
        }
        g_tune.devs.push_back(std::move(dev));
    }
    // Record the symbols/handles the revert path needs, then arm the shutdown-revert watcher
    // -- but only if something was actually applied. Otherwise tear NVML down as before.
    g_tune.SetOff = SetOff; g_tune.SetMemOff = SetMemOff; g_tune.SetPower = SetPower; g_tune.ResetLock = ResetLock;
    g_tune.ResetMemLock = ResetMemLock; g_tune.SetDefFan = SetDefFan; g_tune.ErrStr = ErrStr;
    if (g_tune.did_offset || g_tune.did_mem_offset || g_tune.did_power || g_tune.did_lock || g_tune.did_mem_lock || g_tune.did_fan) {
        g_tune.lib = lib;        // intentionally kept open for the lifetime of the process
        StartTuneRevertWatcher();
    } else {
        if (Shutdown) Shutdown();
        dlclose(lib);
    }
}
