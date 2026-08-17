// matador-miner: startup auto-update. Checks GitHub for the newest release and,
// if the adopt gate (DecideUpdate, update_gate.h) accepts it, downloads the
// matching asset, verifies it, swaps the binary atomically, and re-execs.
// SECTION of the single miner translation unit (uses Config, mlog, and the
// unit-tested gates in update_gate.h / version_compare.h). Extracted verbatim.
#pragma once

// 6c. Startup auto-update. Checks GitHub for the newest release; if it differs from
//     this build, downloads the platform binary, verifies its sha256, atomically
//     replaces this executable, and re-exec's into it (graceful: happens BEFORE the
//     mine loop, no node restart). --no-auto-update -> notify only; --no-update-check
//     -> skip entirely. A re-exec loop guard (MATADOR_UPDATED_TO) prevents churn if a
//     downloaded binary somehow reports an old version.
// ===========================================================================
static bool HttpGet(const std::string& url, std::string& out, long timeout_s)
{
    CURL* c = curl_easy_init();
    if (c == nullptr) return false;
    struct curl_slist* hdr = nullptr;
    hdr = curl_slist_append(hdr, "User-Agent: matador-miner");
    hdr = curl_slist_append(hdr, "Accept: application/vnd.github+json");
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdr);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout_s);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION,
        +[](char* p, size_t s, size_t n, void* u) -> size_t {
            static_cast<std::string*>(u)->append(p, s * n); return s * n; });
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &out);
    const CURLcode rc = curl_easy_perform(c);
    curl_slist_free_all(hdr);
    curl_easy_cleanup(c);
    return rc == CURLE_OK;
}

static std::string SelfExePath()
{
#if defined(__APPLE__)
    char buf[4096]; uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) != 0) return "";
    char real[4096];
    return realpath(buf, real) ? std::string(real) : std::string(buf);
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "";
    buf[n] = '\0';
    return std::string(buf);
#endif
}

static bool IsMultiGpuChild()
{
    const char* child = std::getenv("MATADOR_MULTI_GPU_CHILD");
    return child != nullptr && child[0] != '\0';
}

static std::string GpuWorkerSuffix(const std::string& device)
{
    std::string out;
    out.reserve(device.size());
    for (unsigned char c : device) {
        if (std::isalnum(c)) out.push_back(static_cast<char>(std::tolower(c)));
        else if (c == '-' || c == '_') out.push_back(static_cast<char>(c));
        else out.push_back('_');
    }
    return out.empty() ? "unknown" : out;
}

static void ApplyGpuDeviceEnvironment(const std::string& device, const char* scope)
{
    if (device.empty()) return;
    setenv("CUDA_VISIBLE_DEVICES", device.c_str(), 1);
    setenv("HIP_VISIBLE_DEVICES", device.c_str(), 1);
    setenv("ROCR_VISIBLE_DEVICES", device.c_str(), 1);
    setenv("GPU_DEVICE_ORDINAL", device.c_str(), 1);
    LOGI("[gpu] " << scope << " device=" << device
         << " (CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES/ROCR_VISIBLE_DEVICES/GPU_DEVICE_ORDINAL)");
}

// Enumerate every GPU device id on this host by asking the vendor tool. NVIDIA via
// nvidia-smi (real indices); AMD via rocm-smi (logical 0..N-1). Returns empty on hosts
// with no NVIDIA/AMD tooling (Apple/Metal, CPU-only), where the miner runs the single
// default device. Best-effort: a parse miss just means the operator pins gpus explicitly.
static std::vector<std::string> EnumerateAllGpus()
{
    std::vector<std::string> out;

    // NVIDIA: one clean index per line.
    if (FILE* f = popen("nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null", "r")) {
        char line[64];
        while (fgets(line, sizeof(line), f)) {
            const std::string s = TrimCopy(line);
            if (!s.empty()) out.push_back(s);
        }
        pclose(f);
    }
    if (!out.empty()) return out;

    // AMD/ROCm: count device rows ("card0,...") and emit logical indices 0..N-1, which is
    // what ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES expect for the per-child scoping.
    if (FILE* f = popen("rocm-smi --showid --csv 2>/dev/null", "r")) {
        int n = 0;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            const std::string s = TrimCopy(line);
            if (s.rfind("card", 0) == 0) ++n;   // device rows start with cardN
        }
        pclose(f);
        for (int i = 0; i < n; ++i) out.push_back(std::to_string(i));
    }
    return out;
}

// When the operator did not pin a device list (default), or asked for `gpus: "auto"`,
// detect every GPU and mine on all of them. Honors an explicit list (opt-out) and never
// fans out for an explicit CPU backend.
static void MaybeAutoDetectGpus(Config& cfg)
{
    if (IsMultiGpuChild()) return;   // a forked child runs its single assigned device

    std::string be = cfg.backend;
    for (auto& c : be) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (be == "cpu") return;

    const bool explicitAuto = (cfg.gpu_devices.size() == 1 && cfg.gpu_devices[0] == "auto");
    const bool unspecified  = cfg.gpu_devices.empty();
    if (!explicitAuto && !unspecified) return;   // operator pinned real ids -> honor them

    const std::vector<std::string> all = EnumerateAllGpus();
    if (all.size() <= 1) {
        // 0 or 1 detected: clear any "auto" sentinel so we run the normal single-device path.
        cfg.gpu_devices.clear();
        if (explicitAuto) {
            LOGW("[multi-gpu] gpus=auto: detected " << all.size()
                 << " GPU(s); running the single default device");
        }
        return;
    }
    cfg.gpu_devices = all;
    std::ostringstream o;
    for (size_t i = 0; i < all.size(); ++i) { if (i) o << ","; o << all[i]; }
    LOGI("[multi-gpu] auto-detected " << all.size() << " GPUs (" << o.str()
         << "); mining on all. Pin a single card with gpus:[0] / --gpus 0 to opt out.");
}

static bool MaybeRunMultiGpuSupervisor(const Config& cfg, int argc, char* argv[], int& exit_code)
{
    if (IsMultiGpuChild() || cfg.gpu_devices.size() <= 1) return false;

    if (cfg.api_enabled && cfg.api_port > 0 && cfg.api_port + static_cast<int>(cfg.gpu_devices.size()) - 1 > 65535) {
        LOGE("[multi-gpu] api_port range would exceed 65535: base=" << cfg.api_port
             << " gpus=" << cfg.gpu_devices.size());
        exit_code = 2;
        return true;
    }

    const std::string exe = SelfExePath();
    if (exe.empty()) {
        LOGE("[multi-gpu] cannot resolve own executable path; run one miner per GPU manually");
        exit_code = 2;
        return true;
    }

    LOGW("[multi-gpu] basic process fan-out enabled gpus=" << cfg.gpu_devices.size()
         << " devices=" << [&]() { std::ostringstream o; for (size_t i = 0; i < cfg.gpu_devices.size(); ++i) { if (i) o << ","; o << cfg.gpu_devices[i]; } return o.str(); }()
         << " worker_suffix=" << (cfg.gpu_worker_suffix ? "per-gpu(-gpuN)" : "shared(single worker)")
         << " note=no cross-GPU batching/autotune; one ordinary miner process per GPU");

    // Supervisor with per-child RESPAWN. The old waitpid loop reaped a dead child
    // ONCE and never re-forked it, so a single CUDA crash (XID, OOM, driver reset)
    // idled that GPU forever while the rest of the rig mined on -- silent lost
    // revenue for weeks on an unattended box. Now an ABNORMAL exit (non-zero code
    // or signal) re-forks the same slot with per-child exponential backoff
    // (5s,10s,20s,40s,60s cap); kMaxRespawns crashes without an intervening
    // healthy run (>= kHealthyRunMs) means the fault is not transient, so the
    // parent stops all children and exits non-zero -> systemd restarts the whole
    // rig with fresh driver state. A CLEAN exit (code 0) is deliberate and stays
    // down, preserving the old "all children gone -> parent returns" behavior.
    struct ChildSlot {
        pid_t pid{-1};
        int restarts{0};
        int backoff_s{5};
        int64_t spawn_ms{0};
    };
    constexpr int kMaxRespawns = 10;
    constexpr int kBackoffCapS = 60;
    constexpr int64_t kHealthyRunMs = 10 * 60 * 1000;   // ran >=10min -> reset the ladder

    auto spawn_child = [&](size_t i) -> pid_t {
        const std::string& device = cfg.gpu_devices[i];
        // One source of truth for this card's stratum worker name, used by both the child's
        // env handoff and the parent's spawn log. --no-gpu-suffix drops the per-card -gpuN so
        // every card authorizes as the bare "<addr>.<worker>" (the pool aggregates them; each
        // child still has its own connection + pool-assigned nonce lane + share credit).
        const std::string child_worker = cfg.gpu_worker_suffix
            ? (cfg.worker + "-gpu" + GpuWorkerSuffix(device))
            : cfg.worker;
        const pid_t pid = fork();
        if (pid < 0) {
            LOGE("[multi-gpu] fork failed index=" << i << " error=" << std::strerror(errno));
            return -1;
        }
        if (pid == 0) {
            ApplyGpuDeviceEnvironment(device, "child");
            setenv("MATADOR_MULTI_GPU_CHILD", "1", 1);
            setenv("MATADOR_MULTI_GPU_CHILD_INDEX", std::to_string(i).c_str(), 1);
            setenv("MATADOR_MULTI_GPU_CHILD_COUNT", std::to_string(cfg.gpu_devices.size()).c_str(), 1);
            setenv("MATADOR_MULTI_GPU_CHILD_DEVICE", device.c_str(), 1);
            setenv("MATADOR_MULTI_GPU_CHILD_WORKER", child_worker.c_str(), 1);
            // Auto-hive: every child reports the SAME operator_label (explicit --operator-label
            // if set, else the base worker) so minebtx groups this multi-GPU rig as one "hive".
            const std::string shared_label = cfg.operator_label.empty() ? cfg.worker : cfg.operator_label;
            setenv("MATADOR_MULTI_GPU_OPERATOR_LABEL", shared_label.c_str(), 1);
            if (cfg.api_enabled && cfg.api_port > 0) {
                const int child_port = cfg.api_port + static_cast<int>(i);
                setenv("MATADOR_MULTI_GPU_CHILD_API_PORT", std::to_string(child_port).c_str(), 1);
            }
            (logtee::ExecRestore(), execv(exe.c_str(), argv));
            std::fprintf(stderr, "[multi-gpu] child exec failed: %s\n", std::strerror(errno));
            _exit(127);
        }
        LOGI("[multi-gpu] spawned index=" << i
             << " pid=" << pid
             << " device=" << device
             << " worker=" << child_worker
             << (cfg.api_enabled && cfg.api_port > 0 ? (" api_port=" + std::to_string(cfg.api_port + static_cast<int>(i))) : ""));
        return pid;
    };

    std::vector<ChildSlot> slots(cfg.gpu_devices.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        const pid_t pid = spawn_child(i);
        if (pid < 0) {                 // initial spawn must fully succeed (old behavior)
            exit_code = 1;
            return true;
        }
        slots[i].pid = pid;
        slots[i].spawn_ms = MonoMs();
    }

    // Rig-level [stats-all] roll-up: children drop heartbeat snapshots (mgpu_stats.h);
    // this thread sums the fresh ones every 60s. RAII -> joined on EVERY return path.
    mgpu::SupervisorStats stats_all(slots.size());

    auto live_count = [&]() {
        size_t n = 0;
        for (const ChildSlot& s : slots) if (s.pid > 0) ++n;
        return n;
    };

    int rc = 0;
    while (live_count() > 0) {
        int status = 0;
        const pid_t pid = waitpid(-1, &status, 0);
        if (pid < 0) {
            if (errno == EINTR) continue;
            LOGE("[multi-gpu] waitpid failed error=" << std::strerror(errno));
            rc = 1;
            break;
        }
        auto it = std::find_if(slots.begin(), slots.end(),
                               [&](const ChildSlot& s) { return s.pid == pid; });
        if (it == slots.end()) continue;   // reaped something not ours (shouldn't happen)
        ChildSlot& slot = *it;
        const size_t idx = static_cast<size_t>(it - slots.begin());
        slot.pid = -1;

        bool abnormal = false;
        if (WIFEXITED(status)) {
            const int code = WEXITSTATUS(status);
            LOGW("[multi-gpu] child exited pid=" << pid << " index=" << idx
                 << " code=" << code << " remaining=" << live_count());
            if (code != 0) { rc = code; abnormal = true; }
        } else if (WIFSIGNALED(status)) {
            const int sig = WTERMSIG(status);
            LOGW("[multi-gpu] child signaled pid=" << pid << " index=" << idx
                 << " signal=" << sig << " remaining=" << live_count());
            rc = 128 + sig;
            abnormal = true;
        }
        if (!abnormal) continue;           // clean exit (code 0) stays down

        // A long healthy run means this crash is fresh, not a crash loop: reset the
        // ladder so a rare weekly CUDA hiccup never marches a rig toward the cap.
        if (MonoMs() - slot.spawn_ms >= kHealthyRunMs) {
            slot.restarts = 0;
            slot.backoff_s = 5;
        }

        if (slot.restarts >= kMaxRespawns) {
            LOGE("[multi-gpu] child index=" << idx << " device=" << cfg.gpu_devices[idx]
                 << " crashed " << slot.restarts << "x without a healthy run (cap=" << kMaxRespawns
                 << "); stopping all children and exiting non-zero so systemd restarts the whole rig");
            // SIGTERM the survivors (they revert GPU tuning on the way out), escalate
            // to SIGKILL after 10s, and REAP them all -- an orphan left holding a GPU
            // would fight the systemd-restarted supervisor for the device.
            for (ChildSlot& s : slots) if (s.pid > 0) ::kill(s.pid, SIGTERM);
            const int64_t term_t0 = MonoMs();
            bool killed = false;
            while (live_count() > 0) {
                int st = 0;
                const pid_t p = waitpid(-1, &st, WNOHANG);
                if (p > 0) {
                    for (ChildSlot& s : slots) if (s.pid == p) s.pid = -1;
                    continue;
                }
                if (p < 0 && errno != EINTR) break;   // ECHILD: nothing left to reap
                if (!killed && MonoMs() - term_t0 > 10000) {
                    killed = true;
                    LOGW("[multi-gpu] children still up 10s after SIGTERM; sending SIGKILL");
                    for (ChildSlot& s : slots) if (s.pid > 0) ::kill(s.pid, SIGKILL);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            exit_code = (rc != 0) ? rc : 1;
            return true;
        }

        ++slot.restarts;
        LOGW("[multi-gpu] respawning child index=" << idx << " device=" << cfg.gpu_devices[idx]
             << " in " << slot.backoff_s << "s (restart " << slot.restarts << "/" << kMaxRespawns
             << "; that GPU idles until then)");
        // Sleeping here delays only THIS respawn; other exits queue as zombies and
        // are reaped on the next loop pass (bounded by the 60s backoff cap).
        std::this_thread::sleep_for(std::chrono::seconds(slot.backoff_s));
        slot.backoff_s = std::min(slot.backoff_s * 2, kBackoffCapS);
        const pid_t np = spawn_child(idx);
        if (np < 0) {
            // fork failed (fd/memory pressure): leave the slot down; the cap/backoff
            // machinery handles a persistent failure on the next crash cycle.
            LOGE("[multi-gpu] respawn fork failed index=" << idx << " (slot stays down)");
            rc = 1;
        } else {
            slot.pid = np;
            slot.spawn_ms = MonoMs();
        }
    }
    exit_code = rc;
    return true;
}

static const char* PlatformAssetSuffix()
{
#if defined(__APPLE__)
    return "macos-arm64";
#else
    return "linux-x86_64";
#endif
}

struct ReleaseInfo {
    bool ok{false};
    std::string tag;
    std::string asset_url;     // raw platform binary (NOT the -bundle.tar.gz)
    std::string sha_url;       // its .sha256 sidecar
    std::string html;          // human release page
    std::string published_at;  // ISO8601 (for bake-time)
};

// Fetch the newest release for the configured channel. stable -> /releases/latest
// (GitHub's "Latest", always non-prerelease). prerelease -> /releases?per_page=1
// (newest by creation, INCLUDING prereleases). Default-on auto-update must not push
// prereleases to a whole fleet, so "stable" is the default.
// Release API base. Overridable (MATADOR_RELEASE_API_BASE) so an integration test can
// point the update check at a local mock GitHub without touching real releases.
static std::string ReleaseApiBase()
{
    if (const char* b = std::getenv("MATADOR_RELEASE_API_BASE"))
        if (b[0] != '\0') return b;
    return "https://api.github.com/repos/vanities/matador-miner";
}

static ReleaseInfo FetchLatestRelease(const Config& cfg)
{
    const bool want_prerelease = (ToLowerCopy(cfg.update_channel) == "prerelease");
    const std::string base = ReleaseApiBase();
    // Adopt the newest release that actually carries THIS platform's binary. We fetch the
    // release LIST (newest-first) and skip any release that has no asset for us, instead of
    // pinning to the single newest tag. Reason: a partial cut (a macOS-only or CUDA-only
    // release) must NOT make the other platforms report a phantom "update available" they
    // cannot apply, nor strand them off an older release that does have their binary. This
    // mirrors install.sh's per-platform resolution. With the "release ALL" policy the newest
    // release is almost always the answer, but this also covers the in-flight window while a
    // release's assets are still uploading.
    const std::string url = base + "/releases?per_page=30";
    ReleaseInfo ri;
    std::string body;
    if (!HttpGet(url, body, 8)) { LOGD("[update] check skipped (network)"); return ri; }
    try {
        UniValue v;
        if (!v.read(body) || !v.isArray()) return ri;
        for (const auto& rel : v.getValues()) {            // newest-first
            if (!rel.exists("tag_name")) continue;
            const bool is_draft = rel.exists("draft") && rel["draft"].isBool() && rel["draft"].get_bool();
            const bool is_pre   = rel.exists("prerelease") && rel["prerelease"].isBool() && rel["prerelease"].get_bool();
            if (is_draft) continue;
            if (is_pre && !want_prerelease) continue;      // the stable channel ignores prereleases
            std::string asset_url, sha_url;
            if (rel.exists("assets") && rel["assets"].isArray()) {
                for (const auto& a : rel["assets"].getValues()) {
                    const std::string n = a.exists("name") ? a["name"].get_str() : "";
                    const std::string u = a.exists("browser_download_url") ? a["browser_download_url"].get_str() : "";
                    if (n.find(PlatformAssetSuffix()) == std::string::npos) continue;
                    const bool is_sha = n.size() > 7 && n.compare(n.size() - 7, 7, ".sha256") == 0;
                    const bool is_bundle = n.find("-bundle.tar.gz") != std::string::npos;
                    // Never adopt a "-legacy-" asset. The pre-Ampere lane was retired and its
                    // binaries mine the withdrawn v3 algorithm, so an old release still carrying
                    // one must not be picked up by a running miner.
                    if (n.find("-legacy-") != std::string::npos) continue;
                    // Auto-update replaces this executable in place, so it must pick the
                    // raw binary asset, not the install bundle tarball published beside it.
                    if (is_bundle) continue;
                    if (is_sha) sha_url = u;
                    else        asset_url = u;
                }
            }
            if (asset_url.empty()) continue;               // no binary for us here; try the next older release
            ri.tag = rel["tag_name"].get_str();
            ri.html = rel.exists("html_url") ? rel["html_url"].get_str()
                                             : "https://github.com/vanities/matador-miner/releases";
            if (rel.exists("published_at") && rel["published_at"].isStr())
                ri.published_at = rel["published_at"].get_str();
            ri.asset_url = asset_url;
            ri.sha_url = sha_url;
            ri.ok = true;
            return ri;                                     // newest release carrying our asset wins
        }
        // No release in the list carries our asset: leave ri.ok = false (no update info).
    } catch (...) { LOGD("[update] parse failed"); return ReleaseInfo{}; }
    return ri;
}

// Age in seconds of an ISO8601 "2026-06-17T12:34:56Z" timestamp (UTC). Returns -1 if
// it cannot be parsed, so callers treat "unknown age" as "do not block on bake-time".
static int64_t ReleaseAgeSeconds(const std::string& iso8601)
{
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
    if (std::sscanf(iso8601.c_str(), "%d-%d-%dT%d:%d:%dZ", &y, &mo, &d, &h, &mi, &s) != 6) return -1;
    struct tm tm{};
    tm.tm_year = y - 1900; tm.tm_mon = mo - 1; tm.tm_mday = d;
    tm.tm_hour = h; tm.tm_min = mi; tm.tm_sec = s;
    const time_t rel = timegm(&tm);
    if (rel == static_cast<time_t>(-1)) return -1;
    return static_cast<int64_t>(std::time(nullptr) - rel);
}

// Core check used by BOTH the startup tick and the periodic thread. When auto-update
// is enabled and a newer release passes the bake-time gate, it downloads the platform
// binary, sha256-verifies it, atomically swaps this executable, and re-exec's into it
// (no node restart). On a non-main thread the execv simply replaces the whole process.
static void RunUpdateCheck(const Config& cfg, char** argv, UpdateState& ust)
{
    if (!cfg.update_check) return;
    const std::string cur = MATADOR_MINER_VERSION;
    const ReleaseInfo ri = FetchLatestRelease(cfg);
    {
        std::lock_guard<std::mutex> lk(ust.mu);
        ust.last_check_ms = MonoMs();
        if (ri.ok && !ri.tag.empty()) ust.latest_seen = ri.tag;
    }
    if (!ri.ok) return;

    // Compute the bake-time age (only meaningful when min_version_age_s > 0) and read the re-exec
    // loop-guard env up front, then let the pure, unit-tested gate (update_gate.h) decide. Keeping
    // the whole adopt decision in one function is what stops the v0.6.10 lexical-downgrade class of
    // bug from creeping back: the guard ORDER below mirrors DecideUpdate() exactly.
    const int64_t cand_age = (cfg.min_version_age_s > 0) ? ReleaseAgeSeconds(ri.published_at) : -1;
    const char* upd = std::getenv("MATADOR_UPDATED_TO");
    const UpdateDecision dec = DecideUpdate(ri.tag, cur, cfg.auto_update, !ri.asset_url.empty(),
                                            cand_age, cfg.min_version_age_s, upd);

    if (dec == UpdateDecision::UpToDate) {
        LOGI("[update] up to date (" << cur << ", channel=" << cfg.update_channel << ")");
        return;
    }
    LOGW("[update] newer release available: " << ri.tag << " (running " << cur
         << ", channel=" << cfg.update_channel << ")");
    switch (dec) {
        case UpdateDecision::AutoUpdateOff:
            LOGW("[update]   " << ri.html << "   (auto-update off: --no-auto-update)"); return;
        case UpdateDecision::NoAsset:
            LOGW("[update]   no " << PlatformAssetSuffix() << " asset; update manually: " << ri.html); return;
        case UpdateDecision::Baking:
            LOGI("[update] holding " << ri.tag << " for bake-time (age=" << cand_age
                 << "s < min_version_age_s=" << cfg.min_version_age_s << "); adopting on a later check"); return;
        case UpdateDecision::StampMismatch:
            LOGW("[update] already updated to " << ri.tag << " but still report " << cur
                 << " (version-stamp mismatch); not re-updating"); return;
        case UpdateDecision::Adopt: break;
        case UpdateDecision::UpToDate: return;  // handled above; listed to keep the switch exhaustive
    }

    LOGI("[update] auto-updating " << cur << " -> " << ri.tag << " ...");
    std::string bin;
    if (!HttpGet(ri.asset_url, bin, 180) || bin.size() < 100000) {
        LOGW("[update] download failed; staying on " << cur); return;
    }
    const std::string exe = SelfExePath();
    if (exe.empty()) { LOGW("[update] cannot resolve own path; staying on " << cur); return; }
    const std::string tmp = exe + ".new";
    { std::ofstream of(tmp, std::ios::binary | std::ios::trunc);
      if (!of) { LOGW("[update] cannot write " << tmp << " (permission?); update via install.sh"); return; }
      of.write(bin.data(), static_cast<std::streamsize>(bin.size())); }

    // sha256-verify the downloaded file before swapping (sha256sum on Linux, shasum
    // on macOS). Required: never run an unverified auto-downloaded binary -- and
    // that means FAIL CLOSED. The old flow only aborted on an outright MISMATCH:
    // a missing .sha256 asset, a failed sidecar fetch, or an unparseable hash all
    // quietly waived verification and installed anyway, i.e. verification was
    // skipped exactly when a release was most suspect (partial upload, MITM'd CDN,
    // tampered asset list). Now every non-verified outcome skips this release and
    // stays on the running version; a later periodic check retries once the
    // sidecar is fetchable.
    {
        if (ri.sha_url.empty()) {
            LOGW("[update] no .sha256 asset published for " << ri.tag
                 << "; refusing to install an unverified binary (skipping this release)");
            std::remove(tmp.c_str());
            return;
        }
        std::string shafile;
        if (!HttpGet(ri.sha_url, shafile, 20)) {
            LOGW("[update] sha256 sidecar download failed for " << ri.tag
                 << "; refusing to install an unverified binary (will retry on a later check)");
            std::remove(tmp.c_str());
            return;
        }
        std::istringstream ss(shafile); std::string expect; ss >> expect;
        std::string got;
        const std::string cmd = "sha256sum '" + tmp + "' 2>/dev/null || shasum -a 256 '" + tmp + "' 2>/dev/null";
        if (FILE* f = popen(cmd.c_str(), "r")) {
            char line[256];
            if (std::fgets(line, sizeof(line), f)) { std::istringstream ls(line); ls >> got; }
            pclose(f);
        }
        // A sha256 is exactly 64 hex chars; anything else means the sidecar parse or
        // the local hash tool failed -> that is NOT "verified", so do not install.
        expect = ToLowerCopy(expect);
        got = ToLowerCopy(got);
        if (expect.size() != 64 || got.size() != 64) {
            LOGW("[update] sha256 verification unavailable (sidecar_len=" << expect.size()
                 << " local_len=" << got.size() << "); refusing to install an unverified binary");
            std::remove(tmp.c_str());
            return;
        }
        if (expect != got) {
            LOGE("[update] sha256 MISMATCH (want " << expect.substr(0, 12) << " got " << got.substr(0, 12)
                 << ") - ABORT, staying on " << cur);
            std::remove(tmp.c_str());
            return;
        }
        LOGI("[update] sha256 verified");
    }

    chmod(tmp.c_str(), 0755);
    if (std::rename(tmp.c_str(), exe.c_str()) != 0) {
        LOGW("[update] cannot replace binary at " << exe << " (permission?); "
             << "ensure the service user owns the install dir (e.g. ReadWritePaths=), or update via install.sh");
        std::remove(tmp.c_str());
        return;
    }
    setenv("MATADOR_UPDATED_TO", ri.tag.c_str(), 1);
    LOGI("[update] installed " << ri.tag << "; re-exec'ing into the new binary (same PID, no node restart) ...");
    (logtee::ExecRestore(), execv(exe.c_str(), argv));
    LOGE("[update] execv failed (" << std::strerror(errno) << "); staying on the running " << cur);
}

// Fast startup tick: check once before the expensive backend init / mine loop so a
// freshly installed rig jumps to the right version immediately (no jitter on startup).
static void DoStartupUpdate(const Config& cfg, char** argv, UpdateState& ust)
{
    RunUpdateCheck(cfg, argv, ust);
}

// Periodic re-check so a rig that stays up for weeks still adopts new releases without
// a manual restart. Sleeps update_interval_s (+ 0..update_jitter_s to de-sync a fleet)
// between checks. update_interval_s <= 0 keeps the legacy startup-only behavior.
static std::thread StartUpdateChecker(const Config& cfg, char** argv,
                                      UpdateState& ust, std::atomic<bool>& stop_all)
{
    return std::thread([&cfg, argv, &ust, &stop_all]() {
        if (!cfg.update_check || cfg.update_interval_s <= 0) {
            LOGI("[update] periodic checks disabled (startup-only)");
            return;
        }
        LOGI("[update] periodic checker enabled interval_s=" << cfg.update_interval_s
             << " jitter_s=" << cfg.update_jitter_s
             << " channel=" << cfg.update_channel
             << " auto_update=" << (cfg.auto_update ? "on" : "off")
             << " min_version_age_s=" << cfg.min_version_age_s);
        std::mt19937 rng(static_cast<uint32_t>(MonoMs()) ^ static_cast<uint32_t>(::getpid()));
        while (!stop_all.load()) {
            int wait_s = std::max(1, cfg.update_interval_s);
            if (cfg.update_jitter_s > 0) {
                std::uniform_int_distribution<int> jitter(0, cfg.update_jitter_s);
                wait_s += jitter(rng);
            }
            for (int i = 0; i < wait_s * 2 && !stop_all.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (stop_all.load()) break;
            RunUpdateCheck(cfg, argv, ust);   // may execv into the new binary (never returns)
        }
        LOGI("[update] periodic checker stopped");
    });
}

// ===========================================================================
