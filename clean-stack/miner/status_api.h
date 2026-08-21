// matador-miner: local read-only HTTP status API (default-off, 127.0.0.1) plus
// the GPU runtime-metric / thermal JSON builders and the pool watchdog thread.
// Exposes counters and public topology only -- never RPC/pool secrets.
// SECTION of the single miner translation unit (uses Config, Stats, mlog, and
// the live pool-telemetry atomics); #included at the point those are in scope.
// Extracted verbatim.
// (Exception: the pure GPU-metric/JSON helpers ARE standalone -- fenced by
// MATADOR_STATUS_API_GPU_JSON_ONLY, unit-tested in
// harness/status_api_gpu_json_test.cpp.)
#pragma once

#include "gpu_telemetry.h"   // GpuTelemetry/QueryGpuTelemetry: fork-free NVML numerics
                             // (a no-op in the miner TU, which includes it far earlier)

// std deps of the standalone-testable helper block below (all long since in the
// miner TU; spelled out so the harness test can compile just this header).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// 6b. Local read-only HTTP status API. Default-off, bound to 127.0.0.1 unless
//     configured otherwise. It exposes counters and public topology only; never
//     RPC cookies/passwords or pool passwords.
// ===========================================================================
static std::string JsonEscape(const std::string& in)
{
    std::ostringstream o;
    for (unsigned char c : in) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b"; break;
            case '\f': o << "\\f"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) {
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                      << std::dec << std::setfill(' ');
                } else {
                    o << static_cast<char>(c);
                }
        }
    }
    return o.str();
}

static std::string JsonString(const std::string& s)
{
    return "\"" + JsonEscape(s) + "\"";
}

static std::vector<std::string> SplitCsvTrim(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ls(line);
    std::string tok;
    while (std::getline(ls, tok, ',')) {
        const size_t a = tok.find_first_not_of(" \t\r\n");
        const size_t b = tok.find_last_not_of(" \t\r\n");
        out.push_back(a == std::string::npos ? std::string() : tok.substr(a, b - a + 1));
    }
    return out;
}

static bool ParseDoubleFinite(const std::string& value, double& out)
{
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || !std::isfinite(parsed)) return false;
    while (*end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) return false;
        ++end;
    }
    out = parsed;
    return true;
}

struct GpuRuntimeMetric {
    std::string uuid;
    std::string vendor;
    double util_pct{0.0};
    double power_w{0.0};
    double temp_c{0.0};
    bool has_util{false};
    bool has_power{false};
    bool has_temp{false};
};

static bool ContainsLower(std::string value, const std::string& needle)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value.find(needle) != std::string::npos;
}

// Map one fork-free NVML sample (gpu_telemetry.h) onto the /summary metric row.
// Field map vs the nvidia-smi columns this row used to be parsed from:
//   util_pct <- nvmlDeviceGetUtilizationRates .gpu    == utilization.gpu
//   power_w  <- nvmlDeviceGetPowerUsage (mW -> W)     == power.draw
//   temp_c   <- nvmlDeviceGetTemperature (GPU sensor) == temperature.gpu
// GpuTelemetry also carries sm/mem clocks + fan, but /summary never exposed
// those and the hub/matlog parsers pin the JSON shape -- so they are not added.
// GpuTelemetry has no per-field validity (a failed getter just leaves 0), so
// mirror nvidia-smi's "[N/A]" -> null semantics where 0 is impossible on live
// silicon: power_w/temp_c 0 -> null (boards whose sensors NVML cannot read are
// the same ones nvidia-smi prints [N/A] for). util_pct keeps 0 as a REAL
// number: an idle or gate-paused GPU genuinely reads 0%.
static GpuRuntimeMetric MetricFromNvmlTelemetry(const GpuTelemetry& t, const std::string& uuid)
{
    GpuRuntimeMetric m;
    m.uuid = uuid;
    m.vendor = "nvidia";
    m.util_pct = static_cast<double>(t.util_pct);
    m.power_w = static_cast<double>(t.pow_w);
    m.temp_c = static_cast<double>(t.temp_c);
    m.has_util = t.ok;
    m.has_power = t.ok && t.pow_w > 0;
    m.has_temp = t.ok && t.temp_c > 0;
    return m;
}

// Everything below needs the miner TU (Timer, mlog, Config, Stats, popen);
// fenced off so harness/status_api_gpu_json_test.cpp can compile just the pure
// helpers above (same idiom as MATADOR_CONFIG_PARSE_HELPERS_ONLY).
#ifndef MATADOR_STATUS_API_GPU_JSON_ONLY

// -- static NVIDIA identity (uuid) cache ------------------------------------
// gpu_uuid is the one /summary field the NVML fast path cannot supply (the
// dlopen'd handle in gpu_telemetry.h exposes numeric getters only) -- and a
// GPU's uuid cannot change mid-run. So it is latched from the FIRST successful
// nvidia-smi read and reused forever; polls after that never fork for it.
// Mutex-guarded: the API thread (/summary) and the watchdog thread poll
// concurrently.
struct NvidiaGpuIdentity {
    std::mutex mu;
    bool latched{false};
    std::vector<std::string> uuids;   // one per GPU, nvidia-smi row order
};

static NvidiaGpuIdentity& NvidiaGpuIdentityCache()
{
    static NvidiaGpuIdentity c;
    return c;
}

static void LatchNvidiaGpuUuids(const std::vector<GpuRuntimeMetric>& rows)
{
    if (rows.empty()) return;   // never latch a failed read; the next poll retries
    auto& c = NvidiaGpuIdentityCache();
    std::lock_guard<std::mutex> lk(c.mu);
    if (c.latched) return;
    for (const auto& m : rows) c.uuids.push_back(m.uuid);
    c.latched = true;
    LOGD("[api] nvidia identity latched gpus=" << c.uuids.size()
         << " (static; later polls stop forking nvidia-smi for uuid)");
}

static std::vector<GpuRuntimeMetric> QueryNvidiaGpuRuntimeMetrics()
{
    Timer sp;
    std::vector<GpuRuntimeMetric> out;
    FILE* f = popen("nvidia-smi --query-gpu=uuid,utilization.gpu,power.draw,temperature.gpu "
                    "--format=csv,noheader,nounits 2>/dev/null", "r");
    if (f == nullptr) {
        LOGD("[api] gpu telemetry unavailable: nvidia-smi popen failed in " << sp.ms() << "ms");
        return out;
    }
    char line[512];
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        const std::vector<std::string> col = SplitCsvTrim(line);
        if (col.size() < 4) continue;
        GpuRuntimeMetric m;
        m.uuid = col[0];
        m.vendor = "nvidia";
        m.has_util = ParseDoubleFinite(col[1], m.util_pct);
        m.has_power = ParseDoubleFinite(col[2], m.power_w);
        m.has_temp = ParseDoubleFinite(col[3], m.temp_c);
        out.push_back(m);
    }
    const int rc = pclose(f);
    LOGD("[api] gpu telemetry rows=" << out.size() << " rc=" << rc << " in " << sp.ms() << "ms");
    LatchNvidiaGpuUuids(out);   // a successful forked read doubles as the one-time identity latch
    return out;
}

static std::vector<GpuRuntimeMetric> QueryAmdGpuRuntimeMetrics()
{
    Timer sp;
    std::vector<GpuRuntimeMetric> out;
    FILE* f = popen("rocm-smi --showuniqueid --showuse --showpower --showtemp --csv 2>/dev/null", "r");
    if (f == nullptr) {
        LOGD("[api] amd gpu telemetry unavailable: rocm-smi popen failed in " << sp.ms() << "ms");
        return out;
    }

    char line[1024];
    std::vector<std::string> header;
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        const std::vector<std::string> col = SplitCsvTrim(line);
        if (col.empty()) continue;
        if (header.empty()) {
            header = col;
            continue;
        }

        GpuRuntimeMetric m;
        m.vendor = "amd";
        for (size_t i = 0; i < col.size() && i < header.size(); ++i) {
            const std::string& key = header[i];
            const std::string& value = col[i];
            if (m.uuid.empty() &&
                (ContainsLower(key, "unique") || ContainsLower(key, "guid") || ContainsLower(key, "id"))) {
                m.uuid = value.empty() ? "AMD-GPU" : ("AMD-" + value);
                continue;
            }
            if (!m.has_util && (ContainsLower(key, "use") || ContainsLower(key, "util"))) {
                m.has_util = ParseDoubleFinite(value, m.util_pct);
                continue;
            }
            if (!m.has_power && ContainsLower(key, "power")) {
                m.has_power = ParseDoubleFinite(value, m.power_w);
                continue;
            }
            if (!m.has_temp && ContainsLower(key, "temp")) {
                m.has_temp = ParseDoubleFinite(value, m.temp_c);
                continue;
            }
        }
        if (m.uuid.empty()) {
            m.uuid = "AMD-GPU-" + std::to_string(out.size());
        }
        if (m.has_util || m.has_power || m.has_temp) {
            out.push_back(m);
        }
    }
    const int rc = pclose(f);
    LOGD("[api] amd gpu telemetry rows=" << out.size() << " rc=" << rc << " in " << sp.ms() << "ms");
    return out;
}

// Fork-free NVML fast path. popen("nvidia-smi") fork()s the miner -- a process
// mapping a multi-GiB CUDA address space -- once per /summary request and every
// watchdog thermal tick, just to read three counters. Even with copy-on-write,
// that fork+exec costs tens of ms of page-table churn next to the live CUDA
// contexts; the dlopen'd NVML getters in gpu_telemetry.h answer the same
// numbers in microseconds, in-process. Returns empty (the caller then falls
// back to the forked path) when NVML is unavailable (no driver / non-NVIDIA
// box) OR the box is not single-GPU: the shared NVML handle is device-0-only,
// and serving just row 0 would silently drop GPUs 1..N-1 from /summary and the
// thermal watch.
static std::vector<GpuRuntimeMetric> QueryNvmlGpuRuntimeMetrics()
{
    const GpuTelemetry t = QueryGpuTelemetry();   // microseconds; ok=false when NVML/device absent
    if (!t.ok) return {};
    std::string uuid;
    {
        auto& c = NvidiaGpuIdentityCache();
        std::lock_guard<std::mutex> lk(c.mu);
        // Not latched yet (first poll) or multi-GPU: let nvidia-smi answer --
        // its successful read is exactly what latches the identity.
        if (!c.latched || c.uuids.size() != 1) return {};
        uuid = c.uuids[0];
    }
    return {MetricFromNvmlTelemetry(t, uuid)};
}

static std::vector<GpuRuntimeMetric> QueryGpuRuntimeMetrics()
{
    // NVML first (no fork; the steady state on an NVIDIA rig), then the
    // nvidia-smi fork (still the first-poll uuid source and the multi-GPU /
    // no-NVML fallback), then rocm-smi. Absent-GPU behavior is unchanged:
    // every source returns empty and the JSON keeps its []/nulls.
    auto metrics = QueryNvmlGpuRuntimeMetrics();
    if (!metrics.empty()) return metrics;
    metrics = QueryNvidiaGpuRuntimeMetrics();
    if (!metrics.empty()) return metrics;
    return QueryAmdGpuRuntimeMetrics();
}

#endif  // MATADOR_STATUS_API_GPU_JSON_ONLY (query paths need the miner TU)

static std::string JsonNumberOrNull(bool has_value, double value)
{
    if (!has_value) return "null";
    std::ostringstream o;
    o << value;
    return o.str();
}

static std::string GpuRuntimeJson(const std::vector<GpuRuntimeMetric>& metrics)
{
    std::ostringstream arr;
    arr << "[";
    for (size_t i = 0; i < metrics.size(); ++i) {
        if (i != 0) arr << ",";
        const auto& m = metrics[i];
        arr << "{\"gpu_uuid\":" << JsonString(m.uuid)
            << ",\"vendor\":" << JsonString(m.vendor)
            << ",\"util_pct\":" << JsonNumberOrNull(m.has_util, m.util_pct)
            << ",\"power_w\":" << JsonNumberOrNull(m.has_power, m.power_w)
            << ",\"temp_c\":" << JsonNumberOrNull(m.has_temp, m.temp_c)
            << "}";
    }
    arr << "]";
    return arr.str();
}

// Throttle reasons + foreign GPU tenants, straight off NVML (fork-free). Both answer
// questions /summary previously could not: "why is this card slow" and "is something
// else on it". A slow rig with throttle=["sw-power-cap"] is at its power limit, not
// broken; one with foreign_procs>0 is sharing the GPU and its ep/s is not comparable.
static std::string GpuHealthJson()
{
    const GpuTelemetry t = QueryGpuTelemetry();
    std::ostringstream o;
    o << "{\"throttle_reasons\":";
    if (t.ok && t.has_throttle) o << JsonString(ThrottleReasonsStr(t.throttle));
    else                        o << "null";
    o << ",\"throttle_bits\":";
    if (t.ok && t.has_throttle) o << t.throttle; else o << "null";
    o << ",\"foreign_procs\":";
    if (t.ok && t.has_procs) o << t.foreign_procs; else o << "null";
    o << ",\"foreign_mib\":";
    if (t.ok && t.has_procs) o << t.foreign_mib; else o << "null";
    o << "}";
    return o.str();
}

#ifndef MATADOR_STATUS_API_GPU_JSON_ONLY

static std::string ThermalStatusJson(const Config& cfg, const std::vector<GpuRuntimeMetric>& metrics)
{
    std::string status = cfg.thermal_enabled ? "ok" : "disabled";
    double max_temp = 0.0;
    double max_power = 0.0;
    bool have_temp = false;
    bool have_power = false;
    std::ostringstream warnings;
    warnings << "[";
    bool first_warning = true;
    auto add_warning = [&](const std::string& warning) {
        if (!first_warning) warnings << ",";
        first_warning = false;
        warnings << JsonString(warning);
    };

    if (cfg.thermal_enabled) {
        if (metrics.empty()) status = "unavailable";
        for (const auto& m : metrics) {
            if (m.has_temp) {
                have_temp = true;
                max_temp = std::max(max_temp, m.temp_c);
                if (cfg.thermal_critical_temp_c > 0 && m.temp_c >= cfg.thermal_critical_temp_c) {
                    status = "critical";
                    std::ostringstream msg;
                    msg << "gpu " << m.uuid << " temp_c=" << m.temp_c
                        << " >= critical_temp_c=" << cfg.thermal_critical_temp_c;
                    add_warning(msg.str());
                } else if (status != "critical" && cfg.thermal_warn_temp_c > 0 && m.temp_c >= cfg.thermal_warn_temp_c) {
                    status = "warning";
                    std::ostringstream msg;
                    msg << "gpu " << m.uuid << " temp_c=" << m.temp_c
                        << " >= warn_temp_c=" << cfg.thermal_warn_temp_c;
                    add_warning(msg.str());
                }
            }
            if (m.has_power) {
                have_power = true;
                max_power = std::max(max_power, m.power_w);
                if (cfg.thermal_warn_power_w > 0.0 && m.power_w >= cfg.thermal_warn_power_w) {
                    if (status == "ok") status = "warning";
                    std::ostringstream msg;
                    msg << "gpu " << m.uuid << " power_w=" << m.power_w
                        << " >= warn_power_w=" << cfg.thermal_warn_power_w;
                    add_warning(msg.str());
                }
            }
        }
    }
    warnings << "]";

    std::ostringstream o;
    o << "{\"enabled\":" << (cfg.thermal_enabled ? "true" : "false")
      << ",\"status\":" << JsonString(status)
      << ",\"warn_temp_c\":" << cfg.thermal_warn_temp_c
      << ",\"critical_temp_c\":" << cfg.thermal_critical_temp_c
      << ",\"warn_power_w\":" << cfg.thermal_warn_power_w
      << ",\"max_temp_c\":" << JsonNumberOrNull(have_temp, max_temp)
      << ",\"max_power_w\":" << JsonNumberOrNull(have_power, max_power)
      << ",\"warnings\":" << warnings.str()
      << "}";
    return o.str();
}

static std::vector<PoolEndpoint> EffectivePools(const Config& cfg)
{
    std::vector<PoolEndpoint> pools = cfg.pools;
    if (pools.empty() && !cfg.pool_host.empty() && cfg.pool_port > 0) {
        pools.push_back(PoolEndpoint{cfg.pool_host, cfg.pool_port, "primary"});
    }
    return pools;
}

static std::string BuildPoolsJson(const Config& cfg)
{
    const auto pools = EffectivePools(cfg);
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < pools.size(); ++i) {
        if (i != 0) o << ",";
        o << "{\"index\":" << i
          << ",\"host\":" << JsonString(pools[i].host)
          << ",\"port\":" << pools[i].port
          << ",\"label\":" << JsonString(pools[i].label)
          << "}";
    }
    o << "]";
    return o.str();
}

static std::string BuildSummaryJson(const Config& cfg,
                                    const Stats& stats,
                                    const UpdateState& ust,
                                    const std::string& mode,
                                    const std::string& backend)
{
    const auto now = std::chrono::steady_clock::now();
    const double up_s = std::chrono::duration<double>(now - stats.started).count();
    const int64_t now_ms = MonoMs();
    auto age_s = [&](int64_t ts) -> int64_t {
        return ts > 0 ? std::max<int64_t>(0, (now_ms - ts) / 1000) : -1;
    };
    std::string wd_status, wd_warning, wd_action;
    {
        std::lock_guard<std::mutex> lk(stats.watchdog_mu);
        wd_status = stats.watchdog_status;
        wd_warning = stats.watchdog_last_warning;
        wd_action = stats.watchdog_last_action;
    }
    std::string upd_current, upd_latest, upd_channel;
    int64_t upd_last_ms; bool upd_auto;
    {
        std::lock_guard<std::mutex> lk(ust.mu);
        upd_current = ust.current; upd_latest = ust.latest_seen;
        upd_channel = ust.channel; upd_last_ms = ust.last_check_ms; upd_auto = ust.auto_update;
    }
    const auto p = ProbeMatMulSolvePipelineStats();
    // NVML-first, fork-free in steady state (QueryNvmlGpuRuntimeMetrics): this
    // used to fork the whole CUDA-mapped miner through popen(nvidia-smi) on
    // EVERY /summary request.
    const auto gpu_metrics = QueryGpuRuntimeMetrics();
    auto jnum = [](double x) -> std::string {
        std::ostringstream s; s.setf(std::ios::fixed); s.precision(1); s << x; return s.str();
    };
    // full-precision variant: pool share difficulty is ~1e-5, which jnum's fixed .1 rounds to 0.0
    auto jnumg = [](double x) -> std::string {
        std::ostringstream s; s << std::setprecision(9) << x; return s.str();
    };
    // Pool credit ratio: episodes the POOL credited (accepted shares x expected episodes per
    // share) over episodes this rig PRODUCED, same window. ~1.0 = the pool pays for the work;
    // persistently below ~0.9 over 24h WITH a real share count (see acc_per_hr) = work is not
    // being credited -- stale-heavy connection, mis-set difficulty, or a short-paying pool.
    // Share luck is Poisson, so the 1h figure swings with few shares; trust 24h first. The
    // ~1% dev-fee window mines on a separate session, so a fraction of produced episodes is
    // credited there, not here: ~0.99 is the honest healthy baseline, not exactly 1.0.
    // Solo mode has no pool credit: the ratio serves 0 there, meaning "not applicable".
    auto credit_ratio = [](const Stats::WindowAvg& w) -> double {
        const double eps = w.episode_per_s.load(std::memory_order_relaxed);
        const double pool = w.pool_episode_per_s.load(std::memory_order_relaxed);
        return eps > 0.0 ? pool / eps : 0.0;
    };
    std::ostringstream o;
    o << "{"
      << "\"status\":\"ok\""
      << ",\"version\":" << JsonString(MATADOR_MINER_VERSION)
      << ",\"mode\":" << JsonString(mode)
      << ",\"backend\":" << JsonString(backend)
      << ",\"mining_state\":" << JsonString(MiningStateStr())
      << ",\"gate_reason\":" << JsonString(g_gate_enabled.load() ? GateReason() : "")
      << ",\"uptime_sec\":" << static_cast<uint64_t>(up_s)
      << ",\"worker\":" << JsonString(cfg.worker)
      << ",\"operator_label\":" << JsonString(cfg.operator_label.empty() ? cfg.worker : cfg.operator_label)
      << ",\"chain\":" << JsonString(cfg.chain)
      << ",\"payoutaddress\":" << JsonString(cfg.payoutaddress)
      << ",\"shares\":{"
      << "\"accepted\":" << stats.accepted.load()
      << ",\"rejected\":" << stats.rejected.load()
      << ",\"stale\":" << stats.stale.load()
      << ",\"dev\":" << stats.dev_shares.load()
      << "}"
      << ",\"nonces\":{"
      << "\"total\":" << stats.total_nonces.load()
      // ENC_RC proof of life and the miner's real unit of work. The v3 pipeline counters
      // that used to sit here (batched attempts/digest requests/batch size/pre-hash scanned)
      // described a pipeline that no longer exists; monitoring that alerted on "nonces
      // stopped" would page on a working rig. These are the fields that actually move.
      << ",\"rc_episodes\":" << p.rc_episodes
      << ",\"solve_windows\":" << p.solve_windows
      << ",\"rc_active\":"
      << (Params().GetConsensus().IsMatMulRCActive(
              static_cast<int32_t>(g_pool_block_height.load(std::memory_order_relaxed)))
              ? "true" : "false")
      << "}"
      // Live rate over the last heartbeat interval. LEAD WITH THIS for anything that
      // displays or tunes: the windowed averages below are the sustained-throughput
      // numbers, but a 1h window barely moves within minutes of a clock change, and on
      // the solo path it has no history behind it at all. window_sec 0 = not real yet.
      << ",\"rate\":{"
      << "\"episode_per_s\":" << jnumg(stats.rate_episode_per_s.load(std::memory_order_relaxed))
      << ",\"window_sec\":" << jnum(stats.rate_window_sec.load(std::memory_order_relaxed))
      // High-water mark of the live rate (full windows only) -- the rig's own demonstrated
      // capability, never decayed. rate/peak << 1 sustained = degraded (throttle, foreign
      // tenant, sick card); the watchdog's degradation rule uses exactly this comparison.
      << ",\"peak_episode_per_s\":" << jnumg(stats.rate_peak_episode_per_s.load(std::memory_order_relaxed))
      << "}"
      << ",\"averages\":{"
      << "\"5m\":{"
      << "\"episode_per_s\":" << jnumg(stats.avg_5m.episode_per_s.load(std::memory_order_relaxed))
      << ",\"pool_episode_per_s\":" << jnumg(stats.avg_5m.pool_episode_per_s.load(std::memory_order_relaxed))
      << ",\"acc_per_hr\":" << jnum(stats.avg_5m.acc_per_hr.load(std::memory_order_relaxed))
      << "},\"1h\":{"
      << "\"episode_per_s\":" << jnumg(stats.avg_1h.episode_per_s.load(std::memory_order_relaxed))
      << ",\"pool_episode_per_s\":" << jnumg(stats.avg_1h.pool_episode_per_s.load(std::memory_order_relaxed))
      << ",\"acc_per_hr\":" << jnum(stats.avg_1h.acc_per_hr.load(std::memory_order_relaxed))
      << ",\"pool_credit_ratio\":" << jnumg(credit_ratio(stats.avg_1h))
      << "},\"24h\":{"
      << "\"episode_per_s\":" << jnumg(stats.avg_24h.episode_per_s.load(std::memory_order_relaxed))
      << ",\"pool_episode_per_s\":" << jnumg(stats.avg_24h.pool_episode_per_s.load(std::memory_order_relaxed))
      << ",\"acc_per_hr\":" << jnum(stats.avg_24h.acc_per_hr.load(std::memory_order_relaxed))
      << ",\"pool_credit_ratio\":" << jnumg(credit_ratio(stats.avg_24h))
      << "}}"
      // The v3 "validation" object (freivalds/phase2/transcript check counters) is
      // gone with the verify paths that fed it. ENC_RC needs no separate validation
      // stage: the episode digest IS the proof, so what a consumer wants here is
      // episode throughput, which lives under "solver" above.
      << ",\"solver_path\":\"enc-rc\""
      << ",\"watchdog\":{"
      << "\"status\":" << JsonString(wd_status)
      << ",\"last_warning\":" << JsonString(wd_warning)
      << ",\"last_action\":" << JsonString(wd_action)
      << ",\"reject_streak\":" << stats.reject_streak.load()
      << ",\"last_notify_age_sec\":" << age_s(stats.last_notify_ms.load())
      << ",\"last_share_age_sec\":" << age_s(stats.last_share_ms.load())
      << ",\"last_accept_age_sec\":" << age_s(stats.last_accept_ms.load())
      << ",\"last_nonce_age_sec\":" << age_s(stats.last_nonce_ms.load())
      << ",\"reconnect_requested\":" << (stats.watchdog_reconnect_requested.load() ? "true" : "false")
      << "}"
      // Live pool-link telemetry (stratum path; zeros/-1 in solo mode): connection age,
      // request->response RTT EMA, job staleness, and the difficulty pair the [stats]
      // log line already shows -- so a fleet hub gets them without scraping logs.
      << ",\"pool_link\":{"
      << "\"connected\":" << (g_pool_connected_ms.load(std::memory_order_relaxed) > 0 ? "true" : "false")
      << ",\"conn_age_sec\":" << age_s(g_pool_connected_ms.load(std::memory_order_relaxed))
      << ",\"latency_ms\":" << jnum(g_pool_latency_ms.load(std::memory_order_relaxed))
      << ",\"last_job_age_sec\":" << age_s(stats.last_notify_ms.load())
      << ",\"net_diff\":" << jnumg(g_pool_net_diff.load(std::memory_order_relaxed))
      << ",\"share_diff\":" << jnumg(g_pool_share_diff.load(std::memory_order_relaxed))
      << ",\"block_height\":" << g_pool_block_height.load(std::memory_order_relaxed)
      << "}"
      << ",\"thermal\":" << ThermalStatusJson(cfg, gpu_metrics)
      << ",\"gpu_runtime\":" << GpuRuntimeJson(gpu_metrics)
      // Why the card is at the clock it is, and who else is on it. Deliberately a
      // top-level object rather than extra gpu_runtime fields: that row has an
      // nvidia-smi CSV fallback which cannot supply either value, and its JSON shape
      // is pinned by harness/status_api_gpu_json_test.cpp. One process serves one GPU,
      // so this describes THIS process's card. null = NVML could not answer.
      << ",\"gpu_health\":" << GpuHealthJson()
      << ",\"update\":{"
      << "\"current\":" << JsonString(upd_current)
      << ",\"latest_seen\":" << JsonString(upd_latest)
      << ",\"last_check_age_sec\":" << age_s(upd_last_ms)
      << ",\"channel\":" << JsonString(upd_channel)
      << ",\"auto_update\":" << (upd_auto ? "true" : "false")
      << "}"
      << ",\"pools\":" << BuildPoolsJson(cfg)
      << "}";
    return o.str();
}

static bool SendAll(int fd, const std::string& data)
{
    size_t off = 0;
    while (off < data.size()) {
#ifdef MSG_NOSIGNAL
        constexpr int flags = MSG_NOSIGNAL;
#else
        constexpr int flags = 0;
#endif
        const ssize_t n = ::send(fd, data.data() + off, data.size() - off, flags);
        if (n <= 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

static void SendHttpJson(int fd, int code, const std::string& reason, const std::string& body)
{
    std::ostringstream resp;
    resp << "HTTP/1.1 " << code << " " << reason << "\r\n"
         << "Content-Type: application/json\r\n"
         << "Cache-Control: no-store\r\n"
         << "Connection: close\r\n"
         << "Content-Length: " << body.size() << "\r\n\r\n"
         << body;
    SendAll(fd, resp.str());
}

static std::thread StartStatusApi(const Config& cfg,
                                  Stats& stats,
                                  const UpdateState& ust,
                                  std::atomic<bool>& stop_all,
                                  const std::string& mode,
                                  const std::string& backend)
{
    return std::thread([&, mode, backend]() {
        Timer api_sp;
        int fd = -1;
        struct addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_PASSIVE;
        struct addrinfo* res = nullptr;
        const std::string port = std::to_string(cfg.api_port);
        const int gai = getaddrinfo(cfg.api_listen.c_str(), port.c_str(), &hints, &res);
        if (gai != 0) {
            LOGE("[api] getaddrinfo failed listen=" << cfg.api_listen << ":" << cfg.api_port
                 << " error=\"" << gai_strerror(gai) << "\"");
            return;
        }
        for (struct addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
            fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd < 0) continue;
            // Close-on-exec: the auto-update re-exec replaces this process while the API
            // is live; without CLOEXEC the listening socket fd leaks into the new image
            // and its rebind fails with EADDRINUSE, so the API never comes back after an
            // auto-update. fcntl form is portable (Linux + macOS); closes it on exec so
            // the re-exec'd process binds the port cleanly.
            fcntl(fd, F_SETFD, fcntl(fd, F_GETFD, 0) | FD_CLOEXEC);
            int yes = 1;
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
            struct timeval tv{};
            tv.tv_sec = 1;
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            if (::bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && ::listen(fd, 16) == 0) break;
            ::close(fd);
            fd = -1;
        }
        freeaddrinfo(res);
        if (fd < 0) {
            LOGE("[api] bind failed listen=" << cfg.api_listen << ":" << cfg.api_port
                 << " error=\"" << std::strerror(errno) << "\"");
            return;
        }

        LOGI("[api] listening http://" << cfg.api_listen << ":" << cfg.api_port
             << " endpoints=/health,/summary,/pools in " << api_sp.ms() << "ms");

        while (!stop_all.load()) {
            struct sockaddr_storage ss{};
            socklen_t slen = sizeof(ss);
            const int cfd = ::accept(fd, reinterpret_cast<struct sockaddr*>(&ss), &slen);
            if (cfd < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
                LOGW("[api] accept failed error=\"" << std::strerror(errno) << "\"");
                continue;
            }

            Timer req_sp;
            char buf[2048];
            const ssize_t n = ::recv(cfd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) {
                ::close(cfd);
                continue;
            }
            buf[n] = '\0';
            std::istringstream req(std::string(buf, static_cast<size_t>(n)));
            std::string method, path, proto;
            req >> method >> path >> proto;
            const size_t q = path.find('?');
            if (q != std::string::npos) path = path.substr(0, q);

            if (method != "GET") {
                SendHttpJson(cfd, 405, "Method Not Allowed", "{\"error\":\"method_not_allowed\"}");
            } else if (path == "/health") {
                SendHttpJson(cfd, 200, "OK", "{\"status\":\"ok\"}");
            } else if (path == "/summary" || path == "/") {
                SendHttpJson(cfd, 200, "OK", BuildSummaryJson(cfg, stats, ust, mode, backend));
            } else if (path == "/pools") {
                SendHttpJson(cfd, 200, "OK", "{\"pools\":" + BuildPoolsJson(cfg) + "}");
            } else {
                SendHttpJson(cfd, 404, "Not Found", "{\"error\":\"not_found\"}");
            }
            LOGD("[api] request method=" << method << " path=" << path << " in " << req_sp.ms() << "ms");
            ::close(cfd);
        }
        ::close(fd);
        LOGI("[api] stopped");
    });
}

static std::thread StartPoolWatchdog(const Config& cfg,
                                     Stats& stats,
                                     std::atomic<bool>& stop_all)
{
    return std::thread([&]() {
        // Zero-nonce threshold (seconds). Env-tunable like its siblings
        // (MATADOR_WATCHDOG_NO_SHARE_S etc.); 0 disables. Default 300s: a healthy
        // rig bumps total_nonces every batch (~1s), so 5 minutes of silence while
        // connected + gate-open is unambiguously a dead solve pipeline, not luck.
        int zero_nonce_s = 300;
        if (const char* v = std::getenv("MATADOR_WATCHDOG_ZERO_NONCE_S")) {
            int parsed = 0;
            if (SafeParseInt(v, parsed) && parsed >= 0) zero_nonce_s = parsed;
            else LOGW("[watchdog] ignoring bad MATADOR_WATCHDOG_ZERO_NONCE_S=\"" << v
                      << "\" (keeping " << zero_nonce_s << "s)");
        }
        // Degradation thresholds: warn when the live rate sits below degraded_pct% of this
        // rig's own peak for degraded_s. The zero-nonce rule above catches DEAD; this
        // catches ALIVE-BUT-SLOW -- throttling, a foreign GPU tenant, a sick card -- which
        // passes every other check because episodes keep advancing. Observe-only by design:
        // a degraded card is alive, so no restart can fix it; the value is a loud WHY
        // (throttle state + tenant list) instead of an ep/s number quietly sagging.
        int degraded_pct = 70;   // 0 disables
        int degraded_s = 180;    // must exceed several ~30s heartbeat windows: one bad window is a job switch, not degradation
        if (const char* v = std::getenv("MATADOR_WATCHDOG_DEGRADED_PCT")) {
            int parsed = 0;
            if (SafeParseInt(v, parsed) && parsed >= 0 && parsed < 100) degraded_pct = parsed;
            else LOGW("[watchdog] ignoring bad MATADOR_WATCHDOG_DEGRADED_PCT=\"" << v
                      << "\" (keeping " << degraded_pct << ")");
        }
        if (const char* v = std::getenv("MATADOR_WATCHDOG_DEGRADED_S")) {
            int parsed = 0;
            if (SafeParseInt(v, parsed) && parsed > 0) degraded_s = parsed;
            else LOGW("[watchdog] ignoring bad MATADOR_WATCHDOG_DEGRADED_S=\"" << v
                      << "\" (keeping " << degraded_s << "s)");
        }
        LOGI("[watchdog] enabled check_s=" << cfg.watchdog_check_s
             << " reject_streak=" << cfg.watchdog_reject_streak
             << " no_share_s=" << cfg.watchdog_no_share_s
             << " zero_nonce_s=" << zero_nonce_s
             << " degraded_pct=" << degraded_pct
             << " degraded_s=" << degraded_s
             << " thermal=" << (cfg.thermal_enabled ? "on" : "off")
             << " temp_warn_c=" << cfg.thermal_warn_temp_c
             << " temp_critical_c=" << cfg.thermal_critical_temp_c
             << " power_warn_w=" << cfg.thermal_warn_power_w
             << " actions=reconnect/failover (zero-nonce: process exit)");
        // Zero-nonce progress tracking: last observed nonce counter + the last time
        // we saw it move (or were in a state where not moving is expected).
        // stats.total_nonces only updates when a SolveMatMul CALL RETURNS -- with the
        // share sink active one call runs for the whole life of a job, so on a pool/hub
        // whose jobs live longer than the threshold (keepalive notifies keep 'connected'
        // true) it freezes while the GPU mines and submits shares fine. That false
        // positive _Exit(70)'d a healthy miner (2026-07-03, hub user, shares ACCEPTED
        // seconds before the kill). So ALSO watch the LIVE pipeline counters -- the same
        // ProbeMatMulSolvePipelineStats the stats heartbeat uses because it "reflects
        // work happening RIGHT NOW, even inside one long-running SolveMatMul call".
        // Counters reset per solve call, so ANY CHANGE (not just increase) = progress.
        uint64_t zn_last_total = stats.total_nonces.load();
        uint64_t zn_last_live = 0;
        int64_t zn_progress_ms = MonoMs();
        int64_t last_thermal_log_ms = 0;
        // Degradation-rule state: when the rate first fell below the floor (0 = not below),
        // last WARN emission (rate-limited to 1/min), and whether we are currently degraded
        // (so recovery gets one explicit log line instead of silently going quiet).
        int64_t dg_below_ms = 0;
        int64_t dg_last_log_ms = 0;
        bool dg_active = false;
        while (!stop_all.load()) {
            const int check_s = std::max(1, cfg.watchdog_check_s);
            for (int i = 0; i < check_s * 2 && !stop_all.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            if (stop_all.load()) break;

            const int64_t now_ms = MonoMs();
            const uint64_t reject_streak = stats.reject_streak.load();
            if (cfg.watchdog_reject_streak > 0 &&
                reject_streak >= static_cast<uint64_t>(cfg.watchdog_reject_streak)) {
                std::ostringstream msg;
                msg << "reject_streak=" << reject_streak
                    << " threshold=" << cfg.watchdog_reject_streak;
                WatchdogSetStatus(stats, "warning", msg.str(), "requested reconnect/failover");
                LOGW("[watchdog] " << msg.str() << "; requesting reconnect/failover");
                stats.watchdog_reconnect_requested.store(true);
                stats.reject_streak.store(0);
                continue;
            }

            if (cfg.watchdog_no_share_s > 0) {
                const int64_t last_nonce = stats.last_nonce_ms.load();
                const int64_t last_accept = stats.last_accept_ms.load();
                const bool hashing_recently =
                    last_nonce > 0 && (now_ms - last_nonce) <= (static_cast<int64_t>(check_s) + 5) * 1000;
                const bool no_accept_yet = last_accept <= 0;
                const int64_t accept_age_ms = no_accept_yet ? (now_ms - stats.started_ms) : (now_ms - last_accept);
                if (hashing_recently && accept_age_ms >= static_cast<int64_t>(cfg.watchdog_no_share_s) * 1000) {
                    std::ostringstream msg;
                    msg << "hashing_without_accepted_share threshold_s=" << cfg.watchdog_no_share_s
                        << " no_accept_yet=" << (no_accept_yet ? "true" : "false");
                    WatchdogSetStatus(stats, "warning", msg.str(), "requested reconnect/failover");
                    LOGW("[watchdog] " << msg.str() << "; requesting reconnect/failover");
                    stats.watchdog_reconnect_requested.store(true);
                    continue;
                }
            }

            // Zero-nonce rule: connected + gate-open + NO nonce progress at all. The
            // no-share rule above cannot see this failure mode -- it requires
            // hashing_recently, which is false EXACTLY when the GPU is dead (wedged
            // CUDA context, XID/driver reset, fallen-off-the-bus card). A pool
            // reconnect cannot revive a dead GPU context, so this rule uses the
            // strongest recovery there is: exit the process non-zero and let systemd
            // restart the miner with a FRESH CUDA context (the same fix an operator
            // would apply by hand -- and on a live rig this idles real money until
            // someone notices otherwise).
            if (zero_nonce_s > 0) {
                const uint64_t total = stats.total_nonces.load();
                const int64_t last_nonce = stats.last_nonce_ms.load();
                const int64_t last_notify = stats.last_notify_ms.load();
                // Live in-call progress: completed episodes and entered solve windows both
                // advance continuously while the solver works, and both freeze when the CUDA
                // context is truly wedged (which is the only failure this rule exists to
                // catch). Episodes are ~825 ms apiece, so the watchdog window must stay well
                // above that or a healthy miner looks stalled between ticks.
                const auto zn_pipe = ProbeMatMulSolvePipelineStats();
                const uint64_t live = zn_pipe.rc_episodes + zn_pipe.solve_windows;
                // "Connected" proxy: a mining.notify within the stratum stall window
                // (pools push work every ~5-30s; the reader's own stall watchdog
                // forces a reconnect after 120s without one). Never set in solo mode,
                // so this rule is pool-only by construction.
                const bool connected = last_notify > 0 && (now_ms - last_notify) <= 120000;
                if (total != zn_last_total || live != zn_last_live ||
                    (last_nonce > 0 && last_nonce > zn_progress_ms)) {
                    zn_last_total = total;                 // GPU alive: restart the window
                    zn_last_live = live;
                    zn_progress_ms = now_ms;
                } else if (!connected || !GateAllowsMining()) {
                    // Disconnected or idle-gated: zero nonces is EXPECTED, so hold the
                    // clock. This guarantees the rule can never fire during a pause or
                    // an outage, and the full threshold must elapse after resuming.
                    zn_progress_ms = now_ms;
                } else if (now_ms - zn_progress_ms >= static_cast<int64_t>(zero_nonce_s) * 1000) {
                    std::ostringstream msg;
                    msg << "zero_nonce_progress threshold_s=" << zero_nonce_s
                        << " stalled_s=" << ((now_ms - zn_progress_ms) / 1000)
                        << " total_nonces=" << total
                        << " live_pipeline=" << live
                        << " last_nonce_age_sec=" << (last_nonce > 0 ? (now_ms - last_nonce) / 1000 : -1);
                    WatchdogSetStatus(stats, "critical", msg.str(),
                                      "process exit (systemd restart, fresh GPU context)");
                    LOGE("[watchdog] " << msg.str()
                         << "; connected + gate-open but the GPU produced no nonces - dead/wedged"
                            " GPU context. Exiting so systemd restarts the miner with a fresh one.");
                    // NOT std::exit(): that runs static destructors, which can hang on
                    // the very CUDA context we are escaping. _Exit terminates NOW; the
                    // LOGE above is already flushed (Emit ends every line in
                    // std::endl) and the fd-level --log-file tee has the bytes too.
                    std::_Exit(70);   // EX_SOFTWARE; non-zero so systemd Restart= kicks in
                }
            }

            // Degradation rule: ALIVE but sustained well below this rig's own peak. The
            // failure modes this exists for -- throttling, a foreign tenant on the GPU, a
            // sick/renegotiated card -- keep episodes advancing, so zero-nonce never fires
            // and shares still trickle in; the only symptom is a rate quietly parked at a
            // fraction of what THIS rig has demonstrated. Compares against the rig's own
            // high-water mark (rate_peak_episode_per_s, published by the heartbeat), so no
            // absolute expectations are baked in and every card/OC combination calibrates
            // itself. Observe-only: unlike a wedged CUDA context, nothing a restart fixes.
            if (degraded_pct > 0) {
                const double rate = stats.rate_episode_per_s.load(std::memory_order_relaxed);
                const double win = stats.rate_window_sec.load(std::memory_order_relaxed);
                const double peak = stats.rate_peak_episode_per_s.load(std::memory_order_relaxed);
                const int64_t last_notify = stats.last_notify_ms.load();
                const bool connected = last_notify > 0 && (now_ms - last_notify) <= 120000;
                const double floor_eps = peak * degraded_pct / 100.0;
                if (!connected || !GateAllowsMining() || peak <= 0.0 || win < 20.0) {
                    // Paused, disconnected, or no baseline yet: a low rate is expected,
                    // hold the clock so the full threshold must elapse after resuming.
                    dg_below_ms = 0;
                } else if (rate >= floor_eps) {
                    if (dg_active) {
                        LOGI("[watchdog] rate recovered: " << std::fixed << std::setprecision(3)
                             << rate << " ep/s >= " << floor_eps << " (peak " << peak << ")");
                        dg_active = false;
                    }
                    dg_below_ms = 0;
                } else {
                    if (dg_below_ms == 0) dg_below_ms = now_ms;
                    if (now_ms - dg_below_ms >= static_cast<int64_t>(degraded_s) * 1000) {
                        dg_active = true;
                        // The WHY, from the same fork-free NVML path as /summary: is the
                        // card throttling, and is anything else on it.
                        const GpuTelemetry gt = QueryGpuTelemetry();
                        std::ostringstream msg;
                        msg << std::fixed << std::setprecision(3)
                            << "degraded_rate ep/s=" << rate << " peak=" << peak
                            << " floor=" << floor_eps << " (" << degraded_pct << "%)"
                            << " sustained_s=" << ((now_ms - dg_below_ms) / 1000)
                            << " throttle=" << (gt.ok && gt.has_throttle
                                                    ? ThrottleReasonsStr(gt.throttle) : "unknown")
                            << " foreign_procs=" << (gt.ok && gt.has_procs
                                                         ? std::to_string(gt.foreign_procs) : "unknown")
                            << (gt.ok && gt.has_procs && gt.foreign_procs > 0
                                    ? " foreign_mib=" + std::to_string(gt.foreign_mib) : "");
                        WatchdogSetStatus(stats, "warning", msg.str(), "observe-only (degraded, not dead)");
                        if (now_ms - dg_last_log_ms >= 60000) {
                            dg_last_log_ms = now_ms;
                            LOGW("[watchdog] " << msg.str()
                                 << "; rig is mining well below its own demonstrated rate --"
                                    " check throttle/tenants above, clocks, and cooling");
                        }
                        continue;   // hold the warning status; skip the ok reset below
                    }
                }
            }

            if (cfg.thermal_enabled) {
                // Same NVML-first fork-free query as /summary; this tick used to
                // fork nvidia-smi every check_s alongside the live CUDA contexts.
                const auto gpu_metrics = QueryGpuRuntimeMetrics();
                const std::string thermal_json = ThermalStatusJson(cfg, gpu_metrics);
                const bool thermal_bad =
                    thermal_json.find("\"status\":\"warning\"") != std::string::npos ||
                    thermal_json.find("\"status\":\"critical\"") != std::string::npos;
                if (thermal_bad) {
                    WatchdogSetStatus(stats, "warning", "thermal threshold crossed", "observe-only");
                    if (now_ms - last_thermal_log_ms >= 60000) {
                        last_thermal_log_ms = now_ms;
                        LOGW("[thermal] threshold crossed observe_only=true " << thermal_json);
                    }
                    continue;
                }
            }

            WatchdogSetStatus(stats, "ok", "", "");
        }
        LOGI("[watchdog] stopped");
    });
}

#endif  // MATADOR_STATUS_API_GPU_JSON_ONLY

// ===========================================================================
