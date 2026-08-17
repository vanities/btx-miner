// matador-miner: config-file / CLI-argument parsing + RPC-auth resolution.
// This is a SECTION of the single miner translation unit (not standalone): it
// uses Config, the mlog macros, and ParsePoolEndpoint defined/earlier-included
// in matador-miner.cpp, and is #included at the point those are in scope.
// Extracted verbatim from matador-miner.cpp.
// (Exception: the SafeParse* block right below IS standalone -- see its note.)
#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

// ---------------------------------------------------------------------------
// Safe numeric parsing -- pure, dependency-free (no Config/UniValue/mlog).
// Every env/CLI knob used to go straight through std::stoi/stoull/stod, which
// THROW on a typo ("--rpcport 1o0", MAXTRIES=1e9); an uncaught throw during arg
// parsing is std::terminate AT STARTUP, i.e. a systemd crash-loop on an
// unattended rig (it restarts into the same bad argv/env forever). Worse,
// stoi("1o0") WITHOUT a full-consume check silently parses as 1 and
// half-configures the miner. These parse the WHOLE string (modulo surrounding
// whitespace) or report failure, and never throw. `out` is untouched on
// failure so callers keep their defaults. Unit-tested in
// harness/safe_parse_test.cpp, which defines MATADOR_CONFIG_PARSE_HELPERS_ONLY
// to compile JUST this block standalone.
// ---------------------------------------------------------------------------
static bool SafeParseI64(const std::string& s, int64_t& out)
{
    try {
        size_t idx = 0;
        const long long v = std::stoll(s, &idx, 10);
        while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) ++idx;
        if (idx != s.size()) return false;             // trailing garbage ("1o0", "42x")
        out = static_cast<int64_t>(v);
        return true;
    } catch (const std::exception&) { return false; }  // empty / non-numeric / out-of-range
}

static bool SafeParseInt(const std::string& s, int& out)
{
    int64_t v = 0;
    if (!SafeParseI64(s, v)) return false;
    if (v < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        v > static_cast<int64_t>(std::numeric_limits<int>::max())) return false;
    out = static_cast<int>(v);
    return true;
}

static bool SafeParseUint64(const std::string& s, uint64_t& out)
{
    // stoull silently WRAPS negatives ("-5" -> 18446744073709551611): reject any
    // '-' before the digits instead of handing a rig a 2^64-scale maxtries.
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        if (c == '-') return false;
        break;
    }
    try {
        size_t idx = 0;
        const unsigned long long v = std::stoull(s, &idx, 10);
        while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) ++idx;
        if (idx != s.size()) return false;
        out = static_cast<uint64_t>(v);
        return true;
    } catch (const std::exception&) { return false; }
}

static bool SafeParseDouble(const std::string& s, double& out)
{
    try {
        size_t idx = 0;
        const double v = std::stod(s, &idx);
        while (idx < s.size() && std::isspace(static_cast<unsigned char>(s[idx]))) ++idx;
        if (idx != s.size()) return false;
        if (!std::isfinite(v)) return false;           // "nan"/"inf" are not config numbers
        out = v;
        return true;
    } catch (const std::exception&) { return false; }
}

#ifndef MATADOR_CONFIG_PARSE_HELPERS_ONLY

// ===========================================================================
// 6. Arg / env parsing.

// NOTE: struct Config is defined ABOVE section 5b (the pool client) because
// RunPoolLoop needs its full definition.

// Env numeric overrides: a fleet-file typo must WARN and keep the default -- a
// rig that would mine fine on defaults must never die (or silently mis-tune) on
// a bad env value. CLI numeric flags are different: those are an operator AT the
// keyboard/service file, so they fail LOUDLY with a usage error + clean exit(1)
// (see CliInt/CliUint64/CliDouble below).
static int EnvInt(const char* key, const char* raw, int dflt)
{
    int out = 0;
    if (raw != nullptr && SafeParseInt(raw, out)) return out;
    LOGW("[args] ignoring non-integer " << key << "=\"" << (raw ? raw : "") << "\" (keeping " << dflt << ")");
    return dflt;
}

static uint64_t EnvUint64(const char* key, const char* raw, uint64_t dflt)
{
    uint64_t out = 0;
    if (raw != nullptr && SafeParseUint64(raw, out)) return out;
    LOGW("[args] ignoring non-integer " << key << "=\"" << (raw ? raw : "") << "\" (keeping " << dflt << ")");
    return dflt;
}

static double EnvDouble(const char* key, const char* raw, double dflt)
{
    double out = 0.0;
    if (raw != nullptr && SafeParseDouble(raw, out)) return out;
    LOGW("[args] ignoring non-numeric " << key << "=\"" << (raw ? raw : "") << "\" (keeping " << dflt << ")");
    return dflt;
}

// CLI numeric flags: a typo like --rpcport=1o0 must never std::terminate (an
// unhelpful abort with no message) and never silently half-configure
// (stoi("1o0")==1 would quietly mine with the wrong knob). Print the usage
// error and exit(1) cleanly -- parsing happens at startup, before any threads.
static int CliInt(const char* flag, const std::string& v)
{
    int out = 0;
    if (SafeParseInt(v, out)) return out;
    LOGE("[args] --" << flag << " wants an integer, got: \"" << v << "\"");
    std::exit(1);
}

static uint64_t CliUint64(const char* flag, const std::string& v)
{
    uint64_t out = 0;
    if (SafeParseUint64(v, out)) return out;
    LOGE("[args] --" << flag << " wants a non-negative integer, got: \"" << v << "\"");
    std::exit(1);
}

static double CliDouble(const char* flag, const std::string& v)
{
    double out = 0.0;
    if (SafeParseDouble(v, out)) return out;
    LOGE("[args] --" << flag << " wants a number, got: \"" << v << "\"");
    std::exit(1);
}

static std::string GetEnvOr(const char* key, const std::string& dflt)
{
    const char* v = std::getenv(key);
    return v ? std::string(v) : dflt;
}

// ParsePoolEndpoint() moved to miner/endpoint_parse.h (unit-tested).

static std::string TrimCopy(const std::string& in)
{
    const size_t a = in.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = in.find_last_not_of(" \t\r\n");
    return in.substr(a, b - a + 1);
}

// SplitHostPort() moved to miner/endpoint_parse.h (unit-tested).

static bool AddPoolEndpoint(Config& cfg, const std::string& raw, const std::string& label)
{
    std::string host;
    int port = 0;
    bool use_tls = false;
    if (!ParsePoolEndpoint(raw, host, port, &use_tls)) return false;
    cfg.pools.push_back(PoolEndpoint{host, port, label, use_tls});
    cfg.pool_host = host;
    cfg.pool_port = port;
    return true;
}

static bool AddPoolEndpointList(Config& cfg, const std::string& raw, bool replace_existing, const char* scope)
{
    if (replace_existing) {
        cfg.pools.clear();
        cfg.pool_host.clear();
        cfg.pool_port = 0;
    }

    size_t added = 0;
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t end = raw.find(',', start);
        const std::string token = TrimCopy(raw.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty()) {
            if (!AddPoolEndpoint(cfg, token, cfg.pools.empty() ? "primary" : "")) {
                LOGE("[" << scope << "] pool must be host:port (or stratum+tcp://host:port): " << token);
                return false;
            }
            ++added;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    if (added == 0) {
        LOGE("[" << scope << "] pool list is empty");
        return false;
    }
    cfg.pool_host = cfg.pools.front().host;
    cfg.pool_port = cfg.pools.front().port;
    return true;
}

static bool ConfigValue(const UniValue& obj,
                        std::initializer_list<const char*> keys,
                        const UniValue*& out)
{
    if (!obj.isObject()) return false;
    for (const char* key : keys) {
        if (obj.exists(key)) {
            out = &obj[key];
            return true;
        }
    }
    return false;
}

static bool ConfigString(const UniValue& obj,
                         std::initializer_list<const char*> keys,
                         std::string& out)
{
    const UniValue* v = nullptr;
    if (!ConfigValue(obj, keys, v)) return false;
    if (!v->isStr()) {
        LOGW("[config] string key has non-string value; ignoring");
        return false;
    }
    out = v->get_str();
    return true;
}

static bool ConfigInt(const UniValue& obj,
                      std::initializer_list<const char*> keys,
                      int& out)
{
    const UniValue* v = nullptr;
    if (!ConfigValue(obj, keys, v)) return false;
    try {
        if (v->isNum()) { out = static_cast<int>(v->getInt<int64_t>()); return true; }
        if (v->isStr()) { out = std::stoi(v->get_str()); return true; }
    } catch (const std::exception& e) {
        LOGW("[config] integer parse error: " << e.what());
        return false;
    }
    LOGW("[config] integer key has non-integer value; ignoring");
    return false;
}

static bool ConfigUint64(const UniValue& obj,
                         std::initializer_list<const char*> keys,
                         uint64_t& out)
{
    const UniValue* v = nullptr;
    if (!ConfigValue(obj, keys, v)) return false;
    try {
        if (v->isNum()) { out = static_cast<uint64_t>(v->getInt<int64_t>()); return true; }
        if (v->isStr()) { out = std::stoull(v->get_str()); return true; }
    } catch (const std::exception& e) {
        LOGW("[config] uint64 parse error: " << e.what());
        return false;
    }
    LOGW("[config] uint64 key has non-integer value; ignoring");
    return false;
}

static bool ConfigBool(const UniValue& obj,
                       std::initializer_list<const char*> keys,
                       bool& out)
{
    const UniValue* v = nullptr;
    if (!ConfigValue(obj, keys, v)) return false;
    if (v->isBool()) { out = v->get_bool(); return true; }
    if (v->isNum()) { out = v->getInt<int64_t>() != 0; return true; }
    if (v->isStr()) {
        std::string s = v->get_str();
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (s == "1" || s == "true" || s == "yes" || s == "on")  { out = true;  return true; }
        if (s == "0" || s == "false"|| s == "no"  || s == "off") { out = false; return true; }
    }
    LOGW("[config] bool key has non-bool value; ignoring");
    return false;
}

static std::vector<std::string> ParseGpuDeviceList(const std::string& raw)
{
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t end = raw.find(',', start);
        std::string token = TrimCopy(raw.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!token.empty()) out.push_back(token);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

static bool ConfigGpuDevices(const UniValue& obj, Config& cfg, int& applied)
{
    const UniValue* v = nullptr;
    if (!ConfigValue(obj, {"gpus", "gpu_devices", "gpu-devices", "devices"}, v)) return true;

    std::vector<std::string> devices;
    if (v->isStr()) {
        devices = ParseGpuDeviceList(v->get_str());
    } else if (v->isArray()) {
        for (const UniValue& item : v->getValues()) {
            if (item.isStr()) {
                const std::string token = TrimCopy(item.get_str());
                if (!token.empty()) devices.push_back(token);
            } else if (item.isNum()) {
                devices.push_back(std::to_string(item.getInt<int64_t>()));
            } else {
                LOGE("[config] gpus entries must be strings or integers");
                return false;
            }
        }
    } else {
        LOGE("[config] gpus must be a comma string or an array of device ids");
        return false;
    }

    if (devices.empty()) {
        LOGE("[config] gpus must contain at least one device id when present");
        return false;
    }
    cfg.gpu_devices = std::move(devices);
    ++applied;
    return true;
}

static bool ConfigPools(const UniValue& obj, Config& cfg, int& applied)
{
    const UniValue* v = nullptr;
    if (!ConfigValue(obj, {"pools"}, v)) return true;
    if (!v->isArray()) {
        LOGE("[config] pools must be an array of endpoint strings or objects");
        return false;
    }

    cfg.pools.clear();
    cfg.pool_host.clear();
    cfg.pool_port = 0;

    size_t idx = 0;
    for (const UniValue& item : v->getValues()) {
        std::string endpoint;
        std::string label;
        if (item.isStr()) {
            endpoint = item.get_str();
        } else if (item.isObject()) {
            if (item.exists("label") && item["label"].isStr()) label = item["label"].get_str();
            if (item.exists("name") && item["name"].isStr() && label.empty()) label = item["name"].get_str();
            if (item.exists("url") && item["url"].isStr()) {
                endpoint = item["url"].get_str();
            } else if (item.exists("host") && item["host"].isStr() && item.exists("port")) {
                int port = 0;
                try {
                    if (item["port"].isNum()) port = static_cast<int>(item["port"].getInt<int64_t>());
                    else if (item["port"].isStr()) port = std::stoi(item["port"].get_str());
                } catch (const std::exception& e) {
                    LOGE("[config] pools[" << idx << "] port parse error: " << e.what());
                    return false;
                }
                endpoint = item["host"].get_str() + ":" + std::to_string(port);
            } else {
                LOGE("[config] pools[" << idx << "] object requires url or host+port");
                return false;
            }
        } else {
            LOGE("[config] pools[" << idx << "] must be string or object");
            return false;
        }

        if (!AddPoolEndpoint(cfg, endpoint, label.empty() && cfg.pools.empty() ? "primary" : label)) {
            LOGE("[config] pools[" << idx << "] invalid endpoint: " << endpoint);
            return false;
        }
        ++idx;
    }

    if (cfg.pools.empty()) {
        LOGE("[config] pools must contain at least one endpoint");
        return false;
    }
    cfg.pool_host = cfg.pools.front().host;
    cfg.pool_port = cfg.pools.front().port;
    ++applied;
    return true;
}

// Load optional JSON config before env/CLI overrides. This is the farm-operator
// path: one stable file for systemd/Hive-style installs, with secrets kept in the
// file or env and never echoed back. Precedence is defaults < config < env < CLI.
static bool LoadConfigFile(const std::string& path, Config& cfg)
{
    if (path.empty()) return true;

    Timer sp;
    std::ifstream f(path);
    if (!f) {
        LOGE("[config] cannot read config file=" << path);
        return false;
    }
    std::ostringstream body;
    body << f.rdbuf();

    UniValue root;
    if (!root.read(body.str()) || !root.isObject()) {
        LOGE("[config] config file must be a JSON object: " << path);
        return false;
    }

    int applied = 0;
    auto bump = [&](bool changed) { if (changed) ++applied; };
    std::string s;
    int i = 0;
    uint64_t u64 = 0;
    bool b = false;

    bump(ConfigString(root, {"mode"}, cfg.mode));
    bump(ConfigString(root, {"log_file", "log-file", "logfile"}, cfg.log_file_path));
    if (ConfigString(root, {"pool"}, s)) {
        if (!AddPoolEndpointList(cfg, s, /*replace_existing=*/true, "config")) return false;
        ++applied;
    }
    if (!ConfigPools(root, cfg, applied)) return false;
    bump(ConfigString(root, {"worker"}, cfg.worker));
    bump(ConfigString(root, {"operator_label", "operator-label"}, cfg.operator_label));
    bump(ConfigString(root, {"pool_pass", "pool-pass"}, cfg.pool_pass));
    if (ConfigBool(root, {"pool_tls_insecure", "pool-tls-insecure"}, b)) { cfg.pool_tls_insecure = b; ++applied; }
    {
        std::string sp;
        if (ConfigString(root, {"socks5"}, sp)) {
            if (SplitHostPort(sp, cfg.socks5_host, cfg.socks5_port)) ++applied;
            else LOGW("[config] ignoring malformed socks5 (want host:port): " << sp);
        }
    }
    bump(ConfigString(root, {"socks5_user", "socks5-user"}, cfg.socks5_user));
    bump(ConfigString(root, {"socks5_pass", "socks5-pass"}, cfg.socks5_pass));
    bump(ConfigString(root, {"rpcconnect"}, cfg.rpcconnect));
    if (ConfigInt(root, {"rpcport"}, i)) { cfg.rpcport = i; cfg.rpcport_explicit = true; ++applied; }
    bump(ConfigString(root, {"datadir"}, cfg.datadir));
    bump(ConfigString(root, {"rpccookiefile", "rpc_cookie_file"}, cfg.rpccookiefile));
    bump(ConfigString(root, {"rpcuser"}, cfg.rpcuser));
    bump(ConfigString(root, {"rpcpassword"}, cfg.rpcpassword));
    bump(ConfigString(root, {"payoutaddress", "payout_address"}, cfg.payoutaddress));
    bump(ConfigString(root, {"chain"}, cfg.chain));
    if (ConfigUint64(root, {"maxtries", "max_tries"}, u64)) { cfg.maxtries = u64; ++applied; }
    if (ConfigInt(root, {"rc_height", "rc-height"}, i)) { cfg.rc_height = std::max(0, i); ++applied; }
    bump(ConfigString(root, {"attest_key_file", "attest-key-file"}, cfg.attest_key_file));
    bump(ConfigString(root, {"attest_context", "attest-context"}, cfg.attest_context));
    if (ConfigInt(root, {"devfee", "dev_fee", "dev-fee"}, i)) { cfg.devfee = i; ++applied; }
    bump(ConfigString(root, {"devaddress", "dev_address", "dev-address"}, cfg.devaddress));
    bump(ConfigString(root, {"backend"}, cfg.backend));
    if (!ConfigGpuDevices(root, cfg, applied)) return false;
    // "sidecars" only ever configured the external HIP/AMD solver bridge, which went
    // out with the v3 solver. Still shape-check it so a stale config gets a clear
    // message instead of silent acceptance, and say plainly that it does nothing now.
    if (root.exists("sidecars")) {
        if (!root["sidecars"].isObject()) {
            LOGE("[config] sidecars must be an object when present");
            return false;
        }
        LOGW("[config] 'sidecars' is IGNORED: the external HIP/AMD solver bridge was removed "
             "with the v3 solver (no ENC_RC path ever existed for it)");
    }
    if (ConfigInt(root, {"solver_threads", "solver-threads"}, i)) { cfg.solver_threads = std::max(1, i); ++applied; }
    if (ConfigInt(root, {"clk_offset", "clk-offset", "gpu_clock_offset"}, i)) { cfg.clk_offset = i; ++applied; }
    if (ConfigInt(root, {"power_limit", "power_limit_w", "power-limit"}, i)) { cfg.power_limit_w = i; ++applied; }
    if (ConfigInt(root, {"lock_gpu_clock", "lock-gpu-clock", "core_clock_lock"}, i)) { cfg.lock_gpu_clock = i; ++applied; }
    if (ConfigInt(root, {"fan_pct", "fan-pct", "fan_speed", "gpu_fan"}, i)) { cfg.fan_pct = i; ++applied; }
    if (ConfigInt(root, {"mem_clk_offset", "mem-clk-offset", "mem_clock_offset", "vram_clk_offset"}, i)) { cfg.mem_clk_offset = i; ++applied; }
    if (ConfigInt(root, {"lock_mem_clock", "lock-mem-clock", "mem_clock_lock"}, i)) { cfg.lock_mem_clock = i; ++applied; }
    if (ConfigBool(root, {"gpu_worker_suffix", "gpu-worker-suffix"}, b)) { cfg.gpu_worker_suffix = b; ++applied; }
    if (ConfigBool(root, {"update_check", "update-check"}, b)) { cfg.update_check = b; ++applied; }
    if (ConfigBool(root, {"auto_update", "auto-update"}, b)) { cfg.auto_update = b; ++applied; }
    if (ConfigInt(root, {"update_interval_s", "update-interval-s"}, i)) { cfg.update_interval_s = i; ++applied; }
    if (ConfigInt(root, {"update_jitter_s", "update-jitter-s"}, i)) { cfg.update_jitter_s = std::max(0, i); ++applied; }
    if (ConfigInt(root, {"min_version_age_s", "min-version-age-s"}, i)) { cfg.min_version_age_s = std::max(0, i); ++applied; }
    bump(ConfigString(root, {"update_channel", "update-channel"}, cfg.update_channel));
    bump(ConfigString(root, {"fallback_pool", "fallback-pool"}, cfg.fallback_pool));
    if (ConfigInt(root, {"fallback_after_s", "fallback-after-s"}, i)) { cfg.fallback_after_s = std::max(1, i); ++applied; }
    if (ConfigInt(root, {"solo_recheck_s", "solo-recheck-s"}, i)) { cfg.solo_recheck_s = std::max(10, i); ++applied; }
    bump(ConfigString(root, {"should_mine_command", "should-mine-command"}, cfg.should_mine_command));
    if (ConfigInt(root, {"should_mine_interval", "should-mine-interval"}, i)) { cfg.should_mine_interval = std::max(1, i); ++applied; }
    bump(ConfigString(root, {"gate_yield", "gate-yield"}, cfg.gate_yield));
    if (ConfigBool(root, {"api_enabled", "api-enabled"}, b)) { cfg.api_enabled = b; ++applied; }
    bump(ConfigString(root, {"api_listen", "api-listen"}, cfg.api_listen));
    if (ConfigInt(root, {"api_port", "api-port"}, i)) { cfg.api_port = i; cfg.api_enabled = cfg.api_enabled || i > 0; ++applied; }
    if (root.exists("api") && root["api"].isObject()) {
        const UniValue& api = root["api"];
        if (ConfigBool(api, {"enabled"}, b)) { cfg.api_enabled = b; ++applied; }
        bump(ConfigString(api, {"listen", "host", "bind"}, cfg.api_listen));
        if (ConfigInt(api, {"port"}, i)) { cfg.api_port = i; cfg.api_enabled = cfg.api_enabled || i > 0; ++applied; }
    } else if (root.exists("api") && !root["api"].isObject()) {
        LOGE("[config] api must be an object when present");
        return false;
    }
    if (ConfigBool(root, {"watchdog_enabled", "watchdog-enabled"}, b)) { cfg.watchdog_enabled = b; ++applied; }
    if (ConfigInt(root, {"watchdog_check_s", "watchdog-check-s"}, i)) { cfg.watchdog_check_s = std::max(1, i); ++applied; }
    if (ConfigInt(root, {"watchdog_reject_streak", "watchdog-reject-streak"}, i)) { cfg.watchdog_reject_streak = std::max(0, i); ++applied; }
    if (ConfigInt(root, {"watchdog_no_share_s", "watchdog-no-share-s"}, i)) { cfg.watchdog_no_share_s = std::max(0, i); ++applied; }
    if (root.exists("watchdog") && root["watchdog"].isObject()) {
        const UniValue& wd = root["watchdog"];
        if (ConfigBool(wd, {"enabled"}, b)) { cfg.watchdog_enabled = b; ++applied; }
        if (ConfigInt(wd, {"check_s", "check_interval_s"}, i)) { cfg.watchdog_check_s = std::max(1, i); ++applied; }
        if (ConfigInt(wd, {"reject_streak", "reject_streak_threshold"}, i)) { cfg.watchdog_reject_streak = std::max(0, i); ++applied; }
        if (ConfigInt(wd, {"no_share_s", "no_share_action_s"}, i)) { cfg.watchdog_no_share_s = std::max(0, i); ++applied; }
    } else if (root.exists("watchdog") && !root["watchdog"].isObject()) {
        LOGE("[config] watchdog must be an object when present");
        return false;
    }
    if (ConfigBool(root, {"thermal_enabled", "thermal-enabled"}, b)) { cfg.thermal_enabled = b; ++applied; }
    if (ConfigInt(root, {"thermal_warn_temp_c", "thermal-warn-temp-c"}, i)) { cfg.thermal_warn_temp_c = std::max(0, i); ++applied; }
    if (ConfigInt(root, {"thermal_critical_temp_c", "thermal-critical-temp-c"}, i)) { cfg.thermal_critical_temp_c = std::max(0, i); ++applied; }
    if (ConfigInt(root, {"thermal_warn_power_w", "thermal-warn-power-w"}, i)) { cfg.thermal_warn_power_w = std::max(0, i); ++applied; }
    if (root.exists("thermal") && root["thermal"].isObject()) {
        const UniValue& th = root["thermal"];
        if (ConfigBool(th, {"enabled"}, b)) { cfg.thermal_enabled = b; ++applied; }
        if (ConfigInt(th, {"warn_temp_c", "temp_warn_c"}, i)) { cfg.thermal_warn_temp_c = std::max(0, i); ++applied; }
        if (ConfigInt(th, {"critical_temp_c", "temp_critical_c"}, i)) { cfg.thermal_critical_temp_c = std::max(0, i); ++applied; }
        if (ConfigInt(th, {"warn_power_w", "power_warn_w"}, i)) { cfg.thermal_warn_power_w = std::max(0, i); ++applied; }
    } else if (root.exists("thermal") && !root["thermal"].isObject()) {
        LOGE("[config] thermal must be an object when present");
        return false;
    }

    LOGI("[config] loaded file=" << path << " settings=" << applied
         << " bytes=" << body.str().size() << " in " << sp.ms() << "ms");
    return true;
}

static std::string AutoConfigPath()
{
    // Zero-config onboarding: if the operator runs from an unpacked bundle or a
    // rig directory containing ./matador.json, use it automatically. Missing is OK;
    // an explicitly selected --config/MATADOR_CONFIG path still errors if unreadable.
    struct stat st;
    if (stat("matador.json", &st) == 0 && S_ISREG(st.st_mode)) return "matador.json";
    return "";
}

// ASCII splash printed once at startup, straight to stdout and INDEPENDENT of
// LOG_LEVEL (it is a banner, not a log line). Ole.
static void PrintBanner()
{
    std::cout << R"(
   ███╗   ███╗ █████╗ ████████╗ █████╗ ██████╗  ██████╗ ██████╗
   ████╗ ████║██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔═══██╗██╔══██╗
   ██╔████╔██║███████║   ██║   ███████║██║  ██║██║   ██║██████╔╝
   ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║██║  ██║██║   ██║██╔══██╗
   ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██████╔╝╚██████╔╝██║  ██║
   ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝
        s t a n d a l o n e   B T X   M a t M u l   m i n e r
)";
    std::cout <<
        "                                                      " MATADOR_MINER_VERSION "\n"
        R"(
                 ,/         the cape flutters...
       _,--._   /'           ole!
      /      `-'             ___
     |  o  o  |             /   \   <- toro never sees the nonce coming
      \   ^   /---._       ( x x )
       `.___.'     `\======/`-v-'
         |  |        ||    ||
         |  |        ||    ||
        (megafield matmul: 512x512 over M31 - we swing the cape, the GPU lands the blade)
)";
    std::cout << std::flush;
}

static void PrintHelp()
{
    std::cout <<
        "matador-miner: standalone solo GBT miner for BTX\n"
        "\n"
        "Usage: matador-miner [options]\n"
        "\n"
        "  --config=<path>          optional JSON config file (defaults to ./matador.json if present;\n"
        "                           env MATADOR_CONFIG also supported). Precedence:\n"
        "                           defaults < config < env < CLI; secrets are never logged.\n"
        "  --mode=<solo|pool>       solo (default; getblocktemplate -> submitblock to local btxd)\n"
        "                           or pool (env MODE). POOL MODE TARGETS THE minebtx/dexbtx\n"
        "                           pool protocol (v18: pre_hash_block_tier capability, matmul\n"
        "                           mining.notify, worker.report_metrics); other pools may differ.\n"
        "                           In pool mode --payoutaddress is REQUIRED (the stratum user).\n"
        "  --pool=<host:port>       minebtx/dexbtx stratum endpoint (pool mode). Repeat for\n"
        "                           failover, or comma-separate env POOL / config pool.\n"
        "                           host:port or stratum+tcp://host:port; config also supports pools[].\n"
        "                           ssl://host:port / tls:// (also stratum+ssl:// / stratum+tls://)\n"
        "                           encrypts the stratum socket with TLS (e.g. ninjaraider's\n"
        "                           ssl://ninjaraider.com:44921). Verifies the server cert by\n"
        "                           default; see --pool-tls-insecure.\n"
        "  --worker=<name>          stratum worker suffix -> user is \"<payoutaddress>.<worker>\"\n"
        "                           (default matador; env WORKER)\n"
        "  --operator-label=<name>  pool dashboard label (env OPERATOR_LABEL). Defaults to --worker.\n"
        "                           Multi-GPU fan-out auto-shares one label so the rig shows as a\n"
        "                           single \"hive\" on minebtx; set explicitly to group several boxes.\n"
        "  --pool-pass=<pass>       stratum password (default x; env POOL_PASS)\n"
        "  --pool-tls-insecure      skip certificate verification on a TLS (ssl://) pool connection\n"
        "                           (still encrypted, just no identity/MITM check -- for a pool on a\n"
        "                           self-signed cert). Default verifies against the system CA store.\n"
        "                           env MATADOR_POOL_TLS_INSECURE=1; config pool_tls_insecure.\n"
        "  --socks5=<host:port>     route the pool connection through a SOCKS5 proxy (RFC 1928).\n"
        "                           The PROXY resolves the pool host, so tailnet/MagicDNS names\n"
        "                           with no local route work (e.g. Tailscale userspace, ssh -D).\n"
        "                           Env SOCKS5=host:port, or ALL_PROXY=socks5://host:port.\n"
        "  --socks5-user=<user>     SOCKS5 username (RFC 1929 user/pass auth; env SOCKS5_USER)\n"
        "  --socks5-pass=<pass>     SOCKS5 password (env SOCKS5_PASS)\n"
        "  --chain=<net>            main|test|regtest (default main; env CHAIN). Drives consensus\n"
        "                           params + the default RPC port (main 19334, test 29334, regtest 18443)\n"
        "  --rpcconnect=<ip>        btxd host (default 127.0.0.1; env RPCCONNECT)\n"
        "  --rpcport=<port>         btxd RPC port (default per --chain; env RPCPORT)\n"
        "  --datadir=<dir>          btxd datadir; cookie read from <dir>/.cookie (env DATADIR)\n"
        "  --rpccookiefile=<path>   explicit cookie file (overrides datadir; env RPCCOOKIEFILE)\n"
        "  --rpcuser=<user>         RPC user (instead of cookie; env RPCUSER)\n"
        "  --rpcpassword=<pass>     RPC password (env RPCPASSWORD)\n"
        "  --payoutaddress=<addr>   REQUIRED. btx1... P2MR (witness v2 / 32-byte) address (env PAYOUTADDRESS)\n"
        "  --maxtries=<n>           solve budget per template before re-GBT (env MAXTRIES)\n"
        "  --verify                 One-shot: re-verify a block header or a submitted share\n"
        "                           using the SAME episode backend the solver uses. Needs\n"
        "                           --header <hex> --height <n> --parent-mtp <n>; add\n"
        "                           --share-target <64-hex> to grade a pool share. Exits\n"
        "                           0 valid, 1 invalid. No pool/payout/config required.\n"
        "  --rc-height=<n>          PIN the ENC_RC activation height (env BTX_MATMUL_RC_HEIGHT,\n"
        "                           config rc_height). Default 0 = auto: pool mode latches\n"
        "                           activation from the job itself (enc-rc-* profile or the RC\n"
        "                           matmul dim), so no height is compiled in. A pin always wins;\n"
        "                           BTX_RC_STRATUM_AUTOLATCH=0 disables the auto-detect.\n"
        "  --backend=<name>         force solver backend: cuda|cpu (env BTX_MATMUL_BACKEND).\n"
        "                           cpu runs the byte-exact ENC_RC oracle -- a determinism\n"
        "                           cross-check, ~200x too slow to mine with.\n"
        "                           (usually not needed when using the release bundle).\n"
        "  --gpus=<ids>             multi-GPU process fan-out, e.g. 0,1,2 (env MATADOR_GPUS;\n"
        "                           config gpus:[0,1]). One miner process per device, worker\n"
        "                           suffix -gpuN, API ports incremented; no cross-GPU tuning.\n"
        "                           DEFAULT auto-detects ALL GPUs and mines on every one; use\n"
        "                           --gpus auto to force, or --gpus 0 to pin a single card.\n"
        "  --gpu-suffix             multi-GPU: append a per-card -gpuN suffix to the pool worker name\n"
        "                           (one worker row per card on the pool dashboard). DEFAULT OFF: all\n"
        "                           cards report under ONE worker name, like single-process miners.\n"
        "                           Purely cosmetic - each card always has its own pool connection,\n"
        "                           nonce lane, and share credit. env MATADOR_GPU_WORKER_SUFFIX=1;\n"
        "                           config gpu_worker_suffix:true. --no-gpu-suffix forces it off.\n"
        "  --solver-threads=<n>     concurrent in-flight GPU solves over disjoint nonce ranges\n"
        "  --clk-offset=<mhz>       NVIDIA GPC clock offset (MHz) applied via NVML at startup (needs\n"
        "                           root). Power-capped cards: +offset = more clock per watt = more\n"
        "                           nps. Also BTX_GPU_CLK_OFFSET / config \"clk_offset\". 0=stock\n"
        "  --power-limit=<w>        NVIDIA board power limit in watts (needs root). At a power cap,\n"
        "                           watts=nps; guarantees the full budget. BTX_GPU_POWER_LIMIT /\n"
        "                           config \"power_limit\". 0=leave as-is\n"
        "  --lock-gpu-clock=<mhz>   pin the NVIDIA core clock (MHz) for stability/consistency, or to\n"
        "                           cap the boost so a too-high --clk-offset can't wedge the GPU\n"
        "                           (needs root). BTX_GPU_LOCK_CLOCK / config \"lock_gpu_clock\". 0=off\n"
        "  --fan-pct=<pct>          pin the NVIDIA fan duty (1..100%) via NVML at startup (needs root).\n"
        "                           Holds the thermal margin so the boost can't step down (flatter\n"
        "                           nps in a warm room). BTX_GPU_FAN_PCT / config \"fan_pct\". 0=auto.\n"
        "  --mem-clk-offset=<mhz>   NVIDIA memory (GDDR7) clock offset (MHz) via NVML at startup (needs\n"
        "                           root). Power-capped + HBM-traffic-bound digest: faster mem = more\n"
        "                           bandwidth per watt = more nps. CAUTION: too high corrupts digests\n"
        "                           silently -> rejects; raise small + watch rej. BTX_GPU_MEM_CLK_OFFSET\n"
        "                           / config \"mem_clk_offset\". 0=stock\n"
        "  --lock-mem-clock=<mhz>   pin the NVIDIA memory clock (MHz) via NVML at startup (needs root).\n"
        "                           For power/stability tuning; snaps to the card's supported points.\n"
        "                           Overrides --mem-clk-offset. BTX_GPU_LOCK_MEM_CLOCK /\n"
        "                           config \"lock_mem_clock\". 0=unlocked\n"
        "                           All GPU tuning is reverted to stock on shutdown.\n"
        "                           (env MATADOR_SOLVER_THREADS; default 1). Keep 1: an ENC_RC\n"
        "                           episode already saturates the card, and concurrent episodes\n"
        "                           cost a large per-thread-state tax.\n"
        "  --no-api                 disable the local read-only HTTP status API (ON by default,\n"
        "                           loopback 127.0.0.1:4060; env MATADOR_API=0). The fleet hub reads it.\n"
        "  --api-listen=<addr>      status API bind address (default 127.0.0.1; env MATADOR_API_LISTEN)\n"
        "  --api-port=<port>        status API port (default 4060; 0 disables; env MATADOR_API_PORT). Endpoints:\n"
        "                           /health, /summary, /pools. Never exposes RPC/pool secrets.\n"
        "  --watchdog              enable pool watchdog (default on; env MATADOR_WATCHDOG=0 disables)\n"
        "  --watchdog-reject-streak=<n> reconnect/failover after n consecutive rejects\n"
        "                           (default 20; env MATADOR_WATCHDOG_REJECT_STREAK; 0 disables)\n"
        "  --watchdog-no-share-s=<n> optional reconnect if hashing but no accepted share for n sec\n"
        "                           (default 0/off; env MATADOR_WATCHDOG_NO_SHARE_S)\n"
        "  --thermal               enable warning-only GPU temp/power watchdog (default on;\n"
        "                           env MATADOR_THERMAL=0 disables; no clocks/fans/restarts)\n"
        "  --thermal-warn-temp-c=<n> warn when any GPU reaches n C (default 86)\n"
        "  --thermal-critical-temp-c=<n> critical warning level only (default 90)\n"
        "  --thermal-warn-power-w=<n> optional power warning in watts (default 0/off)\n"
        "  --no-update-check        skip the GitHub release check entirely (startup AND periodic)\n"
        "  --no-auto-update         check + notify, but do NOT auto-download/replace. Default is\n"
        "                           auto-update: fetch the latest release, verify sha256, swap the\n"
        "                           binary, and re-exec into it (graceful, same PID, no node restart).\n"
        "  --update-channel=<c>     stable (default; GitHub 'Latest', non-prerelease) or prerelease\n"
        "                           (newest tag incl. prereleases). env MATADOR_UPDATE_CHANNEL.\n"
        "  --update-interval-s=<n>  periodic re-check cadence after startup (default 1800=30min;\n"
        "                           <=0 = startup-only). env MATADOR_UPDATE_INTERVAL_S.\n"
        "  --update-jitter-s=<n>    randomized 0..n s delay before each periodic check to de-sync a\n"
        "                           fleet (default 300; 0=off). env MATADOR_UPDATE_JITTER_S.\n"
        "  --min-version-age-s=<n>  bake-time: only auto-adopt a release >=n s old (default 3600;\n"
        "                           0=off). Stops a fleet jumping onto a brand-new bad release.\n"
        "  --update-check-only      run ONE update check (may swap+re-exec into a newer release)\n"
        "                           then EXIT. No payout/RPC/GPU needed. Use as a systemd .timer\n"
        "                           entrypoint when you prefer systemd to own the update cadence.\n"
        "  --fallback-pool=<p>      solo->pool failover: if the solo node/coordinator (the GBT\n"
        "                           source) is unreachable for --fallback-after-s, fail over to\n"
        "                           this pool, then return to solo when it recovers. env\n"
        "                           MATADOR_FALLBACK_POOL. Empty = no fallback (solo retries).\n"
        "  --fallback-after-s=<n>   solo RPC down this long before failing over (default 60).\n"
        "  --solo-recheck-s=<n>     while on the fallback pool, probe the solo node this often\n"
        "                           to return to solo (default 120).\n"
        "  --should-mine-command=<c> idle-gate: run <c> every --should-mine-interval; exit 0 =\n"
        "                           mine, non-zero = YIELD the GPU (pause). e.g. a GPU-idle or\n"
        "                           session check. Empty = always mine. env MATADOR_SHOULD_MINE_COMMAND.\n"
        "                           Reference gate: scripts/gpu-idle.sh <idle_min> <util%%>.\n"
        "  --should-mine-interval=<n> idle-gate poll cadence in seconds (default 2).\n"
        "  --gate-yield=<mode>      on yield: abort (kill the in-flight solve, free the GPU in\n"
        "                           seconds; default) or finish (let the current solve complete).\n"
        "  --log-file=<path>        ALSO mirror all miner output (stderr) to <path>, on top of the\n"
        "                           console/journal. Appends; ANSI color is stripped from the file;\n"
        "                           the file is NOT rotated. env MATADOR_LOG_FILE; config key log_file.\n"
        "  --help                   this help\n"
        "\n"
        "Logging: LOG_LEVEL=debug|info|warn|error (default info). --log-file=<path> mirrors output to a file.\n"
        "Backend: BTX_MATMUL_BACKEND / BTX_MATMUL_PIPELINE_ASYNC / BTX_GPU_INPUTS etc.\n";
}

// Accepts "--key=value" and "--key value".
static bool ParseArgs(int argc, char* argv[], Config& cfg)
{
    int i = 1;   // loop index, shared with `take` so it can consume "--key value"
    auto take = [&](const std::string& a, const std::string& key, std::string& out) -> bool {
        const std::string pfx = "--" + key;
        if (a.rfind(pfx + "=", 0) == 0) { out = a.substr(pfx.size() + 1); return true; }  // --key=value
        if (a == pfx) {                                                                    // --key value
            if (i + 1 >= argc) {
                // "--rpcport" as the LAST arg used to return true with `out` UNSET;
                // the caller then ran std::stoi("") -> uncaught throw -> std::terminate
                // with no explanation. A missing value is a usage error: say so and
                // exit cleanly (startup, no threads yet -- same as --help).
                LOGE("[args] " << pfx << " requires a value");
                std::exit(1);
            }
            out = argv[++i];
            return true;
        }
        return false;
    };

    // Config path is special: discover it before applying env/CLI so the values
    // inside it can sit at the intended precedence layer (defaults < config < env
    // < CLI). A CLI --config only selects the file; individual CLI knobs below
    // still override settings read from that file.
    cfg.config_path = AutoConfigPath();
    cfg.config_path = GetEnvOr("MATADOR_CONFIG", cfg.config_path);
    for (int j = 1; j < argc; ++j) {
        std::string a = argv[j];
        if (a == "--help" || a == "-h") { PrintHelp(); std::exit(0); }
        if (a.rfind("--config=", 0) == 0) {
            cfg.config_path = a.substr(std::string("--config=").size());
        } else if (a == "--config" && j + 1 < argc) {
            cfg.config_path = argv[++j];
        }
    }
    if (!LoadConfigFile(cfg.config_path, cfg)) return false;

    // env fallbacks override config-file values
    cfg.mode          = GetEnvOr("MODE", cfg.mode);
    cfg.worker        = GetEnvOr("WORKER", cfg.worker);
    cfg.operator_label = GetEnvOr("OPERATOR_LABEL", cfg.operator_label);
    cfg.log_file_path = GetEnvOr("MATADOR_LOG_FILE", cfg.log_file_path);
    cfg.pool_pass     = GetEnvOr("POOL_PASS", cfg.pool_pass);
    if (const char* e = std::getenv("MATADOR_POOL_TLS_INSECURE")) {
        std::string ev = e;
        std::transform(ev.begin(), ev.end(), ev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.pool_tls_insecure = !(ev == "0" || ev == "false" || ev == "no" || ev == "off");
    }
    {   // SOCKS5 proxy: explicit SOCKS5=host:port wins; else honor ALL_PROXY=socks5://host:port
        if (const char* s = std::getenv("SOCKS5")) {
            if (!SplitHostPort(s, cfg.socks5_host, cfg.socks5_port))
                LOGW("[args] ignoring malformed SOCKS5 (want host:port): " << s);
        } else if (const char* ap = std::getenv("ALL_PROXY")) {
            if (SplitHostPort(ap, cfg.socks5_host, cfg.socks5_port))
                LOGI("[args] using ALL_PROXY as socks5 proxy " << cfg.socks5_host << ":" << cfg.socks5_port);
        }
        cfg.socks5_user = GetEnvOr("SOCKS5_USER", cfg.socks5_user);
        cfg.socks5_pass = GetEnvOr("SOCKS5_PASS", cfg.socks5_pass);
    }
    cfg.rpcconnect    = GetEnvOr("RPCCONNECT", cfg.rpcconnect);
    cfg.datadir       = GetEnvOr("DATADIR", cfg.datadir);
    cfg.rpccookiefile = GetEnvOr("RPCCOOKIEFILE", cfg.rpccookiefile);
    cfg.rpcuser       = GetEnvOr("RPCUSER", cfg.rpcuser);
    cfg.rpcpassword   = GetEnvOr("RPCPASSWORD", cfg.rpcpassword);
    cfg.payoutaddress = GetEnvOr("PAYOUTADDRESS", cfg.payoutaddress);
    cfg.chain         = GetEnvOr("CHAIN", cfg.chain);
    cfg.devaddress    = GetEnvOr("DEVADDRESS", cfg.devaddress);
    cfg.backend       = GetEnvOr("BTX_MATMUL_BACKEND", cfg.backend);
    if (const char* v = std::getenv("BTX_GPU_CLK_OFFSET")) { try { cfg.clk_offset = std::stoi(v); } catch (...) { LOGW("[args] ignoring non-integer BTX_GPU_CLK_OFFSET=" << v); } }
    if (const char* v = std::getenv("BTX_GPU_POWER_LIMIT")) { try { cfg.power_limit_w = std::stoi(v); } catch (...) { LOGW("[args] ignoring non-integer BTX_GPU_POWER_LIMIT=" << v); } }
    if (const char* v = std::getenv("BTX_GPU_LOCK_CLOCK")) { try { cfg.lock_gpu_clock = std::stoi(v); } catch (...) { LOGW("[args] ignoring non-integer BTX_GPU_LOCK_CLOCK=" << v); } }
    if (const char* v = std::getenv("BTX_GPU_FAN_PCT")) { try { cfg.fan_pct = std::stoi(v); } catch (...) { LOGW("[args] ignoring non-integer BTX_GPU_FAN_PCT=" << v); } }
    if (const char* v = std::getenv("BTX_GPU_MEM_CLK_OFFSET")) { try { cfg.mem_clk_offset = std::stoi(v); } catch (...) { LOGW("[args] ignoring non-integer BTX_GPU_MEM_CLK_OFFSET=" << v); } }
    if (const char* v = std::getenv("BTX_GPU_LOCK_MEM_CLOCK")) { try { cfg.lock_mem_clock = std::stoi(v); } catch (...) { LOGW("[args] ignoring non-integer BTX_GPU_LOCK_MEM_CLOCK=" << v); } }
    if (const char* g = std::getenv("MATADOR_GPUS")) cfg.gpu_devices = ParseGpuDeviceList(g);
    if (const char* e = std::getenv("MATADOR_GPU_WORKER_SUFFIX")) {
        std::string ev = e;
        std::transform(ev.begin(), ev.end(), ev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.gpu_worker_suffix = !(ev == "0" || ev == "false" || ev == "no" || ev == "off");
    }
    cfg.api_listen    = GetEnvOr("MATADOR_API_LISTEN", cfg.api_listen);
    if (const char* pool = std::getenv("POOL")) {
        if (!AddPoolEndpointList(cfg, pool, /*replace_existing=*/true, "args")) return false;
    }
    // Numeric env overrides go through the SafeParse/Env* helpers (top of this
    // header): a typo'd value logs a WARN and keeps the default instead of the old
    // raw std::stoi/stoull, which threw -> std::terminate -> systemd crash-loop.
    if (const char* p = std::getenv("RPCPORT")) {
        int port = 0;
        if (SafeParseInt(p, port)) { cfg.rpcport = port; cfg.rpcport_explicit = true; }
        else LOGW("[args] ignoring non-integer RPCPORT=\"" << p << "\" (keeping chain default)");
    }
    if (const char* m = std::getenv("MAXTRIES")) cfg.maxtries = EnvUint64("MAXTRIES", m, cfg.maxtries);
    // BTX_MATMUL_RC_HEIGHT is ALSO read directly by Consensus::Params (it self-pins at first use,
    // which is how probes/tests get it without a Config). Mirrored into cfg only so --print-config
    // and the startup banner report the value the solver will actually use.
    if (const char* h = std::getenv("BTX_MATMUL_RC_HEIGHT")) cfg.rc_height = std::max(0, EnvInt("BTX_MATMUL_RC_HEIGHT", h, cfg.rc_height));
    if (const char* d = std::getenv("DEVFEE"))   cfg.devfee = EnvInt("DEVFEE", d, cfg.devfee);
    if (const char* ak = std::getenv("BTX_ATTEST_KEY_FILE")) cfg.attest_key_file = ak;
    if (const char* ac = std::getenv("BTX_ATTEST_CONTEXT"))  cfg.attest_context = ac;
    if (const char* t = std::getenv("MATADOR_SOLVER_THREADS")) cfg.solver_threads = std::max(1, EnvInt("MATADOR_SOLVER_THREADS", t, cfg.solver_threads));
    if (const char* p = std::getenv("MATADOR_API_PORT")) {
        int port = 0;
        if (SafeParseInt(p, port)) { cfg.api_port = port; cfg.api_enabled = port > 0; }
        else LOGW("[args] ignoring non-integer MATADOR_API_PORT=\"" << p << "\" (keeping " << cfg.api_port << ")");
    }
    if (const char* e = std::getenv("MATADOR_API")) {
        std::string ev = e;
        std::transform(ev.begin(), ev.end(), ev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.api_enabled = (ev == "1" || ev == "true" || ev == "yes" || ev == "on");
    }
    if (const char* e = std::getenv("MATADOR_WATCHDOG")) {
        std::string ev = e;
        std::transform(ev.begin(), ev.end(), ev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.watchdog_enabled = !(ev == "0" || ev == "false" || ev == "no" || ev == "off");
    }
    if (const char* v = std::getenv("MATADOR_WATCHDOG_CHECK_S")) cfg.watchdog_check_s = std::max(1, EnvInt("MATADOR_WATCHDOG_CHECK_S", v, cfg.watchdog_check_s));
    if (const char* v = std::getenv("MATADOR_WATCHDOG_REJECT_STREAK")) cfg.watchdog_reject_streak = std::max(0, EnvInt("MATADOR_WATCHDOG_REJECT_STREAK", v, cfg.watchdog_reject_streak));
    if (const char* v = std::getenv("MATADOR_WATCHDOG_NO_SHARE_S")) cfg.watchdog_no_share_s = std::max(0, EnvInt("MATADOR_WATCHDOG_NO_SHARE_S", v, cfg.watchdog_no_share_s));
    if (const char* e = std::getenv("MATADOR_THERMAL")) {
        std::string ev = e;
        std::transform(ev.begin(), ev.end(), ev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.thermal_enabled = !(ev == "0" || ev == "false" || ev == "no" || ev == "off");
    }
    if (const char* v = std::getenv("MATADOR_THERMAL_WARN_TEMP_C")) cfg.thermal_warn_temp_c = std::max(0, EnvInt("MATADOR_THERMAL_WARN_TEMP_C", v, cfg.thermal_warn_temp_c));
    if (const char* v = std::getenv("MATADOR_THERMAL_CRITICAL_TEMP_C")) cfg.thermal_critical_temp_c = std::max(0, EnvInt("MATADOR_THERMAL_CRITICAL_TEMP_C", v, cfg.thermal_critical_temp_c));
    if (const char* v = std::getenv("MATADOR_THERMAL_WARN_POWER_W")) cfg.thermal_warn_power_w = std::max(0.0, EnvDouble("MATADOR_THERMAL_WARN_POWER_W", v, cfg.thermal_warn_power_w));
    if (const char* e = std::getenv("MATADOR_AUTO_UPDATE")) {
        std::string ev = e;
        std::transform(ev.begin(), ev.end(), ev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.auto_update = !(ev == "0" || ev == "false" || ev == "no" || ev == "off");
    }
    if (const char* v = std::getenv("MATADOR_UPDATE_INTERVAL_S")) cfg.update_interval_s = EnvInt("MATADOR_UPDATE_INTERVAL_S", v, cfg.update_interval_s);
    if (const char* v = std::getenv("MATADOR_UPDATE_JITTER_S")) cfg.update_jitter_s = std::max(0, EnvInt("MATADOR_UPDATE_JITTER_S", v, cfg.update_jitter_s));
    if (const char* v = std::getenv("MATADOR_MIN_VERSION_AGE_S")) cfg.min_version_age_s = std::max(0, EnvInt("MATADOR_MIN_VERSION_AGE_S", v, cfg.min_version_age_s));
    cfg.update_channel = GetEnvOr("MATADOR_UPDATE_CHANNEL", cfg.update_channel);
    cfg.fallback_pool = GetEnvOr("MATADOR_FALLBACK_POOL", cfg.fallback_pool);
    if (const char* v = std::getenv("MATADOR_FALLBACK_AFTER_S")) cfg.fallback_after_s = std::max(1, EnvInt("MATADOR_FALLBACK_AFTER_S", v, cfg.fallback_after_s));
    if (const char* v = std::getenv("MATADOR_SOLO_RECHECK_S")) cfg.solo_recheck_s = std::max(10, EnvInt("MATADOR_SOLO_RECHECK_S", v, cfg.solo_recheck_s));
    cfg.should_mine_command = GetEnvOr("MATADOR_SHOULD_MINE_COMMAND", cfg.should_mine_command);
    if (const char* v = std::getenv("MATADOR_SHOULD_MINE_INTERVAL")) cfg.should_mine_interval = std::max(1, EnvInt("MATADOR_SHOULD_MINE_INTERVAL", v, cfg.should_mine_interval));
    cfg.gate_yield = GetEnvOr("MATADOR_GATE_YIELD", cfg.gate_yield);
    if (const char* e = std::getenv("MATADOR_UPDATE_CHECK_ONLY")) {
        std::string ev = e;
        std::transform(ev.begin(), ev.end(), ev.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        cfg.update_check_only = (ev == "1" || ev == "true" || ev == "yes" || ev == "on");
    }

    bool cli_pool_seen = false;
    for (; i < argc; ++i) {
        std::string a = argv[i];
        std::string v;
        if (a == "--help" || a == "-h") { PrintHelp(); std::exit(0); }
        else if (take(a, "config", v))       cfg.config_path = v; // already loaded in pre-scan
        else if (take(a, "log-file", v))      cfg.log_file_path = v; // tee installed early in main() too

        else if (take(a, "mode", v))          cfg.mode = v;
        else if (take(a, "pool", v)) {
            if (!cli_pool_seen) {
                cfg.pools.clear();
                cfg.pool_host.clear();
                cfg.pool_port = 0;
                cli_pool_seen = true;
            }
            if (!AddPoolEndpointList(cfg, v, /*replace_existing=*/false, "args")) return false;
        }
        else if (take(a, "worker", v))        cfg.worker = v;
        else if (take(a, "operator-label", v)) cfg.operator_label = v;
        else if (take(a, "pool-pass", v))     cfg.pool_pass = v;
        else if (a == "--pool-tls-insecure")  cfg.pool_tls_insecure = true;
        else if (take(a, "socks5", v)) {
            if (!SplitHostPort(v, cfg.socks5_host, cfg.socks5_port)) {
                LOGE("[args] --socks5 must be host:port (or socks5://host:port): " << v); return false;
            }
        }
        else if (take(a, "socks5-user", v))   cfg.socks5_user = v;
        else if (take(a, "socks5-pass", v))   cfg.socks5_pass = v;
        else if (take(a, "rpcconnect", v))    cfg.rpcconnect = v;
        else if (take(a, "datadir", v))       cfg.datadir = v;
        else if (take(a, "rpccookiefile", v)) cfg.rpccookiefile = v;
        else if (take(a, "rpcuser", v))       cfg.rpcuser = v;
        else if (take(a, "rpcpassword", v))   cfg.rpcpassword = v;
        else if (take(a, "payoutaddress", v)) cfg.payoutaddress = v;
        else if (take(a, "clk-offset", v))    { try { cfg.clk_offset = std::stoi(v); } catch (...) { LOGE("[args] --clk-offset wants integer MHz, got: " << v); } }
        else if (take(a, "power-limit", v))   { try { cfg.power_limit_w = std::stoi(v); } catch (...) { LOGE("[args] --power-limit wants integer watts, got: " << v); } }
        else if (take(a, "lock-gpu-clock", v)){ try { cfg.lock_gpu_clock = std::stoi(v); } catch (...) { LOGE("[args] --lock-gpu-clock wants integer MHz, got: " << v); } }
        else if (take(a, "fan-pct", v))       { try { cfg.fan_pct = std::stoi(v); } catch (...) { LOGE("[args] --fan-pct wants integer percent (1..100), got: " << v); } }
        else if (take(a, "mem-clk-offset", v)){ try { cfg.mem_clk_offset = std::stoi(v); } catch (...) { LOGE("[args] --mem-clk-offset wants integer MHz, got: " << v); } }
        else if (take(a, "lock-mem-clock", v)){ try { cfg.lock_mem_clock = std::stoi(v); } catch (...) { LOGE("[args] --lock-mem-clock wants integer MHz, got: " << v); } }
        else if (take(a, "chain", v))         cfg.chain = v;
        // Numeric flags go through CliInt/CliUint64/CliDouble (top of this header):
        // a typo prints a usage error and exit(1)s instead of the old raw
        // std::stoi/stoull, which threw -> std::terminate with no explanation.
        else if (take(a, "rpcport", v))       { cfg.rpcport = CliInt("rpcport", v); cfg.rpcport_explicit = true; }
        else if (take(a, "maxtries", v))      cfg.maxtries = CliUint64("maxtries", v);
        else if (take(a, "rc-height", v))     cfg.rc_height = std::max(0, CliInt("rc-height", v));
        else if (take(a, "dev-fee", v))       cfg.devfee = CliInt("dev-fee", v);
        else if (take(a, "dev-address", v))   cfg.devaddress = v;
        else if (take(a, "backend", v))       cfg.backend = v;
        else if (take(a, "gpus", v))          cfg.gpu_devices = ParseGpuDeviceList(v);
        else if (take(a, "gpu-devices", v))   cfg.gpu_devices = ParseGpuDeviceList(v);
        else if (take(a, "solver-threads", v)) cfg.solver_threads = std::max(1, CliInt("solver-threads", v));
        else if (take(a, "api-listen", v))    cfg.api_listen = v;
        else if (take(a, "api-port", v))      { cfg.api_port = CliInt("api-port", v); cfg.api_enabled = cfg.api_port > 0; }
        else if (a == "--api")                cfg.api_enabled = true;
        else if (a == "--no-api")             cfg.api_enabled = false;
        else if (take(a, "watchdog-check-s", v)) cfg.watchdog_check_s = std::max(1, CliInt("watchdog-check-s", v));
        else if (take(a, "watchdog-reject-streak", v)) cfg.watchdog_reject_streak = std::max(0, CliInt("watchdog-reject-streak", v));
        else if (take(a, "watchdog-no-share-s", v)) cfg.watchdog_no_share_s = std::max(0, CliInt("watchdog-no-share-s", v));
        else if (a == "--watchdog")           cfg.watchdog_enabled = true;
        else if (a == "--no-watchdog")        cfg.watchdog_enabled = false;
        else if (take(a, "thermal-warn-temp-c", v)) cfg.thermal_warn_temp_c = std::max(0, CliInt("thermal-warn-temp-c", v));
        else if (take(a, "thermal-critical-temp-c", v)) cfg.thermal_critical_temp_c = std::max(0, CliInt("thermal-critical-temp-c", v));
        else if (take(a, "thermal-warn-power-w", v)) cfg.thermal_warn_power_w = std::max(0.0, CliDouble("thermal-warn-power-w", v));
        else if (a == "--thermal")            cfg.thermal_enabled = true;
        else if (a == "--no-thermal")         cfg.thermal_enabled = false;
        else if (a == "--gpu-suffix")         cfg.gpu_worker_suffix = true;
        else if (a == "--no-gpu-suffix")      cfg.gpu_worker_suffix = false;
        else if (a == "--no-update-check")    cfg.update_check = false;
        else if (a == "--no-auto-update")     cfg.auto_update = false;
        else if (take(a, "update-interval-s", v)) cfg.update_interval_s = CliInt("update-interval-s", v);
        else if (take(a, "update-jitter-s", v))   cfg.update_jitter_s = std::max(0, CliInt("update-jitter-s", v));
        else if (take(a, "min-version-age-s", v)) cfg.min_version_age_s = std::max(0, CliInt("min-version-age-s", v));
        else if (take(a, "update-channel", v))    cfg.update_channel = v;
        else if (a == "--update-check-only")      cfg.update_check_only = true;
        else if (a == "--poolcore")               cfg.poolcore = true;
        else if (take(a, "fallback-pool", v))     cfg.fallback_pool = v;
        else if (take(a, "fallback-after-s", v))  cfg.fallback_after_s = std::max(1, CliInt("fallback-after-s", v));
        else if (take(a, "solo-recheck-s", v))    cfg.solo_recheck_s = std::max(10, CliInt("solo-recheck-s", v));
        else if (take(a, "should-mine-command", v)) cfg.should_mine_command = v;
        else if (take(a, "should-mine-interval", v)) cfg.should_mine_interval = std::max(1, CliInt("should-mine-interval", v));
        else if (take(a, "gate-yield", v))        cfg.gate_yield = v;
        // RETIRED FLAGS: accept and ignore, never hard-fail.
        // An unknown argument aborts startup, and on HiveOS these live in the flight
        // sheet's "Extra config arguments" where an operator cannot see the error. A rig
        // carrying --no-overlap from a v3-era sheet would simply stop mining on adopt,
        // which is a far worse outcome than running without a knob that no longer does
        // anything. Warn loudly instead so the sheet still gets cleaned up.
        else if (a == "--overlap" || a == "--no-overlap") {
            LOGW("[args] " << a << " is RETIRED and ignored: it drove the v3 prepare/digest "
                 "overlap, which no longer exists. Remove it from your config or flight sheet.");
        }
        else if (take(a, "hip-solver", v)) {
            LOGW("[args] --hip-solver is RETIRED and ignored: the external AMD/HIP sidecar spoke "
                 "the v3 protocol and there is no ENC_RC AMD solver. Remove it from your config "
                 "or flight sheet.");
        }
        else { LOGE("[args] unknown argument: " << a); return false; }
    }
    if (cfg.gpu_devices.empty() && std::getenv("MATADOR_GPUS") != nullptr) {
        LOGE("[args] MATADOR_GPUS/--gpus must include at least one device id when set");
        return false;
    }
    if (const char* child_worker = std::getenv("MATADOR_MULTI_GPU_CHILD_WORKER")) {
        if (child_worker[0] != '\0') cfg.worker = child_worker;
    }
    // Multi-GPU children share ONE operator_label (the supervisor's base label) so minebtx
    // groups the rig's cards as a single "hive", while each child keeps its own distinct
    // "<addr>.<worker>-gpuN" stratum user for per-GPU work-split + share credit.
    if (const char* op_label = std::getenv("MATADOR_MULTI_GPU_OPERATOR_LABEL")) {
        if (op_label[0] != '\0') cfg.operator_label = op_label;
    }
    if (const char* child_api_port = std::getenv("MATADOR_MULTI_GPU_CHILD_API_PORT")) {
        // Set by our own multi-GPU supervisor, so garbage means operator tampering:
        // warn and keep the inherited api settings rather than dying at startup.
        int port = 0;
        if (SafeParseInt(child_api_port, port)) { cfg.api_port = port; cfg.api_enabled = port > 0; }
        else LOGW("[args] ignoring non-integer MATADOR_MULTI_GPU_CHILD_API_PORT=\"" << child_api_port << "\"");
    }
    return true;
}

// Reads the "user:password" auth string: cookie file (preferred) or rpcuser/pass.
static std::string ResolveAuth(const Config& cfg)
{
    std::string cookie_path = cfg.rpccookiefile;
    if (cookie_path.empty() && !cfg.datadir.empty()) {
        cookie_path = cfg.datadir + "/.cookie";
    }
    if (!cookie_path.empty()) {
        std::ifstream f(cookie_path);
        if (f) {
            std::string line;
            std::getline(f, line);
            if (!line.empty()) {
                LOGI("[auth] using cookie file=" << cookie_path);  // path ok; contents NOT logged
                return line;  // already "__cookie__:<hex>"
            }
            LOGW("[auth] cookie file present but empty: " << cookie_path);
        } else if (!cfg.rpccookiefile.empty()) {
            // explicit cookie requested but unreadable -> fatal
            LOGE("[auth] cannot read cookie file: " << cookie_path);
            throw std::runtime_error("cannot read rpccookiefile: " + cookie_path);
        }
    }
    if (!cfg.rpcuser.empty()) {
        LOGI("[auth] using rpcuser/rpcpassword (user=" << cfg.rpcuser << ")");  // pass NOT logged
        return cfg.rpcuser + ":" + cfg.rpcpassword;
    }
    throw std::runtime_error("no RPC auth: provide --rpccookiefile/--datadir or --rpcuser/--rpcpassword");
}

#endif  // MATADOR_CONFIG_PARSE_HELPERS_ONLY
