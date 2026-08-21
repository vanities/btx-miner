// matador-miner: POOL MODE -- the minebtx/dexbtx (shib) stratum client. Connects,
// subscribes+authorizes, runs the reader thread, parses mining.notify jobs
// (StratumJob), and submits shares. SECTION of the single miner translation unit
// (uses Config, Stats, mlog, and the live g_pool_* telemetry atomics defined in
// matador-miner.cpp). Extracted verbatim.
#pragma once
#include "pool_socket_opts.h"  // ApplyPoolSocketOpts (keepalive + TCP_NODELAY); unit-tested
#include "gpu_telemetry.h"     // QueryGpuTelemetry: fork-free NVML numerics for the metrics heartbeat

#include <univalue.h>          // vendored (core/vendor/univalue/include on every build's -I path)

#include <algorithm>           // std::transform (case-fold of a pool's reject reason)
#include <cctype>
#include <functional>          // JobObserver hook (RC activation accounting; see below)
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <fcntl.h>             // O_NONBLOCK (bounded connect), FD_CLOEXEC (survive auto-update execv)
#include <poll.h>              // poll() deadline on the non-blocking connect

// ---------------------------------------------------------------------------
// Pure wire-parsing helpers -- no Config/Stats/mlog deps. Everything a pool
// SENDS must be assumed hostile-or-buggy: vendored UniValue's typed getters
// (get_str/get_bool/getInt) THROW std::runtime_error on any type mismatch
// (getInt<> also throws on floats like 10.0), and an uncaught throw in the
// reader thread std::terminates the whole miner -> systemd restart -> the pool
// replays the same line -> crash loop. These helpers never throw (or their
// callers catch). Unit-tested in harness/stratum_dispatch_test.cpp, which
// defines MATADOR_STRATUM_PARSE_HELPERS_ONLY to compile JUST this block
// standalone (the client class below needs the whole miner TU).
// ---------------------------------------------------------------------------

// Tiny JSON string escaper (quotes + backslashes) for the few values WE emit
// (user/pass/worker, operator_label, capability echo). A worker name or pool
// password containing '"' or '\' must not be able to break out of its JSON
// string in the handshake we send.
static std::string JsonEscapeMinimal(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

// Parse a single uint from a UniValue that may be a JSON number or a hex/dec
// string. Stratum mixes int and string encodings across pools, so be lenient.
// NOTE: still THROWS on garbage ("abc", float numbers) BY DESIGN: a garbage
// notify field must fail the WHOLE job parse -- the miner keeps the previous
// job and the pool re-notifies -- rather than silently become 0 and corrupt
// the seed/digest into rejected shares. Callers wrap in try/catch
// (ParseNotifyParams / ParseLuckyPoolJob / ParseSubscribeResult).
static uint64_t AsUint(const UniValue& v, int base = 10)
{
    if (v.isNum()) return static_cast<uint64_t>(v.getInt<int64_t>());
    if (v.isStr()) {
        const std::string& s = v.get_str();
        if (s.empty()) return 0;
        // auto-detect 0x; otherwise honor the requested base
        if (s.size() > 2 && (s[0] == '0') && (s[1] == 'x' || s[1] == 'X'))
            return std::stoull(s.substr(2), nullptr, 16);
        return std::stoull(s, nullptr, base);
    }
    return 0;
}

// Tolerant JSON-RPC response-id -> uint64. Pools are sloppy with ids ("id":"3"
// from login-dialect pools, "id":10.0 from float-happy JSON emitters) and
// getInt<int64_t>() throws on BOTH. Returns false -- never throws -- when the
// id is unusable, so DispatchLine can log-and-drop the one response instead of
// the reader thread dying on it.
static bool StratumIdToUint(const UniValue& v, uint64_t& out)
{
    try {
        if (v.isNum()) {
            // getInt<> rejects non-integral numbers (10.0): fall back to a
            // truncating get_real so a float id still matches its pending submit.
            try { out = static_cast<uint64_t>(v.getInt<int64_t>()); return true; }
            catch (...) { out = static_cast<uint64_t>(v.get_real()); return true; }
        }
        if (v.isStr()) {
            const std::string& s = v.get_str();
            if (s.empty()) return false;
            size_t idx = 0;
            const unsigned long long parsed = std::stoull(s, &idx, 10);
            if (idx != s.size()) return false;   // "12x" is not an id
            out = static_cast<uint64_t>(parsed);
            return true;
        }
    } catch (...) {}                             // out-of-range / non-numeric string
    return false;
}

// Tolerant stratum error extraction: err arrives as [code,"msg"], {"message":..},
// a bare string, a bare number, or garbage -- and "message" itself may be
// non-string ({"message":42} used to get_str()-throw and kill the reader).
// NEVER throws; degrades to the raw JSON so the operator still sees SOMETHING.
static std::string StratumErrorMessage(const UniValue& err)
{
    try {
        if (err.isStr()) return err.get_str();
        if (err.isArray() && err.size() >= 2 && err[1].isStr()) return err[1].get_str();
        if (err.isObject() && err.exists("message") && err["message"].isStr()) return err["message"].get_str();
        return err.write();
    } catch (...) { return "(unparseable error)"; }
}

// Split raw recv() bytes into protocol lines. Appends `data[0..n)` to `carry`
// (the partial line spanning recv boundaries) and invokes on_line(line) for
// every complete '\n'-terminated line; '\r' is stripped ANYWHERE (not just
// before '\n') and blank lines are skipped -- byte-identical semantics to the
// old one-recv-per-BYTE reader loop, which this replaces (that loop cost
// ~600-1000 recv syscalls per mining.notify, on the job-switch latency path).
template <typename OnLine>
static void FeedRecvBytes(std::string& carry, const char* data, size_t n, OnLine&& on_line)
{
    for (size_t k = 0; k < n; ++k) {
        const char ch = data[k];
        if (ch == '\n') {
            if (!carry.empty()) {
                on_line(carry);
                carry.clear();
            }
            continue;
        }
        if (ch != '\r') carry.push_back(ch);
    }
}

// nvidia-smi CSV column -> JSON number or null. nvidia-smi prints "[N/A]" (or
// "N/A") for fields a GPU/driver cannot report; interpolated bare that is
// INVALID JSON ("power_w":[N/A]) and the pool drops the whole metrics report.
// Mirrors status_api.h's ParseDoubleFinite+JsonNumberOrNull (distinct name on
// purpose: both headers land in the same miner TU). The column passes through
// VERBATIM when it parses as one finite number; anything else becomes null.
static std::string CsvNumberOrNull(const std::string& col)
{
    char* end = nullptr;
    const double parsed = std::strtod(col.c_str(), &end);
    if (end == col.c_str() || !std::isfinite(parsed)) return "null";
    while (*end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) return "null";
        ++end;
    }
    return col;
}

// Does a pool's share-reject reason mean THIS SESSION's authorization is gone?
// Byron says verbatim "unauthorized: send mining.authorize first" (probed 2026-08-21);
// other stratum servers phrase the same state as "not authorized" / "login first".
// It matters because a session in this state is a total loss that looks alive: Byron
// keeps DISPATCHING mining.notify jobs to unauthorized sessions (also probed), so the
// miner keeps solving and every share bounces. Before this classifier existed the only
// exit was the watchdog's 20-reject streak, ~5-10 minutes of burned work per cycle --
// and a reject-with-this-reason now requests an immediate reconnect instead (which also
// advances the pool failover index, so a deterministic auth failure rotates pools
// rather than looping). Substring match on a lowered copy: pool wording is free text.
static bool ReasonIsAuthFailure(const std::string& reason)
{
    std::string r;
    r.reserve(reason.size());
    for (char c : reason) r.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return r.find("unauthorized") != std::string::npos ||
           r.find("unauthorised") != std::string::npos ||
           r.find("not authorized") != std::string::npos ||
           r.find("not authorised") != std::string::npos ||
           r.find("authorize first") != std::string::npos ||
           r.find("login first") != std::string::npos;
}

#ifndef MATADOR_STRATUM_PARSE_HELPERS_ONLY

#include <openssl/err.h>
#include <openssl/ssl.h>       // ssl://, tls:// pool schemes (see ParsePoolEndpoint / SetUseTls)
#include <openssl/x509v3.h>

// ===========================================================================
// 5b. POOL MODE: minebtx/dexbtx (shib) stratum client. POOL-SPECIFIC.
//
//     This speaks shib's dexbtx/minebtx pool protocol specifically (the v18
//     "pre_hash_block_tier_v18" tier), reverse-engineered from the live pool +
//     thekillsquad007/btx-nvidia-miner + dexbtx/minebtx. The capability handshake,
//     the matmul mining.notify extension, worker.report_metrics, and the hardware
//     json schema are all dexbtx-specific - a different BTX pool may use a different
//     protocol and would need its own client path. It mirrors the dexbtx protocol
//     EXACTLY:
//       - mining.subscribe ["matador-miner/<ver>", {protocol_compliant:[...],
//         hardware:{...}, operator_label, session_id}]
//       - mining.authorize ["<addr>.<worker>", "<pass>"]
//       - mining.notify  (9 params, see ParseNotifyLine)
//       - mining.submit  ["<addr>.<worker>", job_id, extranonce2, ntime(8h), nonce64(16h)]
//
//     SHARES are solved with the SAME SolveMatMul the solo loop uses (so our
//     overlap +~31% applies), but with a share_target_override (the pool's
//     easier per-share target) and parent_median_time_past from the job object.
//
//     ENDIANNESS (see notes at ParseNotifyLine): prevhash / merkleroot / target
//     are 32-byte hex in RPC/display order and are consumed with uint256::FromHex
//     exactly like the solo AssembleBlock consumes GBT's previousblockhash
//     (matador-miner.cpp:408). nbits is a compact-bits hex word parsed with
//     std::stoul(...,16), exactly like solo parses GBT "bits" (line 409).
// ===========================================================================

// One stratum job (the 9 mining.notify params + the v18 job object).
struct StratumJob {
    std::string job_id;            // [0]
    uint32_t    version{0};        // [1] header nVersion
    std::string prevhash_hex;      // [2] 64-hex, RPC/display order
    std::string merkleroot_hex;    // [3] 64-hex, RPC/display order
    uint32_t    ntime{0};          // [4]
    uint32_t    nbits{0};          // [5] compact bits
    std::string target_hex;        // [6] 64-hex share target, RPC/display order
    bool        clean_jobs{false}; // [7]
    // [8] job object:
    int32_t     block_height{0};
    int64_t     parent_mtp{0};
    uint64_t    nonce64_start{0};
    uint16_t    matmul_n{0};       // matmul dimension for this job
    // Optional algorithm/profile label ("enc-rc-v47", "btx-live", ...). The poolcore contract has
    // carried this since v0.9.2; on stratum it is what lets a pool announce RC activation without
    // us compiling a height in (see MaybeLatchRCFromJob). Absent on every pool today -> empty.
    std::string profile;
    uint32_t    nonce_bits{0};     // LuckyPool: bits we scan under the pool's noncePrefix (submit low nonce_bits)
    bool        valid{false};      // set true once a full notify is parsed
};

// Observer invoked once per PARSED JOB, from the reader thread, before the job is published to the
// solve loop. Exists so job-stream accounting (RC activation detection) sees every job the pool
// sent rather than the subset a busy solve loop happens to pick up. Kept as a hook rather than a
// direct call so this header stays free of consensus/solver dependencies. Unset = no-op.
inline std::function<void(const StratumJob&)>& JobObserver()
{
    static std::function<void(const StratumJob&)> f;
    return f;
}

static std::string ToLowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

// Parse the 9-param mining.notify array into a StratumJob. Returns false if the
// shape is wrong. Field order/types per the confirmed-live minebtx v18 protocol.
static bool ParseNotifyParams(const UniValue& p, StratumJob& out)
{
    if (!p.isArray() || p.size() < 8) return false;
    try {
        out.job_id        = p[0].get_str();
        out.version       = static_cast<uint32_t>(AsUint(p[1]));
        out.prevhash_hex  = p[2].get_str();
        out.merkleroot_hex= p[3].get_str();
        out.ntime         = static_cast<uint32_t>(AsUint(p[4]));
        // nbits: compact-bits hex word, same convention as solo GBT "bits".
        out.nbits         = static_cast<uint32_t>(std::stoul(p[5].get_str(), nullptr, 16));
        out.target_hex    = p[6].get_str();
        out.clean_jobs    = p[7].isBool() ? p[7].get_bool()
                                          : (p[7].isStr() ? (p[7].get_str() == "true") : false);
        if (p.size() >= 9 && p[8].isObject()) {
            const UniValue& jo = p[8];
            if (jo.exists("block_height"))  out.block_height  = static_cast<int32_t>(AsUint(jo["block_height"]));
            // minebtx sends "parent_mtp"; byron-pool sends "parent_median_time_past" - same
            // v3 seed input (feeds SolveMatMul seed derivation). Read whichever the pool
            // provides; a missing value here silently corrupts the digest -> rejected shares.
            if (jo.exists("parent_mtp"))
                out.parent_mtp = static_cast<int64_t>(AsUint(jo["parent_mtp"]));
            else if (jo.exists("parent_median_time_past"))
                out.parent_mtp = static_cast<int64_t>(AsUint(jo["parent_median_time_past"]));
            if (jo.exists("nonce64_start")) out.nonce64_start = AsUint(jo["nonce64_start"]);
            if (jo.exists("matmul_n"))      out.matmul_n      = static_cast<uint16_t>(AsUint(jo["matmul_n"]));
            // Optional profile/algo label. "profile" is the poolcore spelling; accept "algo" too so
            // a pool that already labels jobs that way needs no server change.
            if (jo.exists("profile") && jo["profile"].isStr())   out.profile = jo["profile"].get_str();
            else if (jo.exists("algo") && jo["algo"].isStr())    out.profile = jo["algo"].get_str();
        }
        out.valid = true;
        return true;
    } catch (const std::exception& e) {
        LOGW("[stratum] notify parse error: " << e.what());
        return false;
    }
}

// Parse a LuckyPool "job" params object into a StratumJob. Same consensus fields as the minebtx
// notify, just named (nVersion/prevHash/merkleRoot/nTime/nBits/matmulDim/shareTarget/...) and
// delivered as one JSON object. The nonce lane arrives as (noncePrefix, nonceBits): the prefix is
// the high (64-nonceBits) bits of the 64-bit nonce; we scan the low nonceBits window under it.
static bool ParseLuckyPoolJob(const UniValue& p, StratumJob& out)
{
    try {
        if (!p.isObject() || !p.exists("jobId")) return false;
        out.job_id         = p["jobId"].get_str();
        out.version        = static_cast<uint32_t>(AsUint(p["nVersion"]));
        out.prevhash_hex   = p["prevHash"].get_str();
        out.merkleroot_hex = p["merkleRoot"].get_str();
        out.ntime          = static_cast<uint32_t>(AsUint(p["nTime"]));
        out.nbits          = static_cast<uint32_t>(std::stoul(p["nBits"].get_str(), nullptr, 16));
        out.target_hex     = p["shareTarget"].get_str();
        out.clean_jobs     = p.exists("cleanJobs") && p["cleanJobs"].isBool() ? p["cleanJobs"].get_bool() : true;
        out.block_height   = static_cast<int32_t>(AsUint(p["height"]));
        out.parent_mtp     = p.exists("parentMtp") ? static_cast<int64_t>(AsUint(p["parentMtp"])) : 0;
        out.matmul_n       = static_cast<uint16_t>(AsUint(p["matmulDim"]));
        if (p.exists("profile") && p["profile"].isStr())   out.profile = p["profile"].get_str();
        else if (p.exists("algo") && p["algo"].isStr())    out.profile = p["algo"].get_str();
        const uint32_t nonce_bits = p.exists("nonceBits") ? static_cast<uint32_t>(AsUint(p["nonceBits"])) : 40U;
        const uint64_t prefix = p.exists("noncePrefix") ? std::stoull(p["noncePrefix"].get_str(), nullptr, 10) : 0ULL;
        out.nonce_bits     = nonce_bits;
        out.nonce64_start  = (nonce_bits < 64U) ? (prefix << nonce_bits) : 0ULL;
        out.valid = true;
        return true;
    } catch (const std::exception& e) {
        LOGW("[pool-lucky] job parse error: " << e.what());
        return false;
    }
}

// The stratum client: connects, subscribes+authorizes, runs a reader thread that
// updates the current job (mutex-guarded) and signals abort_flag on every new
// job, and exposes submit_share. The SOLVE happens in the pool main loop below
// (RunPoolLoop), not here, so it can reuse SolveMatMul exactly like solo.
class StratumClient {
public:
    StratumClient(std::string host, int port, std::string user, std::string pass,
                  Stats* stats = nullptr)
        : m_host(std::move(host)), m_port(port),
          m_user(std::move(user)), m_pass(std::move(pass)), m_stats(stats)
    {
        // random hex session id (32 hex chars). Not security-sensitive; just an id.
        std::mt19937_64 rng(static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        std::ostringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(16) << rng()
                       << std::setw(16) << rng();
        m_session_id = ss.str();

        // Work-staleness watchdog window: force a reconnect if no mining.notify
        // arrives for this long (pools push work every ~5-30s). Env-overridable.
        const char* env = std::getenv("MATADOR_POOL_STALL_TIMEOUT_S");
        if (env != nullptr && env[0] != '\0') {
            const int s = std::atoi(env);
            if (s >= 10 && s <= 3600) m_stall_timeout_ms = static_cast<int64_t>(s) * 1000;
        }
    }

    // Milliseconds on the monotonic clock (stall watchdog timestamps).
    static int64_t NowMs()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Fold one measured request->response round trip into the g_pool_latency_ms EMA
    // (first sample taken verbatim). Telemetry only -- read by [stats] and /summary.
    static void RecordPoolLatency(int64_t rtt_ms)
    {
        if (rtt_ms < 0) return;
        const double prev = g_pool_latency_ms.load(std::memory_order_relaxed);
        g_pool_latency_ms.store(prev <= 0.0 ? static_cast<double>(rtt_ms)
                                            : 0.7 * prev + 0.3 * static_cast<double>(rtt_ms),
                                std::memory_order_relaxed);
    }

    ~StratumClient() { Disconnect(); if (m_ssl_ctx != nullptr) SSL_CTX_free(m_ssl_ctx); }

    // Connect the socket + perform the subscribe/authorize handshake. Throws on
    // failure (caller reconnects with backoff). Spawns the reader thread.
    void ConnectAndHandshake()
    {
        Connect();                       // throws on socket failure
        m_auth_kick.store(false);        // fresh connection, fresh auth-escalation debounce
        m_running.store(true);
        m_last_notify_ms.store(NowMs()); // start the stall clock at connect (handshake grace)
        if (m_stats) m_stats->last_notify_ms.store(MonoMs());
        g_pool_connected_ms.store(MonoMs(), std::memory_order_relaxed);   // [stats] conn= age
        m_reader  = std::thread([this]() { ReaderLoop(); });
        m_metrics = std::thread([this]() { MetricsLoop(); });   // worker.report_metrics heartbeat

        // Send the stratum handshake as a probe. Stratum pools (minebtx/Byron) accept it; a
        // JSON-RPC login pool (LuckyPool/ninjaraider) rejects mining.subscribe with -32601, and
        // DispatchLine then switches this connection to the login dialect. So the dialect is
        // AUTO-DETECTED per connection -- no hostname/config guessing needed.
        m_handshake_sent_ms.store(MonoMs());   // id=1 response -> first latency sample
        SendSubscribe();
        SendAuthorize();
        LOGI("[" << m_tag << "] handshake sent (subscribe+authorize probe) host=" << m_host << ":" << m_port
             << " user=" << m_user << " session=" << Short(m_session_id));
    }

    // LuckyPool JSON-RPC 2.0 login: {"method":"login","params":{"login":<addr.worker>,"pass":..,"agent":..}}
    // -> {"result":true}; the pool then pushes "job" notifications.
    void SendLogin()
    {
        // LuckyPool wants SEPARATE address + worker fields (NOT a combined "login" string) --
        // matching the reference miner's wire format. Jamming "<addr>.<worker>" into one field
        // registers a malformed address and share-tracking breaks after the first submit.
        // m_user is "<address>.<worker>"; BTX addresses contain no dot, so split on the first.
        std::string addr = m_user, worker;
        const auto dot = m_user.find('.');
        if (dot != std::string::npos) { addr = m_user.substr(0, dot); worker = m_user.substr(dot + 1); }
        std::ostringstream ss;
        // Escape the operator-supplied fields: a quote/backslash in the address or
        // worker must not break out of its JSON string (same treatment operator_label
        // already gets in SendSubscribe).
        ss << "{\"id\":3,\"method\":\"login\",\"params\":{\"address\":\"" << JsonEscapeMinimal(addr)
           << "\",\"agent\":\"matador-miner/" << MATADOR_MINER_VERSION << "\"";
        if (!worker.empty()) ss << ",\"worker\":\"" << JsonEscapeMinimal(worker) << "\"";
        ss << "}}\n";
        SendLine(ss.str());
    }

    // The dev-fee side session must NEVER bounce the primary: both clients share one
    // Stats, and watchdog_reconnect_requested tears down the whole pool loop. The
    // primary keeps the default (escalate); the dev client opts out at construction.
    void SetAuthEscalation(bool on) { m_auth_escalate = on; }

    // Escalate an authorization failure -- a rejected authorize/login, or a share
    // bounced with an auth-shaped reason -- into the reconnect+failover path the solve
    // loop already honours (disconnect, ADVANCE the pool index, backoff, reconnect).
    // Debounced per connection: when the pool drops our session, every in-flight
    // submit usually bounces at once, and one kick is enough.
    void RequestAuthReconnect(const char* why)
    {
        if (m_auth_kick.exchange(true)) return;
        if (!m_auth_escalate || m_stats == nullptr) {
            LOGW("[" << m_tag << "] " << why << " -- session not authorized at the pool;"
                 " side session will retry on its own (primary mining unaffected)");
            return;
        }
        LOGW("[" << m_tag << "] " << why << " -- this session is NOT authorized at the pool,"
             " so every further share here is a guaranteed reject while jobs keep flowing."
             " Requesting immediate reconnect/failover.");
        m_stats->watchdog_reconnect_requested.store(true);
    }

    void Disconnect()
    {
        m_running.store(false);
        g_pool_connected_ms.store(0, std::memory_order_relaxed);   // [stats] shows conn=down until reconnect
        // Teardown ORDER matters: shutdown() first (wakes the reader/metrics threads
        // blocked in recv/send on this socket), then JOIN them, and only then
        // close(). The old close-before-join freed the fd NUMBER while the reader
        // could still be entering recv() on it -- if any other subsystem (status API
        // accept, popen, curl, a reconnect) reopened that number in the gap, the
        // reader would quietly recv() on a stranger's connection. Keeping the fd
        // open (merely shut down) until both threads are joined makes that
        // impossible; m_sock is atomic because the reader polls it unsynchronized.
        const int fd = m_sock.load();
        if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
        if (m_reader.joinable()) m_reader.join();
        if (m_metrics.joinable()) m_metrics.join();
        // TLS teardown AFTER the threads join (they may still be inside SSL_read/SSL_write on
        // m_ssl) and BEFORE close(fd) (SSL_shutdown/SSL_free touch the fd one last time).
        if (m_ssl != nullptr) {
            SSL_shutdown(m_ssl);   // best-effort close_notify; ignore the result (fd is closing regardless)
            SSL_free(m_ssl);
            m_ssl = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            m_sock.store(-1);
        }
    }

    // Periodic worker.report_metrics heartbeat (~60s), notify-style. The pool's
    // dashboard reads these for the per-worker hashrate / GPU columns (matches
    // dexbtx_miner's collect_runtime_metrics payload). Best-effort: no response
    // expected; GPU telemetry via nvidia-smi, "[]" if unavailable. Payouts are by
    // accepted shares, so this is display-only - but without it the worker shows
    // "- - -" on the UI.
    void MetricsLoop()
    {
        uint64_t prev_eps = 0;
        auto prev_t = std::chrono::steady_clock::now();
        auto naptick = [this](int ticks) {  // sleep `ticks`*0.5s, but bail fast on stop
            for (int i = 0; i < ticks && m_running.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        };
        naptick(10);   // ~5s initial jitter
        while (m_running.load()) {
            naptick(120);   // ~60s
            if (!m_running.load()) break;
            const auto now = std::chrono::steady_clock::now();
            const auto p = ProbeMatMulSolvePipelineStats();
            const double dt = std::chrono::duration<double>(now - prev_t).count();
            // solver_nps is the pool-facing throughput field. On the v4 path the unit is
            // the ENC_RC episode, not the nonce; it is fractional (~1.2/s on a 5090), so it
            // is reported scaled by 1000 to survive the integer cast below.
            const uint64_t eps = p.rc_episodes;
            const double nps = (eps >= prev_eps && dt > 0.0) ? (eps - prev_eps) * 1000.0 / dt : 0.0;
            prev_eps = eps; prev_t = now;
            const uint64_t shares = m_stats ? (m_stats->accepted.load() + m_stats->rejected.load()) : 0;
            const int64_t ts = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::ostringstream ss;
            ss << "{\"method\":\"worker.report_metrics\",\"params\":[{"
               << "\"session_id\":\"" << m_session_id << "\","
               << "\"timestamp\":" << ts << ","
               << "\"cpu_util_pct\":0,\"ram_gb_used\":0,"
               << "\"gpus\":" << GpuRuntimeJson() << ","
               << "\"solver_nps\":" << static_cast<uint64_t>(nps) << ","
               << "\"shares_session_total\":" << shares << ","
               << "\"wrapper_version\":\"" MATADOR_MINER_VERSION "\","
               << "\"solver_sha256\":\"\","
               << "\"solver_backend\":\"cuda\""
               << "}]}\n";
            try { SendLine(ss.str()); }
            catch (...) { break; }
            LOGI("[stratum] worker.report_metrics sent: nps=" << static_cast<uint64_t>(nps)
                 << " shares=" << shares);
        }
    }

    // Best-effort runtime GPU telemetry as a JSON array; "[]" if nvidia-smi is
    // unavailable. Schema matches dexbtx _gpu_runtime() EXACTLY: the pool keys each
    // entry to a worker GPU by gpu_uuid (must match the static hardware json sent in
    // subscribe), so without gpu_uuid the dashboard shows blank Util/Watts.
    //
    // Fork-free steady state (same rationale as status_api.h's NVML fast path):
    // popen("nvidia-smi") fork()s the miner -- a process mapping a multi-GiB CUDA
    // address space -- and BOTH stratum sessions (primary + warm dev-fee) ran it
    // every ~60s, freezing the solve thread for the fork's page-table churn. The
    // dlopen'd NVML getters answer the same three counters in microseconds,
    // in-process. gpu_uuid (which NVML's numeric handle cannot supply) is latched
    // from the FIRST forked read and reused forever; multi-GPU / no-NVML boxes
    // keep the forked path (NVML handle is device-0-only).
    static std::string GpuRuntimeJson()
    {
        struct UuidLatch {
            std::mutex mu;
            bool latched{false};
            std::vector<std::string> uuids;
        };
        static UuidLatch latch;
        {
            std::lock_guard<std::mutex> lk(latch.mu);
            if (latch.latched && latch.uuids.size() == 1) {
                const GpuTelemetry t = QueryGpuTelemetry();
                if (t.ok) {
                    std::ostringstream one;
                    one << "[{\"gpu_uuid\":\"" << latch.uuids[0]
                        << "\",\"util_pct\":" << t.util_pct
                        << ",\"power_w\":" << (t.pow_w > 0 ? std::to_string(t.pow_w) : std::string("null"))
                        << ",\"temp_c\":" << (t.temp_c > 0 ? std::to_string(t.temp_c) : std::string("null"))
                        << "}]";
                    return one.str();
                }
            }
        }
        FILE* f = popen("nvidia-smi --query-gpu=uuid,utilization.gpu,power.draw,temperature.gpu "
                        "--format=csv,noheader,nounits 2>/dev/null", "r");
        if (f == nullptr) return "[]";
        std::ostringstream arr; arr << "[";
        bool first = true; char line[512];
        std::vector<std::string> seen_uuids;
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            std::vector<std::string> col; std::stringstream ls(line); std::string tok;
            while (std::getline(ls, tok, ',')) {
                const size_t a = tok.find_first_not_of(" \t\r\n");
                const size_t b = tok.find_last_not_of(" \t\r\n");
                col.push_back(a == std::string::npos ? std::string() : tok.substr(a, b - a + 1));
            }
            if (col.size() < 4) continue;
            if (!first) arr << ",";
            first = false;
            seen_uuids.push_back(col[0]);
            // Numeric columns via CsvNumberOrNull (top of this header): nvidia-smi
            // prints "[N/A]" for fields it cannot read, and interpolating that bare
            // produced invalid JSON ("power_w":[N/A]) -- the pool then drops the whole
            // metrics report and the worker shows blank Util/Watts on the dashboard.
            arr << "{\"gpu_uuid\":\"" << col[0] << "\",\"util_pct\":" << CsvNumberOrNull(col[1])
                << ",\"power_w\":" << CsvNumberOrNull(col[2])
                << ",\"temp_c\":" << CsvNumberOrNull(col[3]) << "}";
        }
        pclose(f);
        arr << "]";
        if (!seen_uuids.empty()) {
            // A successful forked read doubles as the one-time identity latch;
            // single-GPU boxes never fork for metrics again after this.
            std::lock_guard<std::mutex> lk(latch.mu);
            if (!latch.latched) {
                latch.uuids = std::move(seen_uuids);
                latch.latched = true;
            }
        }
        return arr.str();
    }

    // Static hardware json for mining.subscribe. The dashboard ties runtime metrics
    // (GpuRuntimeJson) to this worker's GPUs by gpu_uuid + uses model/cpu/backend for
    // its tuning recommendations. Mirrors dexbtx collect_static_hardware's key fields.
    static std::string BuildHardwareJson()
    {
        std::ostringstream gpus; gpus << "[";
        std::string driver;
        FILE* f = popen("nvidia-smi --query-gpu=uuid,name,memory.total,driver_version "
                        "--format=csv,noheader,nounits 2>/dev/null", "r");
        if (f != nullptr) {
            bool first = true; char line[512];
            while (std::fgets(line, sizeof(line), f) != nullptr) {
                std::vector<std::string> col; std::stringstream ls(line); std::string tok;
                while (std::getline(ls, tok, ',')) {
                    const size_t a = tok.find_first_not_of(" \t\r\n");
                    const size_t b = tok.find_last_not_of(" \t\r\n");
                    col.push_back(a == std::string::npos ? std::string() : tok.substr(a, b - a + 1));
                }
                if (col.size() < 4) continue;
                if (driver.empty()) driver = col[3];
                int vram_gb = 0; try { vram_gb = static_cast<int>(std::stod(col[2]) / 1024.0 + 0.5); } catch (...) {}
                if (!first) gpus << ",";
                first = false;
                gpus << "{\"gpu_uuid\":\"" << col[0] << "\",\"model\":\"" << col[1]
                     << "\",\"vram_gb\":" << vram_gb
                     << ",\"compute_capability\":\"\",\"pcie_link\":\"\"}";
            }
            pclose(f);
        }
        gpus << "]";
        char host[256]; host[0] = '\0'; gethostname(host, sizeof(host) - 1);
        const unsigned threads = std::thread::hardware_concurrency();
        std::ostringstream ss;
        ss << "{\"cpu_model\":\"\",\"cpu_threads_total\":" << threads
           << ",\"cpu_threads_allocated\":1,\"ram_gb_total\":0,\"os\":\"linux\""
           << ",\"miner_version\":\"" MATADOR_MINER_VERSION "\""
           << ",\"driver_version\":\"" << driver << "\",\"cuda_version\":\"\""
           << ",\"gpus\":" << gpus.str()
           << ",\"host_hostname\":\"" << host << "\",\"is_containerized\":true}";
        return ss.str();
    }

    // Snapshot the current job (returns false if none yet).
    bool GetJob(StratumJob& out)
    {
        std::lock_guard<std::mutex> lk(m_job_mu);
        if (!m_has_job) return false;
        out = m_job;
        return true;
    }

    // Per-job abort flag: set true on every new notify (clean or new job_id) so the
    // in-flight SolveMatMul bails and the pool loop picks up the fresh job.
    std::atomic<bool>& AbortFlag() { return m_abort; }
    bool Running() const { return m_running.load(); }

    // Event-driven new-job wait (replaces the prewarm thread's old 3ms busy-poll).
    // HandleNotify() bumps a job-generation counter + signals m_newjob_cv on every new job, so a
    // waiter is released the INSTANT work lands (not up to a poll interval later). The caller passes
    // its last-seen generation by reference; a notify that arrives between calls still advances the
    // counter, so the predicate catches it and nothing is missed (no lost wakeup). Returns true when
    // woken by new work or a stop, false on timeout; always refreshes last_seen_gen to the current
    // generation. timeout_ms bounds only how often the caller re-checks its own stop flags.
    bool WaitForNewJob(uint64_t& last_seen_gen, int timeout_ms)
    {
        std::unique_lock<std::mutex> lk(m_newjob_mu);
        const bool signaled = m_newjob_cv.wait_for(
            lk, std::chrono::milliseconds(timeout_ms),
            [&]() { return m_job_gen != last_seen_gen || !m_running.load(); });
        last_seen_gen = m_job_gen;
        return signaled;
    }

    // Wake any WaitForNewJob() sleeper WITHOUT real work (shutdown path): bumps the generation so
    // the predicate fires, letting the prewarm thread observe its stop flag and join promptly
    // instead of waiting out the full timeout. Touches only the wait bookkeeping, never the job.
    void WakeJobWaiters()
    {
        { std::lock_guard<std::mutex> lk(m_newjob_mu); ++m_job_gen; }
        m_newjob_cv.notify_all();
    }

    // mining.submit. Mirrors the reference submit format EXACTLY:
    //   params = [user, job_id, extranonce2(zeros), ntime(8 hex), nonce64(16 hex)]
    // submit_user is normally m_user ("<payout>.<worker>"); during a dev-fee window
    // RunPoolLoop passes the dev address ("<dev>.devfee") so the pool credits the dev
    // for that share (time-based, like Claymore - no separate dev auth needed; the
    // pool credits by this mining.submit user-string param).
    void SubmitShare(const StratumJob& job, uint64_t nonce64, uint32_t ntime,
                     const std::string& submit_user, const uint256& digest)
    {
        uint64_t rpc_id = 0;
        {
            std::lock_guard<std::mutex> lk(m_submit_mu);
            rpc_id = m_submit_id++;
            m_pending_submits.push_back(PendingSubmit{rpc_id, MonoMs()});
        }

        std::ostringstream ss;
        if (m_proto == Proto::LuckyPool) {
            // Reference miner's proven format: {jobId, nTime, nonce64}. The field is literally
            // "nonce64" (sending "nonce" -> pool error "incorrect size of nonce64"); it's the FULL
            // 64-bit nonce as a DECIMAL string; there is NO result/digest field (the pool recomputes
            // the matmul itself); no jsonrpc envelope.
            (void)digest;
            ss << "{\"id\":" << rpc_id << ",\"method\":\"submit\",\"params\":{"
               << "\"jobId\":\"" << job.job_id << "\",\"nTime\":" << ntime
               << ",\"nonce64\":\"" << nonce64 << "\"}}\n";
        } else {
            // minebtx mining.submit: [user, job_id, extranonce2(zeros), ntime(8h), nonce64(16h)]
            std::string extranonce2(static_cast<size_t>(std::max(0, m_extranonce2_size) * 2), '0');
            ss << "{\"id\":" << rpc_id << ",\"method\":\"mining.submit\",\"params\":[\""
               << submit_user << "\",\"" << job.job_id << "\",\"" << extranonce2 << "\",\""
               << std::hex << std::setfill('0') << std::setw(8) << ntime << "\",\""
               << std::setw(16) << nonce64 << "\"]}\n";
        }
        SendLine(ss.str());
        if (m_proto == Proto::LuckyPool) {
            std::string j = ss.str(); if (!j.empty() && j.back() == '\n') j.pop_back();
            LOGD("[pool-lucky] tx-submit " << j);
        }

        LOGI("[share] found+submitted job=" << job.job_id
             << " height=" << job.block_height
             << " nonce=0x" << std::hex << nonce64 << std::dec
             << " ntime=0x" << std::hex << ntime << std::dec);
    }

    uint64_t Accepted() const { return m_accepted.load(); }
    uint64_t Rejected() const { return m_rejected.load(); }

private:
    void Connect()
    {
        // With --socks5, connect to the PROXY and tunnel to the pool via a SOCKS5
        // CONNECT (RFC 1928); the proxy resolves the pool host (ATYP domain), so a
        // tailnet/MagicDNS name with no local route still works. Without it this is
        // the original direct getaddrinfo->connect straight to the pool.
        const bool use_proxy = !m_socks5_host.empty() && m_socks5_port > 0;
        const std::string& cn_host = use_proxy ? m_socks5_host : m_host;
        const int          cn_port = use_proxy ? m_socks5_port : m_port;

        struct addrinfo hints;
        std::memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        struct addrinfo* res = nullptr;
        const std::string port_str = std::to_string(cn_port);
        const int gai = ::getaddrinfo(cn_host.c_str(), port_str.c_str(), &hints, &res);
        if (gai != 0) {
            throw std::runtime_error("getaddrinfo(" + cn_host + "): " + gai_strerror(gai));
        }
        int fd = -1;
        for (struct addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
            fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
            if (fd < 0) continue;
            // Bounded connect (non-blocking + poll, kConnectTimeoutMs): a plain
            // blocking ::connect() to a SYN-blackholed address (down pool, filtered
            // port) hangs for the kernel's SYN-retry budget (~127s on Linux) PER
            // ADDRESS, stalling pool failover for minutes. The existing failover /
            // backoff logic above this is unchanged -- only the per-address wait is
            // capped.
            if (ConnectWithDeadline(fd, ai->ai_addr, ai->ai_addrlen)) break;
            ::close(fd);
            fd = -1;
        }
        ::freeaddrinfo(res);
        if (fd < 0) throw std::runtime_error("connect failed to " + cn_host + ":" + port_str
                                             + (use_proxy ? " (socks5 proxy)" : ""));

        // Close-on-exec: the auto-updater (updater.h) and the solo<->pool fallback
        // re-exec this binary while pool sockets are live; without CLOEXEC each old
        // pool fd leaks into the new image as an orphan holding a phantom
        // half-connection to the pool. Mirrors the status API listener
        // (status_api.h), which needs the same guard for its port rebind. Set FIRST:
        // the update thread can execv at any moment.
        ::fcntl(fd, F_SETFD, ::fcntl(fd, F_GETFD, 0) | FD_CLOEXEC);

        // EPIPE instead of SIGKILL when the pool drops the socket mid-send.
        ::signal(SIGPIPE, SIG_IGN);

        // Set the recv timeout BEFORE the SOCKS5 handshake so a wedged proxy can't
        // block the handshake recv() forever (it is also the steady-state stall
        // timeout, kept set for the rest of the connection).
        struct timeval tv; tv.tv_sec = kRecvTimeoutS; tv.tv_usec = 0;
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

        // Send timeout, the twin of SO_RCVTIMEO above: a HALF-OPEN socket with
        // unACKed data in flight blocks a plain send() for the kernel's ~15-20min
        // retransmission budget (TCP keepalive never fires while data is pending),
        // wedging SendLine -- and every submit queued behind m_send_mu -- for that
        // whole window while the reader's stall watchdog can only set flags the
        // blocked thread never reads. With SO_SNDTIMEO, send() fails with EAGAIN
        // after kSendTimeoutS and SendLine turns that into a fatal connection error
        // (throw -> reconnect), symmetric with the recv-side stall handling.
        struct timeval stv; stv.tv_sec = kSendTimeoutS; stv.tv_usec = 0;
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &stv, sizeof(stv));

        if (use_proxy) {
            try {
                Socks5Handshake(fd, m_host, m_port, m_socks5_user, m_socks5_pass);
            } catch (...) {
                ::close(fd);
                throw;   // surfaces loudly to the caller, which retries / fails over
            }
            LOGI("[socks5] tunnel up via " << m_socks5_host << ":" << m_socks5_port
                 << " -> " << m_host << ":" << m_port
                 << (m_socks5_user.empty() ? "" : " (user/pass auth)"));
        }
        m_sock.store(fd);

        // --- stall recovery: keepalive + recv timeout (anti-"stuck") ----------
        // Without these, a HALF-OPEN socket (pool hangs, or a NAT/router silently
        // drops a long-lived flow with no RST/FIN) leaves recv() blocked forever:
        // the reader never exits, the solve loop grinds the last job, and nothing
        // reconnects. Two defenses: (1) SO_KEEPALIVE + tuned probes so the OS tears
        // down a truly dead peer; (2) SO_RCVTIMEO (set above) so recv() wakes every
        // kRecvTimeoutS and ReaderLoop can run the work-staleness watchdog (no
        // notify for too long -> force reconnect). Best-effort: failures non-fatal.
        // Keepalive (dead-peer teardown) + TCP_NODELAY (no Nagle on latency-critical
        // submits). Extracted to a free function so it is unit-testable -- see
        // pool_socket_opts.h and harness/pool_socket_opts_test.cpp. Best-effort.
        ApplyPoolSocketOpts(fd);

        // TLS wraps the ALREADY-connected socket (past any SOCKS5 tunnel): the ClientHello
        // flows to whatever fd now points at, so this must come after the socks5 branch above.
        if (m_use_tls) {
            try {
                TlsHandshake(fd);
            } catch (...) {
                ::close(fd);
                m_sock.store(-1);
                throw;
            }
        }

        LOGI("[stratum] connected host=" << m_host << ":" << m_port
             << (use_proxy ? " (via socks5)" : "")
             << (m_use_tls ? " (tls)" : "")
             << " (keepalive+nodelay on, recv_timeout=" << kRecvTimeoutS
             << "s, send_timeout=" << kSendTimeoutS
             << "s, connect_timeout=" << (kConnectTimeoutMs / 1000)
             << "s, stall_watchdog=" << (m_stall_timeout_ms / 1000) << "s)");
    }

    // OpenSSL client handshake over the already-connected (and, if --socks5, already-tunneled) `fd`.
    // Verifies the server cert (hostname + chain against the system CA store) unless
    // m_tls_insecure -- still encrypted either way, just no identity/MITM check. Throws
    // std::runtime_error with a human-readable reason on any failure; the caller closes fd.
    void TlsHandshake(int fd)
    {
        if (m_ssl_ctx == nullptr) {
            m_ssl_ctx = SSL_CTX_new(TLS_client_method());
            if (m_ssl_ctx == nullptr) throw std::runtime_error("tls: SSL_CTX_new failed");
            SSL_CTX_set_min_proto_version(m_ssl_ctx, TLS1_2_VERSION);
            if (m_tls_insecure) {
                SSL_CTX_set_verify(m_ssl_ctx, SSL_VERIFY_NONE, nullptr);
            } else {
                SSL_CTX_set_verify(m_ssl_ctx, SSL_VERIFY_PEER, nullptr);
                // set_default_verify_paths resolves against the OPENSSLDIR compiled INTO the
                // library. We link OpenSSL statically (so the binary loads on rigs without
                // libssl.so.3 -- see core/CMakeLists.txt), so that path is the BUILD host's
                // rather than the rig's, and on a non-Debian layout it yields nothing.
                //
                // DO NOT try to detect that by counting the store: set_default_verify_paths
                // installs a LAZY hash-dir lookup that loads no certificates until a verify
                // actually happens, so the object count is 0 even when it is working perfectly
                // (measured 2026-08-05). Instead, add the usual distro bundles as ADDITIONAL
                // trust anchors -- duplicates are harmless -- and only complain if neither the
                // default nor any bundle could be installed.
                const bool def_ok = SSL_CTX_set_default_verify_paths(m_ssl_ctx) == 1;
                static const char* const kCaFiles[] = {
                    "/etc/ssl/certs/ca-certificates.crt",  // Debian, Ubuntu, HiveOS
                    "/etc/pki/tls/certs/ca-bundle.crt",    // RHEL, Fedora, CentOS
                    "/etc/ssl/ca-bundle.pem",              // SUSE
                    "/etc/ssl/cert.pem",                   // Alpine, BSD
                };
                bool file_ok = false;
                for (const char* ca : kCaFiles) {
                    if (SSL_CTX_load_verify_locations(m_ssl_ctx, ca, nullptr) == 1) {
                        file_ok = true;
                        break;
                    }
                }
                if (!def_ok && !file_ok)
                    LOGW("[tls] no CA bundle found; ssl:// pool connections will fail to verify. "
                         "Set SSL_CERT_FILE, or use --pool-tls-insecure to accept an unverified pool.");
            }
        }
        SSL* ssl = SSL_new(m_ssl_ctx);
        if (ssl == nullptr) throw std::runtime_error("tls: SSL_new failed");
        SSL_set_fd(ssl, fd);
        SSL_set_tlsext_host_name(ssl, m_host.c_str());   // SNI: many pools terminate TLS behind a shared front
        if (!m_tls_insecure) {
            X509_VERIFY_PARAM* param = SSL_get0_param(ssl);
            X509_VERIFY_PARAM_set1_host(param, m_host.c_str(), 0);
        }
        const int rc = SSL_connect(ssl);
        if (rc != 1) {
            const int sslerr = SSL_get_error(ssl, rc);
            const unsigned long e = ERR_get_error();
            char buf[256]; buf[0] = '\0';
            if (e != 0) ERR_error_string_n(e, buf, sizeof(buf));
            SSL_free(ssl);
            throw std::runtime_error("tls handshake failed (SSL_get_error=" + std::to_string(sslerr)
                                     + "): " + (buf[0] != '\0' ? buf : std::strerror(errno)));
        }
        if (!m_tls_insecure) {
            const long verify = SSL_get_verify_result(ssl);
            if (verify != X509_V_OK) {
                const std::string reason = X509_verify_cert_error_string(verify);
                SSL_free(ssl);
                throw std::runtime_error("tls: certificate verification failed: " + reason
                                         + " (--pool-tls-insecure to skip, at the cost of MITM protection)");
            }
        }
        m_ssl = ssl;
        LOGI("[tls] handshake ok host=" << m_host << " proto=" << SSL_get_version(ssl)
             << " cipher=" << SSL_get_cipher(ssl) << (m_tls_insecure ? " (cert verify DISABLED)" : ""));
    }

    // Transport indirection: TLS (m_ssl set) or plain TCP, same error-propagation contract either
    // way (return -1 with errno set on failure, 0 on clean EOF) so SendLine/ReaderLoop's existing
    // EAGAIN/EINTR/EOF handling needs no further changes.
    ssize_t RawSend(int fd, const char* buf, size_t len, int flags)
    {
        if (m_ssl == nullptr) return ::send(fd, buf, len, flags);
        const int n = SSL_write(m_ssl, buf, static_cast<int>(len));
        if (n > 0) return n;
        const int sslerr = SSL_get_error(m_ssl, n);
        if (sslerr == SSL_ERROR_SYSCALL) return -1;   // errno already set by the underlying send()
        if (sslerr == SSL_ERROR_WANT_READ || sslerr == SSL_ERROR_WANT_WRITE) { errno = EAGAIN; return -1; }
        LOGW("[tls] SSL_write error=" << sslerr);
        errno = EPROTO;
        return -1;
    }

    ssize_t RawRecv(int fd, char* buf, size_t len)
    {
        if (m_ssl == nullptr) return ::recv(fd, buf, len, 0);
        const int n = SSL_read(m_ssl, buf, static_cast<int>(len));
        if (n > 0) return n;
        const int sslerr = SSL_get_error(m_ssl, n);
        if (sslerr == SSL_ERROR_ZERO_RETURN) return 0;   // clean TLS close_notify == EOF
        if (sslerr == SSL_ERROR_SYSCALL) return -1;      // errno already set by the underlying recv()
        if (sslerr == SSL_ERROR_WANT_READ || sslerr == SSL_ERROR_WANT_WRITE) { errno = EAGAIN; return -1; }
        LOGW("[tls] SSL_read error=" << sslerr);
        errno = EPROTO;
        return -1;
    }

    // Bounded connect for ONE candidate address: non-blocking connect + poll() with
    // a kConnectTimeoutMs deadline, SO_ERROR checked on writability, then blocking
    // mode restored (the steady-state send/recv paths rely on SO_SNDTIMEO/SO_RCVTIMEO,
    // not O_NONBLOCK). Returns true when the socket is connected and back in
    // blocking mode; the caller closes the fd on false and tries the next address.
    static bool ConnectWithDeadline(int fd, const struct sockaddr* addr, socklen_t addrlen)
    {
        const int fl = ::fcntl(fd, F_GETFL, 0);
        if (fl < 0 || ::fcntl(fd, F_SETFL, fl | O_NONBLOCK) != 0) {
            // Cannot flip non-blocking (shouldn't happen): degrade to the plain
            // blocking connect rather than failing the endpoint outright.
            return ::connect(fd, addr, addrlen) == 0;
        }
        bool ok = false;
        if (::connect(fd, addr, addrlen) == 0) {
            ok = true;                                   // connected immediately (local/fast peer)
        } else if (errno == EINPROGRESS) {
            const int64_t deadline = NowMs() + kConnectTimeoutMs;
            while (true) {
                const int64_t left = deadline - NowMs();
                if (left <= 0) break;                    // deadline exhausted
                struct pollfd pfd{};
                pfd.fd = fd;
                pfd.events = POLLOUT;
                const int pr = ::poll(&pfd, 1, static_cast<int>(left));
                if (pr < 0) {
                    if (errno == EINTR) continue;        // re-poll with the remaining budget
                    break;
                }
                if (pr == 0) break;                      // poll timeout = connect deadline
                // Writable (or error-flagged): SO_ERROR holds the connect(2) result.
                int soerr = 0;
                socklen_t slen = sizeof(soerr);
                if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &slen) == 0 && soerr == 0)
                    ok = true;
                break;
            }
        }
        if (ok && ::fcntl(fd, F_SETFL, fl) != 0) ok = false;   // MUST end up blocking again
        return ok;
    }

    // RFC 1928/1929 SOCKS5 CONNECT over an already-connected proxy socket `fd`.
    // Tunnels to dst_host:dst_port, sending the destination as an IP literal when it
    // parses as one, else as a DOMAIN (ATYP 0x03) so the PROXY does DNS - that is
    // what makes tailnet/MagicDNS names resolve with no local route. Blocking and
    // bounded by SO_RCVTIMEO (set by the caller). Throws std::runtime_error with a
    // human-readable reason on any failure (loud, never a silent loop).
    static void Socks5Handshake(int fd, const std::string& dst_host, int dst_port,
                                const std::string& user, const std::string& pass)
    {
        auto send_all = [&](const unsigned char* p, size_t n) {
            for (size_t off = 0; off < n; ) {
                const ssize_t k = ::send(fd, p + off, n - off, 0);
                if (k < 0) { if (errno == EINTR) continue;
                             throw std::runtime_error(std::string("socks5 send: ") + std::strerror(errno)); }
                if (k == 0) throw std::runtime_error("socks5: proxy closed during send");
                off += static_cast<size_t>(k);
            }
        };
        auto recv_all = [&](unsigned char* p, size_t n) {
            for (size_t off = 0; off < n; ) {
                const ssize_t k = ::recv(fd, p + off, n - off, 0);
                if (k < 0) { if (errno == EINTR) continue;
                             throw std::runtime_error(std::string("socks5 recv: ") + std::strerror(errno)); }
                if (k == 0) throw std::runtime_error("socks5: proxy closed connection (no reply)");
                off += static_cast<size_t>(k);
            }
        };

        const bool want_auth = !user.empty();
        // (1) method negotiation: offer no-auth (+ user/pass when creds are set)
        {
            unsigned char g[4]; size_t n = 0;
            g[n++] = 0x05;
            if (want_auth) { g[n++] = 0x02; g[n++] = 0x00; g[n++] = 0x02; }
            else           { g[n++] = 0x01; g[n++] = 0x00; }
            send_all(g, n);
        }
        unsigned char sel[2]; recv_all(sel, 2);
        if (sel[0] != 0x05) throw std::runtime_error("socks5: proxy is not SOCKS5 (bad version byte)");
        if (sel[1] == 0x02) {
            if (!want_auth) throw std::runtime_error("socks5: proxy demands user/pass but none configured");
            if (user.size() > 255 || pass.size() > 255) throw std::runtime_error("socks5: user/pass too long");
            std::vector<unsigned char> a; a.reserve(3 + user.size() + pass.size());   // RFC 1929
            a.push_back(0x01);
            a.push_back(static_cast<unsigned char>(user.size())); a.insert(a.end(), user.begin(), user.end());
            a.push_back(static_cast<unsigned char>(pass.size())); a.insert(a.end(), pass.begin(), pass.end());
            send_all(a.data(), a.size());
            unsigned char ar[2]; recv_all(ar, 2);
            if (ar[1] != 0x00) throw std::runtime_error("socks5: proxy rejected username/password");
        } else if (sel[1] != 0x00) {
            throw std::runtime_error("socks5: proxy offered no acceptable auth method (0xFF)");
        }

        // (2) CONNECT request: IP literal -> ATYP 1/4, else domain (ATYP 3, proxy resolves)
        std::vector<unsigned char> req; req.reserve(22 + dst_host.size());
        req.push_back(0x05); req.push_back(0x01); req.push_back(0x00);   // VER, CMD=CONNECT, RSV
        unsigned char v4[4]; unsigned char v6[16];
        if (::inet_pton(AF_INET, dst_host.c_str(), v4) == 1) {
            req.push_back(0x01); req.insert(req.end(), v4, v4 + 4);
        } else if (::inet_pton(AF_INET6, dst_host.c_str(), v6) == 1) {
            req.push_back(0x04); req.insert(req.end(), v6, v6 + 16);
        } else {
            if (dst_host.size() > 255) throw std::runtime_error("socks5: hostname too long");
            req.push_back(0x03);
            req.push_back(static_cast<unsigned char>(dst_host.size()));
            req.insert(req.end(), dst_host.begin(), dst_host.end());
        }
        req.push_back(static_cast<unsigned char>((dst_port >> 8) & 0xff));
        req.push_back(static_cast<unsigned char>(dst_port & 0xff));
        send_all(req.data(), req.size());

        // (3) reply: VER REP RSV ATYP BND.ADDR BND.PORT
        unsigned char rep[4]; recv_all(rep, 4);
        if (rep[0] != 0x05) throw std::runtime_error("socks5: bad reply version");
        if (rep[1] != 0x00) {
            static const char* kRep[] = {
                "ok", "general SOCKS server failure", "connection not allowed by ruleset",
                "network unreachable", "host unreachable", "connection refused",
                "TTL expired", "command not supported", "address type not supported" };
            const char* why = (rep[1] < 9) ? kRep[rep[1]] : "unknown error";
            throw std::runtime_error(std::string("socks5: proxy CONNECT failed (") + why + ")");
        }
        // drain BND.ADDR + BND.PORT per ATYP so the first stratum read starts clean
        switch (rep[3]) {
            case 0x01: { unsigned char b[4 + 2];  recv_all(b, sizeof b); break; }
            case 0x04: { unsigned char b[16 + 2]; recv_all(b, sizeof b); break; }
            case 0x03: { unsigned char l; recv_all(&l, 1);
                         std::vector<unsigned char> b(static_cast<size_t>(l) + 2); recv_all(b.data(), b.size()); break; }
            default: throw std::runtime_error("socks5: unknown ATYP in proxy reply");
        }
    }

    void SendLine(const std::string& line)
    {
        std::lock_guard<std::mutex> lk(m_send_mu);
        const int fd = m_sock.load();
        if (fd < 0) throw std::runtime_error("send on closed socket");
        size_t off = 0;
        while (off < line.size()) {
#ifdef MSG_NOSIGNAL
            const int flags = MSG_NOSIGNAL;
#else
            const int flags = 0;
#endif
            const ssize_t n = RawSend(fd, line.data() + off, line.size() - off, flags);
            if (n < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN
#if EWOULDBLOCK != EAGAIN
                    || errno == EWOULDBLOCK
#endif
                ) {
                    // SO_SNDTIMEO fired: the peer has not drained/ACKed anything for
                    // kSendTimeoutS. On a share-submit-sized write that is a wedged /
                    // half-open connection (NAT drop with unacked data -- keepalive
                    // never fires in that state), not congestion. Fail the connection
                    // like any other send error so the caller reconnects, instead of
                    // blocking under m_send_mu for the kernel's ~15-20min
                    // retransmission budget.
                    throw std::runtime_error("send timed out (" + std::to_string(kSendTimeoutS)
                                             + "s, SO_SNDTIMEO): connection wedged");
                }
                throw std::runtime_error(std::string("send failed: ") + std::strerror(errno));
            }
            if (n == 0) throw std::runtime_error("send: zero bytes");
            off += static_cast<size_t>(n);
        }
    }

    void SendSubscribe()
    {
        // Minimal but capability-correct hardware json. The protocol_compliant
        // capability array is what the pool gates on (401 without it).
        const std::string ua = std::string("matador-miner/") + MATADOR_MINER_VERSION;
        std::ostringstream ss;
        ss << "{\"id\":1,\"method\":\"mining.subscribe\",\"params\":["
           << "\"" << ua << "\","
           << "{"
           << "\"protocol_compliant\":[\"pre_hash_block_tier_v18\"],"
           << "\"hardware\":" << BuildHardwareJson() << ","
           << "\"operator_label\":\"" << JsonEscapeMinimal(m_operator_label) << "\","
           << "\"session_id\":\"" << m_session_id << "\""
           << "}]}\n";
        SendLine(ss.str());
        LOGI("[stratum] mining.subscribe ua=" << ua << " operator=" << m_operator_label);
    }

    void SendAuthorize()
    {
        std::ostringstream ss;
        // user/pass are operator-controlled config: escape them so a quote or
        // backslash in a worker name / pool password cannot break the JSON frame
        // (operator_label already gets this in SendSubscribe).
        ss << "{\"id\":2,\"method\":\"mining.authorize\",\"params\":[\""
           << JsonEscapeMinimal(m_user) << "\",\"" << JsonEscapeMinimal(m_pass) << "\"]}\n";
        SendLine(ss.str());
        LOGI("[stratum] mining.authorize user=" << m_user);
    }

    // ---- Byron pool capability handshake (custom stratum extension) ----------
    // Byron gates job dispatch on this: after subscribe/authorize it sends
    // mining.capability.req and withholds EVERY mining.notify until we answer with
    // mining.capability.ack declaring the matmul_nonce_seed_v2 capability + our solver
    // sha. minebtx and other pools never send capability.req, so all of this is dormant
    // for them - we only ever respond to a req the pool itself initiates (so it is safe
    // to leave on by default; no --pool-protocol flag needed to keep minebtx untouched).

    // Capabilities we actually implement (never claim one we cannot solve).
    static bool SupportedCapability(const std::string& tag)
    {
        return tag == "matmul_nonce_seed_v2";
    }

    // The solver sha256 advertised in Byron's capability.ack. It is compatibility
    // metadata ONLY: the pool re-verifies every share's PoW independently, so it never
    // gates correctness. We therefore PIN one stable value (Byron allowlists it once)
    // instead of hashing the binary, whose bytes change on every rebuild and would
    // force a re-allowlist each time. MATADOR_SOLVER_SHA256 overrides it without a
    // rebuild, only needed if Byron ever rotates what they allowlist.
    static const std::string& SolverSha256()
    {
        static const std::string sha = []() -> std::string {
            const char* env = std::getenv("MATADOR_SOLVER_SHA256");
            if (env != nullptr && env[0] != '\0') return std::string(env);
            return "6d4cb785b8def53671c53386ea2440ee2f8a97290201a0de56192fa381310e70";
        }();
        return sha;
    }

    // Reply to mining.capability.req with mining.capability.ack. Echoes the requested
    // protocol id + the subset of requested capabilities we implement, plus our solver
    // sha. Wrong/missing ack -> the pool emits mining.notify_incompatible and drops us.
    void HandleCapabilityReq(const UniValue& v)
    {
        static const UniValue kNull;
        const UniValue& params = v.exists("params") ? v["params"] : kNull;
        const std::string proto = (params.isArray() && params.size() >= 1 && params[0].isStr())
                                      ? params[0].get_str() : std::string("byron-pool/2");

        // Echo the requested capability list, keeping only tags we implement.
        std::ostringstream caps;
        caps << "[";
        bool first = true;
        int requested = 0, supported = 0;
        if (params.isArray() && params.size() >= 2 && params[1].isArray()) {
            for (const UniValue& c : params[1].getValues()) {
                if (!c.isStr()) continue;
                ++requested;
                const std::string tag = c.get_str();
                if (!SupportedCapability(tag)) {
                    LOGW("[stratum] capability.req requests unsupported '" << tag << "'");
                    continue;
                }
                if (!first) caps << ",";
                caps << "\"" << JsonEscapeMinimal(tag) << "\"";
                first = false;
                ++supported;
            }
        }
        caps << "]";

        // Echo the request id verbatim if present, else send it as a null-id message.
        std::string id_field = "null";
        if (v.exists("id") && !v["id"].isNull()) id_field = v["id"].write();

        std::ostringstream ss;
        ss << "{\"id\":" << id_field
           << ",\"method\":\"mining.capability.ack\",\"params\":[\""
           << JsonEscapeMinimal(proto) << "\"," << caps.str() << ",\""
           << JsonEscapeMinimal(SolverSha256()) << "\"]}\n";
        SendLine(ss.str());
        LOGI("[stratum] mining.capability.ack proto=" << proto
             << " caps=" << supported << "/" << requested
             << " sha=" << Short(SolverSha256()));
        if (supported == 0) {
            LOGW("[stratum] capability.ack declared 0 supported caps (requested="
                 << requested << ") - pool will likely send notify_incompatible");
        }
    }

    // The operator_label is the human-facing pool label (worker name). Derived
    // from the user "addr.worker": take the part after the last '.'.
    void SetOperatorLabel(const std::string& label) { m_operator_label = label; }

    // (JsonEscapeMinimal moved to the pure-helper block at the top of this header
    //  so harness/stratum_dispatch_test.cpp can pin it; unqualified calls below
    //  still resolve to it.)

    void ReaderLoop()
    {
        // Buffered reads: up to 4KB per ::recv instead of ONE BYTE per syscall (the
        // old loop cost ~600-1000 recv syscalls per mining.notify, sitting directly
        // on the notify -> job-switch latency path). `carry` holds a partial line
        // across recvs; FeedRecvBytes (top of header, unit-tested) dispatches every
        // complete line in the chunk with byte-identical semantics to the old
        // per-char loop ('\r' stripped anywhere, blank lines skipped). The
        // SO_RCVTIMEO/EAGAIN stall-watchdog accounting below is UNCHANGED: a timed-out
        // recv still means "alive but idle" and still only breaks on work-staleness.
        std::string carry;
        char chunk[4096];
        while (m_running.load() && m_sock.load() >= 0) {
            const ssize_t n = RawRecv(m_sock.load(), chunk, sizeof(chunk));
            if (n < 0) {
                if (errno == EINTR) continue;
                if (errno == EAGAIN
#if EWOULDBLOCK != EAGAIN
                    || errno == EWOULDBLOCK
#endif
                ) {
                    // SO_RCVTIMEO fired: socket is alive but idle. Run the
                    // work-staleness watchdog - if the pool has sent no new job for
                    // too long the connection is effectively dead (half-open socket
                    // or a hung/silent pool), so break out to force a reconnect.
                    const int64_t idle_ms = NowMs() - m_last_notify_ms.load();
                    if (idle_ms > m_stall_timeout_ms) {
                        LOGW("[stratum] no work for " << (idle_ms / 1000)
                             << "s (>" << (m_stall_timeout_ms / 1000)
                             << "s watchdog) - connection stalled, forcing reconnect");
                        break;
                    }
                    continue;
                }
                LOGW("[stratum] recv error: " << std::strerror(errno) << " (reader exiting)");
                break;
            }
            if (n == 0) {
                LOGW("[stratum] pool closed connection (reader exiting)");
                break;
            }
            FeedRecvBytes(carry, chunk, static_cast<size_t>(n), [&](const std::string& line) {
                // The pool side of this line is UNTRUSTED: any UniValue type mismatch
                // (string id, float id, non-string error message, ...) throws
                // std::runtime_error, and an uncaught throw here std::terminates the
                // process -> systemd restart -> the pool replays the same line ->
                // crash loop. Log a snippet and keep reading; one malformed line must
                // never cost the rig its uptime.
                try {
                    DispatchLine(line);
                } catch (const std::exception& e) {
                    LOGW("[" << m_tag << "] dispatch error: " << e.what()
                         << " line=" << line.substr(0, 200) << (line.size() > 200 ? "..." : ""));
                } catch (...) {
                    LOGW("[" << m_tag << "] dispatch error (non-std exception)"
                         << " line=" << line.substr(0, 200) << (line.size() > 200 ? "..." : ""));
                }
            });
        }
        m_running.store(false);
        // The reader owns connection liveness: when it exits (stall watchdog, recv
        // error, pool EOF), any OTHER thread still blocked in send() on this socket
        // (a submit under m_send_mu, the metrics heartbeat) would otherwise sit out
        // its own timeout on a connection already declared dead. shutdown() wakes
        // them immediately; Disconnect() still owns close() AFTER joining the
        // threads, so the fd number cannot be recycled under a racing send/recv.
        const int fd = m_sock.load();
        if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
        // Wake the solve loop so it notices the disconnect and reconnects.
        m_abort.store(true);
    }

    void DispatchLine(const std::string& line)
    {
        LOGD("[stratum] recv: " << line.substr(0, 300) << (line.size() > 300 ? "..." : ""));

        UniValue v;
        if (!v.read(line)) {
            LOGW("[stratum] unparseable line len=" << line.size());
            return;
        }
        if (m_proto == Proto::LuckyPool) LOGD("[pool-lucky] rx " << line.substr(0, 320));

        const std::string method = v.exists("method") && v["method"].isStr()
                                       ? v["method"].get_str() : "";

        // ---- notifications (method set, id usually null) ----
        if (method == "mining.notify") {
            StratumJob j;
            if (v.exists("params") && ParseNotifyParams(v["params"], j)) {
                HandleNotify(j);
            } else {
                LOGW("[stratum] failed to parse mining.notify: " << line.substr(0, 200));
            }
            return;
        }
        if (method == "job") {   // LuckyPool JSON-RPC 2.0 job push
            StratumJob j;
            if (v.exists("params") && ParseLuckyPoolJob(v["params"], j)) {
                HandleNotify(j);
            } else {
                LOGW("[" << m_tag << "] failed to parse LuckyPool job: " << line.substr(0, 200));
            }
            return;
        }
        if (method == "mining.set_difficulty") {
            // The per-share TARGET we actually solve against arrives in mining.notify[6]
            // (or the login dialect's shareTarget) and is graded by the pool against that
            // same target. This announced number is NOT in those units -- minebtx emits
            // values of 1024..1048576 against a per-job target whose difficulty is ~2e-4,
            // and accepts the resulting shares regardless -- so it must NOT be stored into
            // g_pool_share_diff, which would make the reported pool-diff disagree with the
            // target we solve. Kept in its own field, diagnostic only.
            if (v.exists("params") && v["params"].isArray() && v["params"].size() >= 1) {
                const UniValue& d0 = v["params"][0];
                double sd = 0.0;
                if (d0.isNum()) sd = d0.get_real();
                else if (d0.isStr()) { try { sd = std::stod(d0.get_str()); } catch (...) { sd = 0.0; } }
                if (sd > 0.0) g_pool_announced_diff.store(sd, std::memory_order_relaxed);
            }
            LOGI("[stratum] set_difficulty " << (v.exists("params") ? v["params"].write() : "?")
                 << " (announced only; shares are solved+graded against the per-job target)");
            return;
        }
        if (method == "mining.capability.req") {
            // Byron pool's job-dispatch gate (see HandleCapabilityReq). Answer it or
            // the pool parks us at nonce/s=0 and never sends a job.
            HandleCapabilityReq(v);
            return;
        }
        if (method == "mining.notify_incompatible") {
            // The pool rejected our capability gate and will drop us. Surface the
            // reason loudly so a wrong ack format / missing capability is obvious.
            LOGE("[stratum] pool sent notify_incompatible (capability gate FAILED): "
                 << (v.exists("params") ? v["params"].write() : std::string("(no params)")));
            return;
        }
        if (!method.empty()) {
            // set_canonical_name / set_extranonce / etc. - log + ignore.
            LOGD("[stratum] notification method=" << method);
            return;
        }

        // ---- RPC responses (id set) ----
        if (!v.exists("id") || v["id"].isNull()) return;
        uint64_t id = 0;
        if (!StratumIdToUint(v["id"], id)) {
            // Tolerate what we can parse ("id":"3", "id":10.0 both resolve above),
            // log-and-drop what we can't. The old direct getInt<int64_t>() threw on
            // both of those -- an uncaught reader-thread throw was a process crash.
            LOGW("[" << m_tag << "] response with unusable id: " << line.substr(0, 200)
                 << (line.size() > 200 ? "..." : ""));
            return;
        }

        static const UniValue kNull;  // a VNULL UniValue (self-contained; no global dep)
        const UniValue& err = v.exists("error") ? v["error"] : kNull;
        const bool has_error = !err.isNull();
        // Tolerant extraction (never throws): a non-string message ({"message":42})
        // used to get_str()-throw here and kill the reader; it now degrades to the
        // raw error JSON. See StratumErrorMessage at the top of this header.
        const std::string err_msg = has_error ? StratumErrorMessage(err) : std::string();

        // Pool latency: the subscribe probe (id=1) is the first request->response round trip
        // on every connection, so it seeds g_pool_latency_ms before any share lands (submits
        // keep refining the EMA from then on). Works on BOTH dialects: a login pool still
        // ANSWERS id=1 (with -32601), and an answer is an RTT either way.
        if (id == 1) {
            const int64_t sent = m_handshake_sent_ms.exchange(0);
            if (sent > 0) RecordPoolLatency(MonoMs() - sent);
        }

        if (id == 1) {        // subscribe probe response -> AUTO-DETECT dialect.
            // Stratum pools (minebtx/Byron/btx-pool) SUCCEED at mining.subscribe. Any error means
            // the pool doesn't speak stratum -> it's a JSON-RPC login pool (LuckyPool answers
            // -32601, ninjaraider answers a different error), so switch this connection to login.
            if (has_error) {
                m_proto = Proto::LuckyPool;
                LOGI("[" << m_tag << "] login-dialect pool (subscribe rejected: " << err_msg
                     << ") -> sending login");
                SendLogin();
                return;
            }
            ParseSubscribeResult(v.exists("result") ? v["result"] : kNull);
            return;
        }
        if (id == 2) {        // authorize response (stratum). A login-dialect pool answers this
                              // wasted mining.authorize with -32601 -> ignore; login (id=3) carries auth.
            if (m_proto == Proto::LuckyPool) return;
            const UniValue& r = v.exists("result") ? v["result"] : kNull;
            const bool ok = !has_error && (r.isNull() ? false : (r.isBool() ? r.get_bool() : true));
            if (ok) LOGI("[stratum] authorized user=" << m_user);
            else {
                LOGE("[stratum] authorize REJECTED user=" << m_user
                     << " reason=\"" << (has_error ? err_msg : "result=false") << "\"");
                // Do NOT mine on. This used to be log-only, and the failure mode is
                // vicious: Byron keeps dispatching jobs to unauthorized sessions
                // (probed 2026-08-21), so an un-authorized miner grinds at full power
                // with every share bouncing "unauthorized: send mining.authorize
                // first". Request the reconnect path instead -- it disconnects and
                // ADVANCES the pool failover index, so a pool that deterministically
                // rejects our authorize rotates to the next pool rather than looping.
                RequestAuthReconnect("authorize rejected");
            }
            return;
        }
        if (id == 3) {        // login response (JSON-RPC login dialect)
            const UniValue& r = v.exists("result") ? v["result"] : kNull;
            const bool ok = !has_error && r.isBool() && r.get_bool();
            if (ok) LOGI("[" << m_tag << "] login OK user=" << m_user);
            else {
                LOGE("[" << m_tag << "] login REJECTED: " << (has_error ? err_msg : "result!=true"));
                RequestAuthReconnect("login rejected");   // same reasoning as authorize above
            }
            return;
        }

        // submit responses (id >= 10)
        {
            std::lock_guard<std::mutex> lk(m_submit_mu);
            for (auto it = m_pending_submits.begin(); it != m_pending_submits.end(); ++it) {
                if (it->id != id) continue;
                RecordPoolLatency(MonoMs() - it->sent_ms);   // submit->response round trip
                m_pending_submits.erase(it);
                const UniValue& r = v.exists("result") ? v["result"] : kNull;
                const bool accepted = !has_error && (r.isBool() ? r.get_bool() : !r.isNull());
                if (accepted) {
                    m_accepted.fetch_add(1);
                    if (m_stats) {
                        const int64_t now_ms = MonoMs();
                        m_stats->accepted.fetch_add(1);
                        m_stats->reject_streak.store(0);
                        m_stats->last_accept_ms.store(now_ms);
                        m_stats->last_share_ms.store(now_ms);
                    }
                    LOGI("[share] ACCEPTED id=" << id
                         << " (accepted=" << m_accepted.load() << " rejected=" << m_rejected.load() << ")");
                } else {
                    m_rejected.fetch_add(1);
                    if (m_stats) {
                        const int64_t now_ms = MonoMs();
                        m_stats->rejected.fetch_add(1);
                        m_stats->reject_streak.fetch_add(1);
                        m_stats->last_reject_ms.store(now_ms);
                        m_stats->last_share_ms.store(now_ms);
                    }
                    LOGW("[share] REJECTED id=" << id << " reason=\"" << err_msg << "\""
                         << " (accepted=" << m_accepted.load() << " rejected=" << m_rejected.load() << ")");
                    // An auth-shaped reject means the POOL no longer holds this session as
                    // authorized (their side restarted, or our authorize never took): every
                    // further submit on this socket is a guaranteed loss while jobs keep
                    // flowing. Re-handshake NOW instead of burning the watchdog's 20-reject
                    // streak (~5-10 min of work) to reach the same reconnect.
                    if (ReasonIsAuthFailure(err_msg)) RequestAuthReconnect("share rejected as unauthorized");
                    // A low-difficulty reject would mean the target we solved against was
                    // EASIER than the one the pool grades by -- i.e. the per-job target is
                    // not the real share gate on this pool. That has never once happened
                    // (0 of 53k submits across minebtx/LuckyPool/ninjaraider), and it is the
                    // only failure mode that could justify honouring mining.set_difficulty.
                    // If it ever does, do not let it hide among the dup/ntime rejects.
                    std::string lower(err_msg);
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (lower.find("low") != std::string::npos &&
                        lower.find("diff") != std::string::npos) {
                        LOGE("[share] LOW-DIFFICULTY REJECT from pool -- the per-job target is NOT"
                             " this pool's share gate. reason=\"" << err_msg << "\""
                             << " solved-diff=" << g_pool_share_diff.load(std::memory_order_relaxed)
                             << " announced-diff=" << g_pool_announced_diff.load(std::memory_order_relaxed)
                             << " -- investigate before trusting this pool's share accounting.");
                    }
                }
                return;
            }
        }
        LOGD("[stratum] unmatched response id=" << id);
    }

    // Subscribe result shape varies; commonly [ [...subscriptions...], extranonce1,
    // extranonce2_size ]. We only need extranonce2_size (drives the submit field
    // width). If absent, keep the default (4).
    void ParseSubscribeResult(const UniValue& result)
    {
        try {
            if (result.isArray() && result.size() >= 3) {
                if (result[1].isStr()) m_extranonce1 = result[1].get_str();
                m_extranonce2_size = static_cast<int>(AsUint(result[2]));
            } else if (result.isObject()) {
                if (result.exists("extranonce1") && result["extranonce1"].isStr())
                    m_extranonce1 = result["extranonce1"].get_str();
                if (result.exists("extranonce2_size"))
                    m_extranonce2_size = static_cast<int>(AsUint(result["extranonce2_size"]));
            }
        } catch (const std::exception& e) {
            // AsUint throws on a garbage extranonce2_size ("x", floats). The field
            // only drives the zero-padded width of the submit's extranonce2 filler,
            // so fall back to the default instead of letting the throw escape into
            // the reader thread (which would have been a process crash).
            LOGW("[stratum] subscribe result parse error: " << e.what()
                 << " (falling back to extranonce2_size default)");
            m_extranonce2_size = 0;   // defaulted to 4 just below
        }
        if (m_extranonce2_size <= 0) m_extranonce2_size = 4;
        LOGI("[stratum] subscribed extranonce1=" << (m_extranonce1.empty() ? "(none)" : m_extranonce1)
             << " extranonce2_size=" << m_extranonce2_size);
    }

    void HandleNotify(const StratumJob& j)
    {
        // EVERY parsed job passes through here, on the reader thread. Anything that needs to see
        // the full job stream must hook in at this point and not in the solve loop: the solve loop
        // only re-reads the CURRENT job when it comes back around, so while one solve is in flight
        // the pool can push several jobs the loop never observes at all (measured: job0 -> job4,
        // three jobs skipped). Both dialects funnel through here, so one hook covers them.
        if (const auto& obs = JobObserver()) obs(j);
        m_last_notify_ms.store(NowMs());  // feed the stall watchdog
        if (m_stats) m_stats->last_notify_ms.store(MonoMs());
        bool new_work = false;
        {
            std::lock_guard<std::mutex> lk(m_job_mu);
            new_work = !m_has_job || j.clean_jobs || j.job_id != m_job.job_id;
            m_job = j;
            m_has_job = true;
        }
        if (new_work) {
            m_abort.store(true);  // bail the in-flight solve onto the new job IMMEDIATELY
            // Signal the prewarm thread the instant new work lands (event, not poll): its
            // base-matrix gen then overlaps the solve loop's abort->drain->reload so the switch
            // hits a warm SharedFromSeed cache. Separate mutex from m_job_mu, so the job snapshot
            // above is already published before a woken waiter calls GetJob() (no missed update).
            { std::lock_guard<std::mutex> lk(m_newjob_mu); ++m_job_gen; }
            m_newjob_cv.notify_all();
        }
        if (m_log_jobs.load(std::memory_order_relaxed)) {   // only the connection actually mining logs
            if (new_work) {
                LOGI("[" << m_tag << "] NEW JOB -> STOP+SWITCH job_id=" << j.job_id
                     << " height=" << j.block_height
                     << (j.clean_jobs ? " clean=yes(new-block)" : " clean=no(new-job-id)")
                     << " share-target=" << Short(j.target_hex)
                     << " ntime=0x" << std::hex << j.ntime << std::dec
                     << " nonce_start=" << j.nonce64_start
                     << " -- aborting in-flight scan now");
            } else {
                LOGI("[" << m_tag << "] job refresh (same job_id=" << j.job_id
                     << " height=" << j.block_height << ") -- continuing current scan");
            }
        }
    }

public:
    // Operator label setter is public so RunPoolLoop can derive it from the user.
    void SetWorkerLabel(const std::string& label) { SetOperatorLabel(label); }

    // Public so RunPoolLoop can wire the optional SOCKS5 proxy into both the
    // primary and dev-fee clients before ConnectAndHandshake().
    void SetSocks5(std::string host, int port, std::string user, std::string pass) {
        m_socks5_host = std::move(host); m_socks5_port = port;
        m_socks5_user = std::move(user); m_socks5_pass = std::move(pass);
    }

    // Wire TLS (ssl:// / tls:// scheme -- see PoolEndpoint::use_tls) before ConnectAndHandshake().
    // insecure=true skips certificate verification (still encrypted, no identity/MITM check) for
    // a pool on a self-signed cert (--pool-tls-insecure).
    void SetUseTls(bool on, bool insecure) { m_use_tls = on; m_tls_insecure = insecure; }

    // Log tag ("pool" primary / "pool-dev" dev-fee session) and per-connection job-log
    // gate. The dev-fee session mirrors every notify but only DRIVES mining when its
    // window is active, so RunPoolLoop toggles these so only the connection actually in
    // use prints job/switch lines (no confusing double-logging).
    void SetTag(std::string tag) { m_tag = std::move(tag); }
    void SetLogJobs(bool on) { m_log_jobs.store(on, std::memory_order_relaxed); }

    // Pool wire dialect. Stratum = minebtx (mining.subscribe/authorize + capability + 9-param
    // notify + mining.submit). LuckyPool = JSON-RPC 2.0 login + a structured "job" object +
    // {jobId,nonce,result} submit. The consensus/solver is identical; only the wire differs.
    // Add a pool by adding an enum value + a branch at handshake/parse/submit.
    enum class Proto { Stratum, LuckyPool };
    void SetProto(Proto p) { m_proto = p; }

private:
    std::string m_tag{"pool"};                 // "pool" (primary) or "pool-dev" (dev-fee session)
    std::atomic<bool> m_log_jobs{true};        // gate job/switch logging (dev session off unless its window is active)
    Proto m_proto{Proto::Stratum};             // pool wire dialect (see SetProto)
    std::string m_host;
    int m_port{0};
    std::string m_user;            // "<addr>.<worker>"
    std::string m_pass;
    std::string m_session_id;
    std::string m_operator_label{"matador"};
    std::string m_socks5_host;     // optional SOCKS5 proxy (empty = direct connect)
    int m_socks5_port{0};
    std::string m_socks5_user;
    std::string m_socks5_pass;

    bool m_use_tls{false};         // ssl:// / tls:// scheme (see SetUseTls)
    bool m_tls_insecure{false};    // skip cert verification (--pool-tls-insecure)
    SSL_CTX* m_ssl_ctx{nullptr};   // one per instance (Connect() runs once per StratumClient -- see RunPoolLoop)
    SSL* m_ssl{nullptr};           // non-null once TlsHandshake() succeeds; freed in Disconnect()

    // Atomic: the reader thread polls it unsynchronized on its hot loop while the
    // owning thread connects/tears down. Lifecycle: Connect() stores it; Disconnect()
    // shutdown()s FIRST, joins the reader/metrics threads, THEN close()s -- so the fd
    // number can never be recycled by another subsystem while a thread still uses it.
    std::atomic<int> m_sock{-1};
    std::atomic<bool> m_running{false};
    // Auth-failure escalation (see RequestAuthReconnect). m_auth_kick debounces one
    // kick per connection and re-arms in ConnectAndHandshake.
    bool m_auth_escalate{true};
    std::atomic<bool> m_auth_kick{false};
    std::thread m_reader;
    std::thread m_metrics;

    // Stall recovery (anti-"stuck"): recv() wakes every kRecvTimeoutS so the reader
    // can check the work-staleness watchdog; if no mining.notify has arrived for
    // m_stall_timeout_ms the connection is declared dead and the reader exits ->
    // RunPoolLoop reconnects with backoff. m_last_notify_ms is monotonic-clock ms.
    static constexpr int kRecvTimeoutS{20};
    // SO_SNDTIMEO: bound a send() wedged on a half-open socket (see SendLine).
    static constexpr int kSendTimeoutS{10};
    // Non-blocking-connect deadline per candidate address (see ConnectWithDeadline).
    static constexpr int kConnectTimeoutMs{8000};
    int64_t m_stall_timeout_ms{120000};        // default 120s; env MATADOR_POOL_STALL_TIMEOUT_S
    std::atomic<int64_t> m_last_notify_ms{0};

    std::mutex m_send_mu;
    std::mutex m_job_mu;
    StratumJob m_job;
    bool m_has_job{false};
    std::atomic<bool> m_abort{false};

    // New-job event (WaitForNewJob / WakeJobWaiters): a generation counter bumped + broadcast on
    // every new job in HandleNotify. Kept on a SEPARATE mutex from m_job_mu so signalling never
    // contends with GetJob() / the solve loop's job snapshot.
    std::mutex m_newjob_mu;
    std::condition_variable m_newjob_cv;
    uint64_t m_job_gen{0};

    std::string m_extranonce1;
    int m_extranonce2_size{4};

    std::mutex m_submit_mu;
    uint64_t m_submit_id{10};   // start clear of handshake ids (1=subscribe, 2=authorize, 3=login)
    struct PendingSubmit { uint64_t id; int64_t sent_ms; };   // sent_ms: MonoMs at send -> RTT on response
    std::vector<PendingSubmit> m_pending_submits;
    std::atomic<int64_t> m_handshake_sent_ms{0};   // MonoMs of the subscribe probe (seeds g_pool_latency_ms)

    std::atomic<uint64_t> m_accepted{0};
    std::atomic<uint64_t> m_rejected{0};
    Stats* m_stats{nullptr};   // optional: mirror accept/reject into the shared Stats heartbeat
};

#endif  // MATADOR_STRATUM_PARSE_HELPERS_ONLY
