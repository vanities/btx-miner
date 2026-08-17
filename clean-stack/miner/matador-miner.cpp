// Copyright (c) 2026 The BTX developers / AM2 LLC
// Distributed under the MIT software license.
//
// matador-miner: standalone solo GBT miner for BTX.
//
// Pulls work from a local btxd over JSON-RPC (getblocktemplate), assembles the
// full CBlock (coinbase + tx list + merkle + matmul header), calls btx's OWN
// SolveMatMul() (so it inherits our overlap/occupancy patches in src/pow.cpp ->
// bitcoin_consensus), and submits via submitblock. The node never restarts on a
// miner/kernel update because the solver lives in this separate binary.
//
// CONSENSUS-CRITICAL. A wrong coinbase/merkle/header field = orphaned block.
// Verified against btxchain/btx @ 215170f2 (v0.32.11):
//   - coinbase scriptSig / nVersion / witness reserved value  -> node/miner.cpp:978-991,
//     validation.cpp:9728-9729 (GenerateCoinbaseCommitment / UpdateUncommitted)
//   - P2MR payout predicate (witness v2, 32-byte program)      -> node/miner.cpp:53-58
//   - SolveMatMul signature (ENC_RC episode solve)              -> pow.h
//   - GBT field names (curtime/bits/target/coinbasevalue/...)   -> rpc/mining.cpp:7895-8000
//   - time_policy.tip_mediantime (v3 parent-mtp seed input)     -> rpc/mining.cpp:341
//   - mainnet RPC port 19334                                    -> chainparamsbase.cpp:82

#include <arith_uint256.h>
#include <chainparams.h>
#include <common/args.h>
#include <consensus/merkle.h>       // BlockMerkleRoot
#include <core_io.h>                // DecodeHexTx, EncodeHexTx
#include <key_io.h>                 // DecodeDestination
#include <addresstype.h>           // GetScriptForDestination, CTxDestination
#include <pow.h>                    // SolveMatMul (ENC_RC episode solve), ActiveMatMulSolveVariant
#include <matmul/backend_capabilities.h> // backend::AllCapabilities / ToString / Kind /
                                         // ResolveMiningBackendFromEnvironment
#include <matmul/matmul_v4_rc.h>         // RCConsensusHeaderMatmulDim (RC-height header dim)
#include <primitives/block.h>
#include <primitives/transaction.h>
#include <script/script.h>
#include <streams.h>
#include <uint256.h>
#include <util/chaintype.h>
#include <util/strencodings.h>      // ParseHex, HexStr, EncodeBase64
#include <util/translation.h>
#include <crypto/sha256.h>          // SHA256AutoDetect (upgrade CSHA256 to SHA-NI for the CPU scanner)

#include <univalue.h>

#include <curl/curl.h>

// No <cuda.h>: the only driver-API call left (cuDriverGetVersion, for the driver-floor
// check) is reached through dlopen/dlsym, so this TU compiles without the CUDA toolkit.

#include "version_compare.h"   // VersionGreater() for the auto-update version gate (unit-tested)
#include "update_gate.h"        // DecideUpdate() for the auto-update adopt gate (unit-tested)
#include "devfee_window.h"      // InDevFeeWindow() for the time-based dev-fee gate (unit-tested)
#include "endpoint_parse.h"     // ParsePoolEndpoint()/SplitHostPort() host:port parsing (unit-tested)
#include "log_tee.h"            // optional --log-file: fd-level tee of stderr -> file (console preserved)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>                    // timegm/struct tm (auto-update release bake-time)
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cerrno>                   // errno (pool socket send/recv error paths)
#include <dlfcn.h>                  // dlopen libnvidia-ml at runtime for --clk-offset (no link-time NVML dep)
#include <cstring>                  // std::memset / std::strerror (sockaddr scratch)
#include <fstream>
#include <iomanip>                  // std::setw / std::setfill (stratum hex fields)
#include <initializer_list>
#include <iostream>
#include <memory>                   // unique_ptr (dev-fee dual stratum session)
#include <mutex>
#include <condition_variable>       // prewarm new-job event (StratumClient::WaitForNewJob)
#include <optional>
#include <limits>                   // RC activation-height sentinel (INT32_MAX = inert)
#include <random>                   // session_id entropy
#include <set>                      // submit-side dedup: (nonce,ntime) already sent this block
#include <utility>                  // std::pair (dedup key)
#include <regex>                    // mlog::Colorize -- field palette for journalctl-native color
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <deque>

#include <unistd.h>                 // isatty / STDERR_FILENO (color autodetect), close(), execv, readlink
#include <fcntl.h>                  // FD_CLOEXEC on the API socket (survives auto-update re-exec)
#include <cstdio>                   // popen/pclose/fgets/rename for GPU telemetry + auto-update
#include <sys/stat.h>               // mkdir for the nonce-cursor state dir
#include <unistd.h>                 // getuid (resolve HOME for the state dir under systemd)
#include <pwd.h>                    // getpwuid (HOME fallback when systemd leaves HOME unset)
#include <sys/stat.h>               // chmod (auto-update binary swap)
#if defined(__APPLE__)
#include <mach-o/dyld.h>            // _NSGetExecutablePath (auto-update self path on macOS)
#endif
#include <sys/socket.h>             // socket/connect/send/recv (pool mode, raw POSIX TCP)
#include <sys/wait.h>               // waitpid (basic multi-GPU process fan-out)
#include <netdb.h>                  // getaddrinfo (pool mode)
#include <netinet/in.h>             // IPPROTO_TCP (pool keepalive tuning)
#include <arpa/inet.h>              // inet_pton (socks5 IP-literal detection)
#include <netinet/tcp.h>            // TCP_KEEPIDLE/INTVL/CNT (pool keepalive tuning)
#include <sys/time.h>               // struct timeval (SO_RCVTIMEO stall watchdog)
#include <csignal>                  // SIGPIPE ignore on pool sockets; SIGTERM/SIGINT GPU-tuning revert

#ifndef MATADOR_MINER_VERSION
#define MATADOR_MINER_VERSION "v0.1.0-dev"
#endif

const TranslateFn G_TRANSLATION_FUN{nullptr};

// ---------------------------------------------------------------------------
// MODULE MAP. This miner is ONE translation unit (single .cpp, no LTO): the
// cold orchestration is split into focused headers #included at the point they
// are used, so the hot pool path keeps full cross-function inlining while the
// file stays navigable. Each header is a verbatim section of this TU (most are
// not standalone-compilable by design -- they use Config/Stats/mlog from here).
//
//   miner_log.h      structured scope-tagged logging (mlog) + color + Timer
//   solo_mining.h    SOLO mode: RPC/GBT client, coinbase+block assembly, solve+submit
//   miner_config.h   runtime Config struct (+ PoolEndpoint, dev-fee payout constant)
//   stratum_client.h POOL mode: stratum client + StratumJob + notify parsing
//   miner_format.h   pure difficulty/rate/duration formatters (unit-tested)
//   cuda_driver_floor.h  pure CUDA driver-version floor math (unit-tested)
//   gpu_telemetry.h  NVML [gpu] heartbeat telemetry (dlopen, no fork)
//   cursor_persist.h nonce-cursor persistence (resume scan past a restart)
//   config_parse.h   config-file / CLI parsing + RPC-auth resolution
//   status_api.h     read-only HTTP status API + GPU/thermal JSON + watchdog
//   updater.h        startup auto-update (GitHub release -> swap -> re-exec)
//   gpu_tuning.h     optional NVML clock/power/fan tuning + shutdown-revert
//
// Stays here (the pool-mining core): the shared Stats / gate / pool-telemetry
// state, RunPoolLoop, the idle-gate, and main().
// ---------------------------------------------------------------------------

#include "miner_log.h"

#include "solo_mining.h"

#include "miner_config.h"

#include "stratum_client.h"

#include "miner_format.h"

#include "mgpu_stats.h"   // multi-GPU [stats-all]: child snapshots + supervisor aggregation
                          // (needs FmtRate/LOGI above; included into this single TU only)

#include "gpu_telemetry.h"

// ---- RC auto-latch on the CLASSIC STRATUM path -------------------------------------------
// The poolcore path routes to the RC solver off the job's profile string ("enc-rc-v47"), so it
// never consults an activation height. Plain stratum had no equivalent signal: SolveMatMul picks
// its variant from ActiveMatMulSolveVariant() -> IsMatMulRCActive() -> RCActivationHeight(), which
// is INT32_MAX unless an operator sets it. Result: a pool could serve RC jobs and we would answer
// with base v3 digests forever. This closes that: the JOB announces activation, we latch it.
//
// Two accepted signals, strongest first:
//   1. profile string begins with "enc-rc"  -- explicit, and already the poolcore contract.
//   2. matmul_n == RCConsensusHeaderMatmulDim() (4096) -- implicit but unambiguous: consensus
//      REQUIRES that dim at RC heights and rejects it below them, and live v3 jobs carry 512.
//      Costs pools nothing, since they must already send the RC dim to get a valid header.
// A height is NEVER compiled in (upstream has moved it repeatedly) and an operator pin always
// wins. Opt out with BTX_RC_STRATUM_AUTOLATCH=0.
static bool RCStratumAutoLatchEnabled()
{
    const char* v = std::getenv("BTX_RC_STRATUM_AUTOLATCH");
    return v == nullptr || !(v[0] == '0' && v[1] == '\0');
}

// The signal is pool-controlled data driving a process-wide, otherwise-irreversible switch, so it
// is DEBOUNCED both ways rather than trusted on sight:
//   - latch only after the signal holds across kRCLatchJobs distinct jobs. One malformed job can
//     no longer strand a rig on a path whose every share would be rejected.
//   - un-latch after kRCUnlatchJobs distinct pre-RC jobs at or above the latched height, which is
//     the recovery path if we latched on a false positive. Chain activation is genuinely one-way,
//     so this only ever undoes OUR mistake, never a real activation (a real one keeps announcing).
// Two jobs cost a few seconds at real activation and buy immunity to a single bad job.
// LATCH ON THE FIRST SIGNAL. The original 2-job debounce had the risk backwards, and testing the
// published v0.9.8 asset against a single-job pool proved it: with one RC job in flight we did not
// latch, fell through to the BASE v3 solver at matmul_dim 4096, and submitted 1058 shares that
// consensus can never accept at an RC height. Failing to latch is the EXPENSIVE direction; every
// share produced in that window is garbage aimed at a pool.
//
// Latching eagerly costs at most a few wasted episodes if a job is malformed, and the de-latch
// below already recovers from exactly that. So: one signal latches, three clean pre-RC jobs back
// it out. A real activation keeps announcing itself and never de-latches; our own false positive
// does.
static constexpr int kRCLatchJobs   = 1;
static constexpr int kRCUnlatchJobs = 3;

static void MaybeLatchRCFromJob(const StratumJob& job)
{
    if (job.block_height <= 0) return;
    if (!RCStratumAutoLatchEnabled()) return;

    const std::string prof = ToLowerCopy(job.profile);
    const bool by_profile = prof.rfind("enc-rc", 0) == 0;
    const bool by_dim     = job.matmul_n != 0 &&
                            job.matmul_n == matmul::v4::rc::RCConsensusHeaderMatmulDim();
    const bool announces_rc = by_profile || by_dim;

    // Debounce over DISTINCT jobs. Called from the stratum reader thread via JobObserver, so it
    // sees every job the pool sent. Dedup keeps a short ring rather than just the previous id:
    // the miner runs TWO sessions (main + dev-fee) that each receive the same job stream, and
    // they interleave, so "same as last" would double-count whenever the order is main(j0),
    // main(j1), dev(j0).
    static std::mutex mu;
    static std::deque<std::string> recent_ids;
    static int rc_streak = 0;
    static int plain_streak = 0;
    std::lock_guard<std::mutex> lk(mu);
    if (std::find(recent_ids.begin(), recent_ids.end(), job.job_id) != recent_ids.end()) return;
    recent_ids.push_back(job.job_id);
    if (recent_ids.size() > 8) recent_ids.pop_front();

    if (announces_rc) { ++rc_streak; plain_streak = 0; } else { ++plain_streak; rc_streak = 0; }

    const bool latched = Consensus::Params::RCActivationHeight() != std::numeric_limits<int32_t>::max();
    if (!latched) {
        if (rc_streak < kRCLatchJobs) return;
        if (Consensus::Params::LatchRCActivationHeight(job.block_height)) {
            LOGW("[pool] ENC_RC ACTIVATION detected from pool job " << job.job_id
                 << " at height " << job.block_height
                 << " (signal=" << (by_profile ? "profile" : "matmul_dim")
                 << " profile=" << (job.profile.empty() ? "<none>" : job.profile)
                 << " matmul_n=" << job.matmul_n
                 << ", confirmed over " << rc_streak << " jobs)"
                 << " -- switching this pool session to the RC episode solver from this job on."
                 << " Pin BTX_MATMUL_RC_HEIGHT (or rc_height in config) to override;"
                 << " BTX_RC_STRATUM_AUTOLATCH=0 disables this detection.");
        }
        return;
    }

    // Latched. A pool that keeps serving pre-RC jobs at or above the activation height it just
    // announced is telling us we got it wrong; back out rather than mine work nobody accepts.
    if (plain_streak >= kRCUnlatchJobs &&
        job.block_height >= Consensus::Params::RCActivationHeight()) {
        if (Consensus::Params::UnlatchRCActivationHeight()) {
            LOGW("[pool] ENC_RC de-latched: " << plain_streak << " consecutive jobs at height "
                 << job.block_height << " carry no RC signal (matmul_n=" << job.matmul_n
                 << " profile=" << (job.profile.empty() ? "<none>" : job.profile)
                 << "). Treating the earlier detection as a false positive and returning to the"
                 << " base v3 path. Pin rc_height if this pool's signalling cannot be trusted.");
        }
        plain_streak = 0;
    }
}

// Build a CBlockHeader-equivalent CBlock from a stratum job. Endianness matches
// the solo AssembleBlock: prevhash/merkleroot via uint256::FromHex (RPC/display
// order), nbits already parsed compact in ParseNotifyParams.
static bool BuildHeaderFromJob(const StratumJob& job,
                               const Consensus::Params& consensus,
                               CBlock& block /*out*/,
                               uint256& share_target /*out*/)
{
    // NOTE: RC activation accounting is NOT done here. It runs on the reader thread via
    // JobObserver so it sees every job; doing it here would only see the jobs this loop
    // reaches, which under a long solve is a subset. By the time we build a header the
    // latch state is already settled for this job.
    auto prev = uint256::FromHex(job.prevhash_hex);
    auto merk = uint256::FromHex(job.merkleroot_hex);
    auto targ = uint256::FromHex(job.target_hex);
    if (!prev || !merk || !targ) {
        LOGW("[pool] job " << job.job_id << " has malformed 32-byte field(s)"
             << " prev_ok=" << (prev ? 1 : 0)
             << " merkle_ok=" << (merk ? 1 : 0)
             << " target_ok=" << (targ ? 1 : 0));
        return false;
    }

    block.SetNull();
    block.nVersion      = static_cast<int32_t>(job.version);
    block.hashPrevBlock = prev.value();
    block.hashMerkleRoot= merk.value();
    block.nTime         = job.ntime;
    block.nBits         = job.nbits;
    block.nNonce64      = job.nonce64_start;
    block.nNonce        = static_cast<uint32_t>(job.nonce64_start);
    block.mix_hash.SetNull();
    block.matmul_digest.SetNull();
    // Pool-sent matmul_n always wins. The fallback is RC-aware: at RC heights consensus
    // requires matmul_dim == nMatMulV4Dimension (4096 mainnet) and the field is in the
    // seed/sigma preimages, so the v3 nMatMulDimension fallback would fork every share.
    // ...with ONE exception: at RC heights consensus does not accept a choice. It rejects any
    // header whose matmul_dim != the RC dimension, and the field feeds both the seed and sigma
    // preimages, so honouring a pool's disagreeing value would grind a shape that can never
    // validate -- every share rejected, with nothing in the log to say why. Consensus wins, loudly.
    const bool rc_active = consensus.IsMatMulRCActive(job.block_height);
    const uint16_t rc_dim = matmul::v4::rc::RCConsensusHeaderMatmulDim();
    if (rc_active && job.matmul_n != 0 && job.matmul_n != rc_dim) {
        static std::atomic<int32_t> s_warned_dim{0};
        const int32_t seen = static_cast<int32_t>(job.matmul_n);
        int32_t prev = s_warned_dim.load(std::memory_order_relaxed);
        if (prev != seen && s_warned_dim.compare_exchange_strong(prev, seen, std::memory_order_relaxed)) {
            LOGW("[pool] job " << job.job_id << " at RC height " << job.block_height
                 << " carries matmul_n=" << job.matmul_n << " but ENC_RC consensus requires "
                 << rc_dim << " -- using " << rc_dim << ". Shares would be rejected otherwise;"
                 << " the pool is sending a pre-RC dimension for an RC job.");
        }
    }
    block.matmul_dim    = rc_active ? rc_dim
                        : (job.matmul_n != 0 ? job.matmul_n
                                             : static_cast<uint16_t>(consensus.nMatMulDimension));

    share_target = targ.value();

    // Publish live telemetry for the [stats] heartbeat (never read back into the
    // solve -> the digest stays byte-exact).
    g_pool_net_diff.store(DifficultyFromCompactBits(job.nbits), std::memory_order_relaxed);
    g_pool_attempts_per_share.store(AttemptsPerShare(UintToArith256(share_target)),
                                    std::memory_order_relaxed);
    // pool-diff from the real share target -- the one the pool grades our submits by.
    // A pool's mining.set_difficulty (if it sends one at all) is NOT this and never
    // overwrites it; it lands in g_pool_announced_diff instead.
    g_pool_share_diff.store(DifficultyFromTarget(UintToArith256(share_target)),
                            std::memory_order_relaxed);
    g_pool_block_height.store(job.block_height, std::memory_order_relaxed);
    return true;
}

#include "cursor_persist.h"

// Pool main loop: connect (with backoff), then repeatedly take the current job,
// solve SHARES with SolveMatMul (same solver/overlap as solo), and submit each
// one, re-searching the SAME job until a new notify flips the abort flag.
static void RunPoolLoop(const Config& cfg,
                        const Consensus::Params& consensus,
                        Stats& stats,
                        std::atomic<bool>& stop_all)
{
    const std::string user = cfg.payoutaddress + "." + cfg.worker;
    std::vector<PoolEndpoint> pools = cfg.pools;
    if (pools.empty() && !cfg.pool_host.empty() && cfg.pool_port > 0) {
        pools.push_back(PoolEndpoint{cfg.pool_host, cfg.pool_port, "primary"});
    }
    if (pools.empty()) {
        LOGE("[pool] no pool endpoints configured");
        return;
    }
    LOGI("[pool] failover list endpoints=" << pools.size()
         << " primary=" << pools.front().host << ":" << pools.front().port);

    // ---- pool dev-fee (time-based, like solo + Claymore/ethminer) -------------
    // The pool builds the coinbase, so we can't swap the payout address. And the pool
    // credits the AUTHORIZED session, IGNORING mining.submit's user field - so crediting
    // the dev requires a SECOND connection authorized as the dev address (see the dual
    // session set up after connect below, and private/matador-miner/pool-dev-fee.md).
    // For ~devfee% of wall-clock time we submit found shares on that dev session's socket.
    // The fee is mandatory (>=1%, floored in main). A downloader's payout is THEIR address;
    // the dev address defaults to the baked-in kDevAddress.
    static const double kPoolDevPeriodSec = 3600.0;   // 1h rotation
    const int devfee = cfg.devfee;   // already floored to [1,100] in main
    const std::string dev_user =
        (cfg.devaddress.empty() ? std::string(kDevAddress) : cfg.devaddress) + ".devfee";
    const double dev_window_sec = kPoolDevPeriodSec * (static_cast<double>(devfee) / 100.0);
    bool prev_dev_window = false;
    LOGI("[devfee] pool dev-fee " << devfee << "% (time-based, mandatory): ~"
         << static_cast<uint64_t>(dev_window_sec) << "s of every "
         << static_cast<uint64_t>(kPoolDevPeriodSec) << "s -> shares submitted under "
         << dev_user << ".");

    int backoff_s = 1;
    size_t pool_index = 0;
    // Submit-side dedup, PERSISTED ACROSS RECONNECTS (declared outside the connect loop on
    // purpose). A staleness-watchdog reconnect re-subscribes and resets nonce_cursor to the
    // range start, re-scanning ground already covered - which re-finds and re-submits shares
    // we already sent, drawing a cosmetic "duplicate share" reject. Remember (nonce64, ntime)
    // submitted for the CURRENT block and skip re-sends so the pool reject counter stays
    // clean. Cleared on each new block (when old keys stop mattering). Solution/PoW untouched.
    std::set<std::pair<uint64_t, uint32_t>> submitted_keys;
    int64_t submitted_block = -1;
    // Nonce-cursor persistence (effective-hashrate): load saved per-range cursors so a restart
    // resumes PAST where we scanned instead of re-scanning + re-submitting (duplicate rejects).
    // Updated throttled in the solve loop + flushed on graceful stop.
    CursorRanges cursor_ranges = LoadCursorRanges();
    int64_t last_cursor_save_ms = 0;
    if (!cursor_ranges.empty())
        LOGI("[cursor] loaded " << cursor_ranges.size() << " saved range-cursor(s) from " << CursorStateDir());

    // [downtime] tracker: mining is "down" while we wait for work, the idle-gate pauses us, or we
    // are between pool connections. mark_down() records when a gap opens (once, keeping the first
    // cause); mark_up() emits ONE greppable INFO line with the measured duration when mining
    // resumes -- only for gaps over the threshold, so normal sub-200ms hiccups stay quiet. Lives
    // OUTSIDE the connect loop so a disconnect -> reconnect -> first-job gap is one measured span.
    // Touched only from this (the solve) thread, so it needs no synchronization.
    constexpr int64_t kDowntimeLogThreshMs = 200;
    int64_t down_since_ms = 0;                 // 0 = mining (no open gap)
    std::string down_reason;
    auto mark_down = [&](const char* why) {
        if (down_since_ms == 0) { down_since_ms = MonoMs(); down_reason = why; }
    };
    auto mark_up = [&]() {
        if (down_since_ms != 0) {
            const int64_t gap = MonoMs() - down_since_ms;
            if (gap > kDowntimeLogThreshMs)
                LOGI("[downtime] mining resumed after " << gap << "ms idle (" << down_reason << ")");
            down_since_ms = 0;
        }
    };
    while (!stop_all.load()) {
        const PoolEndpoint& endpoint = pools[pool_index];
        StratumClient client(endpoint.host, endpoint.port, user, cfg.pool_pass, &stats);
        client.SetWorkerLabel(cfg.operator_label.empty() ? cfg.worker : cfg.operator_label);
        client.SetSocks5(cfg.socks5_host, cfg.socks5_port, cfg.socks5_user, cfg.socks5_pass);
        client.SetUseTls(endpoint.use_tls, cfg.pool_tls_insecure);
        // The pool dialect (stratum vs JSON-RPC login) is AUTO-DETECTED at handshake, so no
        // per-pool config is needed. This hostname heuristic ONLY tags the log ("pool-lucky")
        // so the dialect is greppable pre-handshake; it does not gate any behavior (the dev-fee
        // dual session runs on its own connection/lane either way). A wrong guess mislabels the
        // log line and nothing else.
        const bool is_luckypool = endpoint.host.find("lproute") != std::string::npos
                               || endpoint.host.find("luckypool") != std::string::npos
                               || endpoint.host.find("ninjaraider") != std::string::npos
                               || endpoint.label.find("lucky") != std::string::npos;
        if (is_luckypool) client.SetTag("pool-lucky");
        try {
            LOGI("[pool] connecting index=" << pool_index << "/" << pools.size()
                 << " endpoint=" << endpoint.host << ":" << endpoint.port
                 << (endpoint.label.empty() ? "" : (" label=" + endpoint.label)));
            client.ConnectAndHandshake();
        } catch (const std::exception& e) {
            mark_down("pool connect failed");
            const size_t next_index = (pool_index + 1) % pools.size();
            LOGW("[stratum] connect failed pool=" << endpoint.host << ":" << endpoint.port
                 << " error=\"" << e.what() << "\" next_index=" << next_index
                 << " retry in " << backoff_s << "s");
            std::this_thread::sleep_for(std::chrono::seconds(backoff_s));
            pool_index = next_index;
            backoff_s = std::min(backoff_s * 2, 30);
            continue;
        }
        backoff_s = 1;  // reset on a successful connect

        // Nonce cursor PERSISTED across jobs (per assigned range) so same-block jobs
        // (clean=no, same nonce64_start, new ntime every ~5s) keep advancing instead
        // of restarting at the bottom of the range. Reset only when the pool assigns a
        // new range (new block). Avoids redundant re-scanning + stale shares.
        uint64_t nonce_cursor = 0;
        uint64_t cur_range = ~0ULL;
        // [switch] per-job accounting (persists across the job loop): how often we abort onto a
        // new job and how much we scan per job -> the switch-tax (frequent short jobs = regen churn).
        uint64_t sw_jobs = 0, sw_nonces = 0; int64_t sw_ms = 0, sw_last_log_ms = MonoMs();

        // ---- dev-fee DUAL SESSION ------------------------------------------------
        // The pool credits the AUTHORIZED address, so a dev-fee share must be found + submitted
        // on a connection authorized AS the dev address. We open a second warm session as
        // dev_user and, during the fee window, MINE AND SUBMIT ON IT (its own job/nonce lane).
        // This works for both dialects: stratum job_ids are global so mining the dev session's
        // job is the same work; login pools assign each connection its own noncePrefix, so mining
        // the dev session's lane is the only way its shares validate (a primary-lane share would
        // be rejected there). If the dev session is not up, dev-window shares fall back to mining
        // the primary (credit you) - we never drop a share to chase the fee.
        std::unique_ptr<StratumClient> dev_client;
        bool dev_down_warned = false;
        if (devfee > 0) {
            dev_client = std::make_unique<StratumClient>(endpoint.host, endpoint.port,
                                                         dev_user, cfg.pool_pass, &stats);
            dev_client->SetWorkerLabel(cfg.operator_label.empty() ? cfg.worker : cfg.operator_label);
            dev_client->SetTag("pool-dev");
            dev_client->SetLogJobs(false);  // dev session prints job/switch lines only while its fee-window is active
            dev_client->SetSocks5(cfg.socks5_host, cfg.socks5_port, cfg.socks5_user, cfg.socks5_pass);
            dev_client->SetUseTls(endpoint.use_tls, cfg.pool_tls_insecure);
            try {
                dev_client->ConnectAndHandshake();
                LOGI("[devfee] dev session connected (" << dev_user
                     << "); dev-window shares are mined + submitted on it");
            } catch (const std::exception& e) {
                LOGW("[devfee] dev session connect failed (" << e.what()
                     << "); dev-window shares will credit YOU until it reconnects");
                dev_client.reset();
            }
        }

        // (The old per-job "prewarm" thread is GONE: seeds are per-NONCE on the live v2/v3
        // path -- nNonce64 is hashed into DeterministicMatMulSeedV2/V3 -- so the base
        // matrices it warmed for the job-start nonce were never read by the GPU solve
        // path; the only LRU consumer is the per-share CPU confirm at an arbitrary nonce.
        // It cost ~2 base-matrix fills of CPU per job switch and evicted useful LRU
        // entries for nothing. The [switch] setup= stat measured the same dead cache.)

        // New-job generation cursors (see WaitForNewJob): per connection, used for the
        // event-driven first-job wait and the notify-vs-arm race guard below.
        uint64_t job_gen_primary = 0, job_gen_dev = 0;

        // ---- solve loop for this connection ----
        while (client.Running() && !stop_all.load() && !stats.watchdog_reconnect_requested.load()) {
            // idle-gate: if the box is busy, pause here (stay subscribed, don't solve).
            if (!GateAllowsMining()) {
                mark_down("idle-gate paused");
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            // dev-fee: during the fee window (dev session up), MINE + SUBMIT on the dev session's
            // OWN job/lane -- the pool credits the connection's authorized address. Works for
            // stratum (global jobs) and login pools (each connection gets its own noncePrefix, so
            // its shares only validate when mined in its own lane). Outside the window / dev down ->
            // mine the primary (credit you). We never drop a share to chase the fee.
            bool in_dev = false;
            if (devfee > 0) {
                // WALL-CLOCK phase, never process uptime -- see the solo path for the full
                // account. Uptime restarts at 0, so every (re)start lands in the leading dev
                // slice; worse, the session is chosen when a solve BEGINS, so a solve started
                // there credits dev for its whole run. Caught live 2026-08-12 minutes after
                // switching to pool: the miner restarted and the very first share (a 140 s
                // solve begun 7 s into the new process) was submitted on the dev session.
                // v0.9.23 fixed the solo loop and missed this one.
                const double phase_sec = static_cast<double>(std::time(nullptr));
                in_dev = InDevFeeWindow(phase_sec, kPoolDevPeriodSec, devfee);
                if (in_dev != prev_dev_window) {
                    LOGI((in_dev ? "[devfee] >>> pool dev-fee window: mining + crediting the dev"
                                 : "[devfee] <<< pool dev-fee window ended: mining + crediting you"));
                    client.SetLogJobs(!in_dev);
                    if (dev_client) dev_client->SetLogJobs(in_dev);
                    prev_dev_window = in_dev;
                }
            }
            const bool dev_up = dev_client && dev_client->Running();
            if (in_dev && !dev_up && !dev_down_warned) {
                LOGW("[devfee] dev session down during dev window; crediting YOU until it reconnects");
                dev_down_warned = true;
            }
            if (dev_up) dev_down_warned = false;
            const bool use_dev = in_dev && dev_up;
            StratumClient& mine = use_dev ? *dev_client : client;
            const std::string& mine_user = use_dev ? dev_user : user;

            // Snapshot this connection's job generation BEFORE reading the job:
            // HandleNotify bumps it on every accepted job, so a bump observed after
            // arming means a notify raced us -- even if the pool re-minted the same
            // job_id (the old job_id compare missed that case). Timeout 0 = probe.
            uint64_t& mine_job_gen = use_dev ? job_gen_dev : job_gen_primary;
            (void)mine.WaitForNewJob(mine_job_gen, 0);

            StratumJob job;
            if (!mine.GetJob(job) || !job.valid) {
                mark_down("waiting for pool job");
                // Event-driven first-job wait (HandleNotify bumps the generation and
                // wakes us with ~zero latency); the 200ms timeout only bounds the
                // stop-flag re-check, same ceiling as the old fixed poll.
                mine.WaitForNewJob(mine_job_gen, 200);
                continue;
            }

            std::atomic<bool>& abort_flag = mine.AbortFlag();
            abort_flag.store(false);   // arm for this job; reader sets it on new work

            // Re-arm race guard: if the generation moved between the snapshot and the
            // arm, a newer job landed (abort may have been set-then-cleared under us);
            // loop again so we pick up and arm against the NEWEST job.
            if (mine.WaitForNewJob(mine_job_gen, 0)) {
                continue;
            }

            CBlock header;
            uint256 share_target;
            if (!BuildHeaderFromJob(job, consensus, header, share_target)) {
                mark_down("malformed job (header build failed)");
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }

            // Reset the cursor only on a NEW assigned range (new block); on same-range
            // jobs (clean=no) keep advancing it -> no re-scan, fresh ntime avoids stale.
            if (job.nonce_bits > 0 && job.nonce_bits < 64) {
                // Login pools (LuckyPool/ninjaraider) assign a FIXED per-connection nonce lane
                // [start, start + 2^nonce_bits) and REJECT any nonce64 outside it ("nonce64 outside
                // assigned nonce prefix"). The lane prefix is the SAME across jobs, so an
                // accumulating cursor walks out of the 2^nonce_bits window after a few minutes. Each
                // job carries a fresh ntime (fresh digests), so reset to the lane start every job ->
                // fresh work that always stays inside the assigned prefix.
                cur_range    = job.nonce64_start;
                nonce_cursor = job.nonce64_start;
                LOGD("[pool] login-pool lane reset to start=" << nonce_cursor
                     << " (bits=" << job.nonce_bits << ", fresh ntime)");
            } else if (job.nonce64_start != cur_range) {
                cur_range    = job.nonce64_start;
                nonce_cursor = ResumeCursor(cursor_ranges, job.nonce64_start);
                const bool resumed = nonce_cursor != job.nonce64_start;
                LOGI("[pool] nonce range start=" << nonce_cursor
                     << " (new range, clean=" << (job.clean_jobs ? "yes" : "no")
                     << (resumed ? ", RESUMED past restart" : "") << ")");
            } else {
                LOGD("[pool] same range; continuing cursor=" << nonce_cursor << " (no re-scan)");
            }

            LOGI("[solve] pool job=" << job.job_id
                 << " height=" << job.block_height
                 << " matmul_dim=" << header.matmul_dim
                 << " parent_mtp=" << job.parent_mtp
                 << " cursor=" << nonce_cursor
                 << " share-target=" << Short(share_target));

            // Keep searching the SAME job until a new notify aborts us (or the
            // connection drops). On each found share, submit and advance the
            // nonce cursor past the winner, then continue from there.
            // (nonce_cursor persists across jobs - set above, NOT reset to nonce_start here.)
            //
            // NOTE (2026-06-15): a solver-threads fan-out (N disjoint nonce lanes) was
            // tried here to chase the bench's "~6x Metal concurrency" win. It did NOT
            // help live pool mining (flat ~1180 nonce/s at 1 vs 24 lanes; backend
            // pool-slots=16 also flat; forcing batch=1 to enable concurrency was WORSE,
            // 715/s). The live path is the batch=192 nonce-prefilter path, already
            // GPU-throughput-bound and concurrency-immune - the bench's 6x was the
            // batch=1 matmul-only strawman. Reverted; pool mining stays single-stream.

            // We are committed to mining this job now: close any open [downtime] gap (waiting for
            // work / gate pause / reconnect) with one greppable resume line.
            mark_up();
            const int64_t job_t0 = MonoMs();      // [switch] this job's mining start
            const uint64_t job_c0 = nonce_cursor; // [switch] this job's start cursor
            // (No per-job base-matrix "regen"/prewarm here anymore: seeds are per-NONCE,
            // so a job-start SharedFromSeed warmed a cache entry the GPU solve path never
            // reads -- the [switch] setup= stat was measuring its own prewarm. Removed.)

            // [share-sink] Shares are submitted from INSIDE SolveMatMul: the solver hands
            // each share-target hit to this callback and keeps mining on the SAME hot
            // pipeline -- no per-share solver teardown, no discarded in-flight digest
            // window, no cold pipeline rebuild or nonce re-scan, and the submit no longer
            // sits behind the next window's digest drain. Return true = consumed, keep
            // mining; false = stop the solve (submit socket died; the outer loop
            // re-selects the connection). BLOCK-target solves still RETURN from the
            // solver without this callback and are submitted by the post-solve path
            // below, exactly as before.
            bool sink_stop = false;   // sink returned false (socket death), not a block solve
            const std::function<bool(const CBlockHeader&)> share_sink =
                [&](const CBlockHeader& share) -> bool {
                // DROP if a new BLOCK superseded this job: a share for a stale block is
                // doomed (the pool rejects it) - skip the round-trip. A share for the
                // SAME block with a newer ntime/job (clean=no) is still valid.
                {
                    StratumJob latest;
                    if (mine.GetJob(latest) && latest.valid &&
                        (latest.block_height != job.block_height ||
                         latest.prevhash_hex != job.prevhash_hex)) {
                        stats.stale.fetch_add(1);
                        LOGI("[share] DROP stale (block " << job.block_height << "->"
                             << latest.block_height << ") job=" << job.job_id
                             << " nonce64=" << share.nNonce64);
                        nonce_cursor = std::max<uint64_t>(nonce_cursor, share.nNonce64 + 1);
                        return true;   // keep mining; the pending notify aborts the solve anyway
                    }
                }
                // Submit-side dedup (see submitted_keys above): on a new block reset the
                // seen-set, then skip any (nonce, ntime) already submitted this block.
                if (job.block_height != submitted_block) {
                    submitted_keys.clear();
                    submitted_block = job.block_height;
                }
                if (!submitted_keys.insert({share.nNonce64, share.nTime}).second) {
                    LOGI("[share] DROP duplicate (already sent this block) job=" << job.job_id
                         << " nonce64=" << share.nNonce64 << " ntime=" << share.nTime);
                    nonce_cursor = std::max<uint64_t>(nonce_cursor, share.nNonce64 + 1);
                    return true;
                }
                if (use_dev) stats.dev_shares.fetch_add(1);
                try {
                    mine.SubmitShare(job, share.nNonce64, share.nTime, mine_user, share.matmul_digest);
                } catch (const std::exception& e) {
                    LOGW("[share] submit send failed: " << e.what());
                    if (use_dev) dev_client.reset();  // dev socket dead -> fall back to primary
                    sink_stop = true;
                    return false;   // stop the solve; outer loop re-selects the connection
                }
                LOGI("[share] " << (use_dev ? "DEV-fee " : "") << "submit job=" << job.job_id
                     << " h=" << job.block_height
                     << " nonce64=" << share.nNonce64
                     << " ntime=" << share.nTime
                     << " digest=" << Short(share.matmul_digest)
                     << " (pipeline hot)");
                // Advance past the found nonce and persist IMMEDIATELY: a hard kill can
                // then never resume BEFORE an already-sent nonce -> post-restart re-scans
                // draw zero duplicate-share rejects. Shares are rare, the write is cheap.
                nonce_cursor = std::max<uint64_t>(nonce_cursor, share.nNonce64 + 1);
                RecordCursor(cursor_ranges, cur_range, nonce_cursor);
                SaveCursorRanges(cursor_ranges);
                last_cursor_save_ms = MonoMs();
                return true;
            };
            while (mine.Running() && !stop_all.load() && !abort_flag.load() && !stats.watchdog_reconnect_requested.load() && GateAllowsMining()) {
                // Persist scan progress (throttled) so a restart resumes here, not at the range
                // base -> no re-scan + re-submit -> no duplicate-share rejects.
                RecordCursor(cursor_ranges, cur_range, nonce_cursor);
                if (MonoMs() - last_cursor_save_ms > 3000) {
                    SaveCursorRanges(cursor_ranges);
                    last_cursor_save_ms = MonoMs();
                }
                CBlock b = header;
                b.nNonce64 = nonce_cursor;
                b.nNonce   = static_cast<uint32_t>(nonce_cursor);
                const uint64_t call_base = nonce_cursor; // cursor at solve entry (sink moves nonce_cursor)

                uint64_t tries = cfg.maxtries;  // [in/out]
                Timer sp;
                bool solved = false;
                uint64_t attempted = 0;
                sink_stop = false;
                solved = SolveMatMul(
                    b, consensus, tries, job.block_height, &abort_flag,
                    /*share_target_override=*/&share_target,
                    /*parent_median_time_past=*/std::optional<int64_t>(job.parent_mtp),
                    &share_sink);
                attempted = (cfg.maxtries >= tries) ? (cfg.maxtries - tries) : 0;
                stats.total_nonces.fetch_add(attempted);
                if (attempted > 0) stats.last_nonce_ms.store(MonoMs());

                if (solved && sink_stop) {
                    // The sink stopped the solve on a dead submit socket: the returned
                    // share was NOT sent (its submit threw). Leave the inner loop; the
                    // outer loop re-selects the connection / reconnects. The cursor
                    // already sits past every share that WAS submitted.
                    break;
                }

                if (!solved) {
                    // Budget exhausted, or aborted by a new job / disconnect. With the
                    // sink active any shares this pass were already submitted inline.
                    LOGD("[solve] pass done job=" << job.job_id
                         << " attempted=" << attempted
                         << " aborted=" << (abort_flag.load() ? "true" : "false")
                         << " in " << sp.ms() << "ms");
                    // The sink may have advanced nonce_cursor past submitted shares; take
                    // the max with the scanned range so neither advance is lost. (Also
                    // banked on abort now: those nonces WERE scanned; resuming past them
                    // scans fresh ones instead of re-lottery-ing the same range, and
                    // removes a duplicate-share suspect on clean=no same-range refreshes.)
                    nonce_cursor = std::max<uint64_t>(nonce_cursor, call_base + attempted);
                    if (abort_flag.load()) break;          // new job -> reload
                    if (attempted == 0) {
                        // Nothing tried (wedged backend / empty pass): breathe so a dead
                        // GPU cannot hot-spin this loop at zero hashrate and flood the
                        // journal; the zero-nonce watchdog handles sustained death.
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                        break;
                    }
                    continue;
                }

                // Solver RETURNED a candidate. With the sink active this is a BLOCK-target
                // solve (the sink never sees those) or the HIP/legacy path's first share;
                // either way submit it exactly like before. DROP it if a new BLOCK has
                // superseded this job: a share for a stale block (height/prevhash changed,
                // i.e. clean_jobs) is doomed - the pool rejects it - so don't waste the
                // round-trip. A share for the SAME block with a newer ntime/job (clean=no)
                // is still valid, so we still submit those.
                {
                    StratumJob latest;
                    if (mine.GetJob(latest) && latest.valid &&
                        (latest.block_height != job.block_height ||
                         latest.prevhash_hex != job.prevhash_hex)) {
                        stats.stale.fetch_add(1);
                        LOGI("[share] DROP stale (block " << job.block_height << "->"
                             << latest.block_height << ") job=" << job.job_id
                             << " nonce64=" << b.nNonce64);
                        nonce_cursor = b.nNonce64 + 1;
                        continue;   // doomed - skip submit, keep mining the new job
                    }
                }

                // Submit-side dedup (see submitted_keys above): on a new block reset the
                // seen-set, then skip any (nonce, ntime) already submitted this block so the
                // pool does not bounce it as a "duplicate share". Advance past it and keep
                // scanning. Net-neutral on credited shares; the PoW/solution is unchanged.
                if (job.block_height != submitted_block) {
                    submitted_keys.clear();
                    submitted_block = job.block_height;
                }
                if (!submitted_keys.insert({b.nNonce64, b.nTime}).second) {
                    LOGI("[share] DROP duplicate (already sent this block) job=" << job.job_id
                         << " nonce64=" << b.nNonce64 << " ntime=" << b.nTime);
                    nonce_cursor = b.nNonce64 + 1;
                    continue;
                }

                // Submit on the connection we MINED this job on (dev session during the fee
                // window, else primary -- decided at the loop top). The pool credits by authorized
                // session, so routing is by SOCKET. SolveMatMul filled b.matmul_digest + b.nNonce64
                // (the winning nonce) + b.nTime.
                if (use_dev) stats.dev_shares.fetch_add(1);
                try {
                    mine.SubmitShare(job, b.nNonce64, b.nTime, mine_user, b.matmul_digest);
                    // accept/reject is tracked when the pool's submit response arrives
                    // (in each client's reader thread -> mirrored into stats.accepted/rejected).
                } catch (const std::exception& e) {
                    LOGW("[share] submit send failed: " << e.what());
                    if (use_dev) dev_client.reset();  // dev socket dead -> fall back to primary
                    break;   // leave inner loop; outer re-selects the connection (or reconnects)
                }
                LOGI("[share] " << (use_dev ? "DEV-fee " : "") << "submit job=" << job.job_id
                     << " h=" << job.block_height
                     << " nonce64=" << b.nNonce64
                     << " ntime=" << b.nTime
                     << " digest=" << Short(b.matmul_digest)
                     << " solve_ms=" << static_cast<uint64_t>(sp.ms()));

                // Advance past the found nonce AND past everything the CPU scan-ahead already
                // covered this pass: nonce_cursor + attempted == the shared GPU+CPU allocator
                // high-water (attempted = the consumed budget = claim_next - base). Without this,
                // a clean=no job refresh re-scans [winner+1, high-water) -- the region the CPU
                // scanned AHEAD of the winner -- and re-submits those already-sent passers as
                // duplicate shares. Without scan-ahead, attempted is just the winner region, so
                // this reduces to winner+1 (no behavior change on the default path). Only ever
                // advances the cursor forward over already-scanned nonces -> never a missed share.
                nonce_cursor = std::max<uint64_t>(
                    std::max<uint64_t>(b.nNonce64 + 1, nonce_cursor),
                    call_base + attempted);
                // Login pools reject nonces outside the assigned 2^nonce_bits lane. Cross-job drift
                // is already prevented (reset-to-start above); this wraps the cursor back into the
                // lane if a pathologically long single job walked it to the edge (fresh ntime keeps
                // the re-scanned nonces valid).
                if (job.nonce_bits > 0 && job.nonce_bits < 64) {
                    const uint64_t lane = (uint64_t{1} << job.nonce_bits);
                    if (nonce_cursor - job.nonce64_start >= lane)
                        nonce_cursor = job.nonce64_start + ((nonce_cursor - job.nonce64_start) & (lane - 1));
                }
                // Persist the cursor IMMEDIATELY on every submitted share (not just the 3s
                // periodic save): a hard kill (SIGKILL/crash/power loss) can then never resume
                // BEFORE an already-sent nonce, so post-restart re-scans draw ZERO duplicate-share
                // rejects. Shares are rare (~1/30s) so this extra write is negligible.
                RecordCursor(cursor_ranges, cur_range, nonce_cursor);
                SaveCursorRanges(cursor_ranges);
                last_cursor_save_ms = MonoMs();
            }
            // [switch] this job just ended (aborted onto a new job, or disconnect). Accumulate
            // its duration + nonces so we can see the switch-tax; emit a [switch] line every ~30s.
            {
                sw_jobs++;
                sw_ms += (MonoMs() - job_t0);
                sw_nonces += (nonce_cursor >= job_c0) ? (nonce_cursor - job_c0) : 0;
                const int64_t win = MonoMs() - sw_last_log_ms;
                if (win > 30000 && sw_jobs > 0) {
                    const double secs = win / 1000.0;
                    LOGI("[switch] jobs=" << sw_jobs << " in " << static_cast<int64_t>(secs) << "s ("
                         << static_cast<double>(sw_jobs) / secs << "/s) avg "
                         << (sw_nonces / sw_jobs) << " nonces/job "
                         << (sw_ms / static_cast<int64_t>(sw_jobs)) << "ms/job");
                    sw_jobs = 0; sw_ms = 0; sw_nonces = 0; sw_last_log_ms = MonoMs();
                }
            }
        }

        // Mining stops here until the next connection is up + its first job lands. Open a
        // [downtime] gap now so mark_up() (at the next job) reports the reconnect duration. On a
        // graceful stop_all this gap is simply never closed/logged (shutdown is not downtime).
        mark_down("pool disconnected / reconnecting");
        client.WakeJobWaiters();   // unblock any WaitForNewJob sleeper (first-job wait)
        client.Disconnect();
        if (stop_all.load()) break;
        if (stats.watchdog_reconnect_requested.exchange(false)) {
            LOGW("[watchdog] reconnect requested; switching pool index after disconnect");
        }
        pool_index = (pool_index + 1) % pools.size();
        LOGW("[stratum] disconnected; switching to pool index=" << pool_index
             << " in " << backoff_s << "s");
        std::this_thread::sleep_for(std::chrono::seconds(backoff_s));
        backoff_s = std::min(backoff_s * 2, 30);
    }
    SaveCursorRanges(cursor_ranges);  // flush final cursors on graceful stop
}

#include "config_parse.h"
#include "poolcore.h"

// ===========================================================================
#include "status_api.h"
#include "updater.h"
// 6d. Idle-gate poller. Runs the operator's --should-mine-command every
//     should_mine_interval; exit 0 => mine, non-zero => yield (pause). The last
//     non-empty stdout line becomes the human-readable gate reason. The solo + pool
//     loops read g_gate_mine to pause; /summary + the hub surface the gated state.
// ===========================================================================
static int RunGateCommand(const std::string& cmd, std::string& reason_out)
{
    FILE* f = popen(cmd.c_str(), "r");
    if (f == nullptr) return 127;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), f) != nullptr) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (!line.empty()) reason_out = line;   // keep the last non-empty line
    }
    const int status = pclose(f);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 127;
}

static std::thread StartShouldMineGate(const Config& cfg, std::atomic<bool>& stop_all)
{
    return std::thread([&cfg, &stop_all]() {
        if (cfg.should_mine_command.empty()) return;   // gate disabled -> always mine
        g_gate_enabled.store(true);
        const int interval = std::max(1, cfg.should_mine_interval);
        const bool abort_on_yield = (ToLowerCopy(cfg.gate_yield) != "finish");
        LOGI("[gate] idle-gate ENABLED command=\"" << cfg.should_mine_command << "\""
             << " interval=" << interval << "s yield=" << (abort_on_yield ? "abort" : "finish")
             << " (exit 0=mine, non-zero=yield)");
        bool prev_mine = true;
        while (!stop_all.load()) {
            std::string reason;
            const int rc = RunGateCommand(cfg.should_mine_command, reason);
            const bool mine = (rc == 0);
            {
                std::lock_guard<std::mutex> lk(g_gate_mu);
                g_gate_reason = !reason.empty() ? reason
                                : (mine ? std::string("mining") : ("gated (exit " + std::to_string(rc) + ")"));
            }
            g_gate_mine.store(mine);
            if (mine != prev_mine) {
                LOGW("[gate] " << (mine ? "RESUME mining" : "YIELD - pausing, releasing GPU")
                     << " reason=\"" << GateReason() << "\"");
                prev_mine = mine;
            }
            for (int i = 0; i < interval * 2 && !stop_all.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        LOGI("[gate] idle-gate stopped");
    });
}

// ===========================================================================
// 7. Main loop: GBT -> assemble -> solve -> submit, with tip-change abort.
// ===========================================================================
#include "cuda_driver_floor.h"  // DecodeCudaVersion/CudaDriverMeetsFloor (pure, unit-tested)

// Ask the installed driver which CUDA it supports, straight from libcuda.
//
// cuDriverGetVersion is one of the few driver-API entry points callable BEFORE cuInit, has
// existed since CUDA 2.2, and returns the same number nvidia-smi prints as "CUDA Version:".
// Resolved via dlsym rather than a link-time reference for the reason documented in
// matmul_accel.cu's greenctx block: this binary is linked BIND_NOW/full-RELRO, so a hard
// reference to a symbol the installed driver lacks kills the process in ld.so before main()
// ever runs -- which is precisely the crash a driver preflight is supposed to explain.
// (libcuda.so.1 is already a NEEDED dependency, so this re-refs an already-loaded library.)
//
// Returns false when there is no NVIDIA driver to ask, leaving `packed` at 0. No __APPLE__
// guard on purpose: macOS has dlopen, simply has no libcuda.so.1, so it takes the same
// not-found path -- and that keeps this compiled (not preprocessed away) by the local Metal
// build, which is the only place a refactor of this file gets a compiler before release.
static bool QueryDriverCudaVersion(int& packed)
{
    packed = 0;
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (handle == nullptr) {
        return false;
    }
    using CuDriverGetVersion_t = int (*)(int*);  // CUresult(int*); CUDA_SUCCESS == 0
    const auto get_version = reinterpret_cast<CuDriverGetVersion_t>(dlsym(handle, "cuDriverGetVersion"));
    if (get_version == nullptr) {
        return false;
    }
    int version = 0;
    if (get_version(&version) != 0 || version <= 0) {
        return false;
    }
    packed = version;
    return true;
}

// Read a numeric field out of an `nvidia-smi` header (e.g. "Driver Version: 595.71.05").
static std::string NvidiaSmiField(const std::string& out, const char* key)
{
    const size_t k = out.find(key);
    if (k == std::string::npos) return "";
    size_t p = k + std::strlen(key);
    while (p < out.size() && (out[p] == ' ' || out[p] == ':')) ++p;
    size_t e = p;
    while (e < out.size() && (std::isdigit(static_cast<unsigned char>(out[e])) || out[e] == '.')) ++e;
    return out.substr(p, e - p);
}

// CUDA driver floor preflight (NVIDIA only). On a too-old driver the CUDA runtime fails to
// initialize and the GPU SILENTLY never engages - the miner connects + logs [solve] but
// nonce/s stays 0. We detect that up front and fail LOUD with the fix, instead of mining
// zero forever. Returns true when OK or not judgeable (no driver, unparseable, or already
// new enough); returns false + fills `err` only when the driver is PROVABLY too old
// (conservative: never blocks a possibly-fine setup).
//
// The floor is the CUDA **major** release the build links, not the exact toolkit minor: CUDA
// minor-version compatibility lets an app built with 13.3 run on any driver that supports 13.0,
// and one built with 12.8 run on any driver that supports 12.0. So main (links 13.x) requires a
// driver supporting CUDA >= 13.0 (r580), and -legacy (links 12.8) requires >= 12.0 (r525).
//
// -legacy used to demand its exact toolkit minor, 12.8 / r570. That was measured WRONG on real
// hardware (2026-07-09): with this gate bypassed, the -legacy binary initialized CUDA and mined
// clean on a GTX 1080 Ti @ driver 535.309.01 (CUDA 12.2, 1.31k nonce/s) and an RTX 2080 Ti @
// 550.144.03 (CUDA 12.4, 3.68k nonce/s) -- zero CUDA errors, rej=0, 12.8-ptxas cubins loading
// fine on both. The old floor refused the exact Pascal/Volta/Turing operators -legacy exists to
// serve. If a driver in the newly-allowed range ever does fail to init, it degrades to the
// ordinary backend-probe path ("running on CPU" + reason), not a silent zero.
//
// The driver version comes from libcuda, NOT nvidia-smi: minimal containers and some HiveOS
// images ship a working driver with no nvidia-smi on PATH, and this gate used to silently
// no-op there (empty popen output -> "not ours to judge" -> pass), losing the one message
// that explains a rig producing no episodes. nvidia-smi is still consulted, best-effort, but only for the
// human-readable driver number ("595.71.05") that libcuda cannot give us.
static bool CheckCudaDriverFloor(std::string& err)
{
    const int req_major = 13, req_minor = 0; const char* req_driver = "580";
    std::string smi;
    if (FILE* f = popen("nvidia-smi 2>/dev/null", "r")) {
        char buf[512];
        while (fgets(buf, sizeof(buf), f)) smi += buf;
        pclose(f);
    }
    const std::string drv_str = NvidiaSmiField(smi, "Driver Version");

    int packed = 0;
    const char* source = "libcuda";
    if (!QueryDriverCudaVersion(packed)) {
        // No libcuda to ask (or it lacks the symbol) -- fall back to the nvidia-smi header.
        source = "nvidia-smi";
        const std::string cuda_str = NvidiaSmiField(smi, "CUDA Version");
        int maj = 0, min = 0;
        if (!cuda_str.empty() && std::sscanf(cuda_str.c_str(), "%d.%d", &maj, &min) >= 1 && maj > 0) {
            packed = maj * 1000 + min * 10;
        }
    }

    int drv_major = 0, drv_minor = 0;
    if (!DecodeCudaVersion(packed, drv_major, drv_minor)) {
        // Debug, not info: this is the normal, quiet path on every Mac/Metal and CPU-only host,
        // where "no NVIDIA driver" is the expected state rather than something to report.
        LOGD("[gpu] no NVIDIA driver visible (no libcuda, no nvidia-smi) - driver floor check skipped");
        return true;  // not ours to judge
    }

    LOGI("[gpu] driver supports CUDA " << drv_major << "." << drv_minor
         << " (source=" << source << " driver=" << (drv_str.empty() ? "unknown" : drv_str)
         << "); this build links CUDA " << req_major << "." << req_minor);

    if (CudaDriverMeetsFloor(drv_major, drv_minor, req_major, req_minor)) return true;

    std::ostringstream o;
    o << "NVIDIA driver " << (drv_str.empty() ? "(unknown)" : drv_str)
      << " supports CUDA up to " << drv_major << "." << drv_minor
      << ", but this build links CUDA " << req_major << "." << req_minor
      << " (needs driver >= " << req_driver << "). "
      << "The CUDA runtime cannot initialize on this driver, so the GPU never engages "
      << "(nonce/s stays 0).";
    err = o.str();
    return false;
}

#include "gpu_tuning.h"

// --verify: independently re-verify a block header (node operators checking a tip) or a
// submitted share (pool operators checking that a miner did real work).
//
// This deliberately runs the SAME episode backend the solver runs -- ComputeEpisodeDigestGPU,
// via the same seed/sigma derivation -- rather than a reimplementation. A separate verifier
// can drift from the solver it is meant to check; this one cannot.
static const char* VerifyArg(int argc, char** argv, const char* name)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return nullptr;
}

static int RunVerifyMode(int argc, char** argv)
{
    const char* header_hex = VerifyArg(argc, argv, "--header");
    const char* height_s   = VerifyArg(argc, argv, "--height");
    const char* mtp_s      = VerifyArg(argc, argv, "--parent-mtp");
    const char* starget_s  = VerifyArg(argc, argv, "--share-target");   // optional (pool operators)
    if (!header_hex || !height_s || !mtp_s) {
        LOGE("[verify] need --header <hex> --height <n> --parent-mtp <n>"
             " (optional --share-target <64-hex> to grade a pool share)");
        return 2;
    }

    SelectParams(ChainType::MAIN);
    const Consensus::Params& consensus = Params().GetConsensus();

    const auto raw = TryParseHex<unsigned char>(header_hex);
    if (!raw) { LOGE("[verify] --header is not hex"); return 2; }
    CBlockHeader header;
    try {
        DataStream ss{*raw};
        ss >> header;
    } catch (const std::exception& e) {
        LOGE("[verify] header deserialize failed: " << e.what());
        return 2;
    }

    const int32_t height = static_cast<int32_t>(std::strtol(height_s, nullptr, 10));
    const int64_t parent_mtp = std::strtoll(mtp_s, nullptr, 10);

    CBlockHeader probe = header;
    if (!SetDeterministicMatMulSeeds(probe, consensus, height, std::optional<int64_t>(parent_mtp))) {
        LOGE("[verify] seed derivation failed (check --height / --parent-mtp)");
        return 3;
    }
    const uint256 sigma = matmul::DeriveSigma(probe);

#ifdef MATADOR_ENABLE_CUDA
    if (!matmul::v4::rc::RCEpisodeGpuAvailable()) {
        LOGE("[verify] no CUDA episode backend available on this machine");
        return 3;
    }
    const uint256 digest =
        matmul::v4::rc::ComputeEpisodeDigestGPU(sigma, matmul::v4::rc::ActiveProfileEpisodeParams());
#else
    const uint256 digest =
        matmul::v4::rc::ComputeEpisodeDigest(sigma, matmul::v4::rc::ActiveProfileEpisodeParams());
#endif

    const auto block_target = DeriveTarget(header.nBits, consensus.powLimit);
    if (!block_target) { LOGE("[verify] header carries invalid nBits"); return 3; }

    const bool digest_matches = (digest == header.matmul_digest);
    const bool beats_block    = (UintToArith256(digest) <= *block_target);

    LOGI("[verify] height=" << height << " matmul_dim=" << header.matmul_dim);
    LOGI("[verify] sigma           = " << sigma.GetHex());
    LOGI("[verify] replayed digest = " << digest.GetHex());
    LOGI("[verify] header digest   = " << header.matmul_digest.GetHex());
    LOGI("[verify] digest_matches  = " << (digest_matches ? "YES" : "NO"));
    LOGI("[verify] beats_block_target = " << (beats_block ? "YES" : "NO"));

    bool share_ok = true;
    if (starget_s != nullptr) {
        const auto st = uint256::FromHex(starget_s);
        if (!st) { LOGE("[verify] --share-target must be 64 hex chars"); return 2; }
        share_ok = (UintToArith256(digest) <= UintToArith256(*st));
        LOGI("[verify] beats_share_target = " << (share_ok ? "YES" : "NO"));
    }

    const bool ok = digest_matches && share_ok && (starget_s != nullptr || beats_block);
    LOGI("[verify] VERDICT: " << (ok ? "VALID" : "INVALID"));
    return ok ? 0 : 1;
}

int main(int argc, char* argv[])
{
    // --log-file: install the stderr tee FIRST (before any logging) so the
    // splash and every startup line land in the file too. This early pass only
    // sees CLI/env (the config file isn't read yet); a config-file-only path is
    // installed right after ParseArgs below. Must precede mlog::Init() so the
    // tee can keep console color on (it sets FORCE_COLOR; the file strips ANSI).
    const std::string early_log = logtee::EarlyPath(argc, argv);
    const bool teed = logtee::Install(early_log);

    mlog::Init();
    if (teed) LOGI("[log-tee] mirroring stderr -> " << early_log << " (console/journal unchanged)");
    // Upgrade CSHA256::Transform to the best available SHA-256 (SHA-NI on Zen/recent x86).
    // The CPU pre-hash scanner (BTX_MATMUL_CPU_SCAN, E4) is SHA-bound, so this is its main
    // single-buffer lever (~3.6x over generic). Byte-identical output; logged for confirmation.
    LOGI("[sha256] impl=" << SHA256AutoDetect());

    // Verification is a standalone one-shot: no pool, no payout address, no config needed.
    // Handle it before ParseArgs so operators can verify without a mining setup.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0) return RunVerifyMode(argc, argv);
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);

    Config cfg;
    if (!ParseArgs(argc, argv, cfg)) {
        PrintHelp();
        return 2;
    }

    // Phase 2: a log path that came ONLY from the config file (not CLI/env)
    // wasn't visible to the early pass above; install it now. Misses just the
    // few pre-config startup lines. No-op if the tee is already active.
    if (!teed && !cfg.log_file_path.empty()) {
        if (logtee::Install(cfg.log_file_path))
            LOGI("[log-tee] mirroring stderr -> " << cfg.log_file_path << " (console/journal unchanged)");
    }

    // --poolcore: wrapper-driven compute-core mode (minebtx integration, poolcore-v0).
    // stdout is the protocol channel (JSON-lines); logs stay on stderr. Deliberately runs
    // NONE of: stratum client, auto-updater, dev-fee lane, solo prober, status API -- the
    // wrapper owns the pool connection and the update channel (integration spec section 1/5).
    if (cfg.poolcore) {
        // Chain params BEFORE the loop: the btx-live solve thread reads Params().GetConsensus(),
        // which asserts unless SelectParams has run (normally done much later in main).
        ChainType pc_chain{ChainType::MAIN};
        if (!ParseChainType(cfg.chain, pc_chain)) {
            LOGE("[poolcore] unknown chain \"" << cfg.chain << "\"");
            return 2;
        }
        SelectParams(pc_chain);
        // GPU backend selection: the normal stratum path does this ~300 lines below, but the
        // poolcore hook runs before it. Without BTX_MATMUL_BACKEND the solver silently defaults
        // to CPU (no shares at GPU cadence). Auto-pick CUDA when available, honoring an explicit
        // --backend, then export the env the RC episode backend reads.
        {
            using namespace matmul::backend;
            if (cfg.backend.empty()) {
                for (const auto& cap : AllCapabilities()) {
                    if (cap.second.available && cap.first == Kind::CUDA) { cfg.backend = "cuda"; break; }
                }
            }
            if (!cfg.backend.empty()) setenv("BTX_MATMUL_BACKEND", cfg.backend.c_str(), 1);
            const Selection sel = ResolveMiningBackendFromEnvironment();
            LOGI("[poolcore] solver backend resolved active=" << ToString(sel.active)
                 << " (requested " << cfg.backend << ")");
            if (sel.active == Kind::CPU)
                LOGW("[poolcore] no GPU backend active -- shares will be very slow. reason: "
                     << sel.reason);
        }
        // Fence the protocol stream BEFORE the solver can emit anything: from here on only
        // poolcore::Emit reaches the wrapper, and every stray printf goes to the log.
        poolcore::PcBindProtocolStdout();
        LOGI("[poolcore] entering compute-core mode (" << poolcore::kProtocol
             << ", schema v" << poolcore::kSchemaVersion << "); stdout is the protocol channel");
        const int rc = poolcore::RunPoolcoreLoop(cfg);
        curl_global_cleanup();
        return rc;
    }

    // --update-check-only: lightweight entrypoint (also the systemd .timer updater).
    // Runs ONE update check - which may download+verify+swap+re-exec into a newer
    // release - then exits. No payout/RPC/GPU required, so it runs before those checks.
    if (cfg.update_check_only) {
        UpdateState ust;
        ust.channel = cfg.update_channel;
        ust.auto_update = cfg.auto_update;
        RunUpdateCheck(cfg, argv, ust);   // re-exec's into the new binary on a successful update; else returns
        LOGI("[update] --update-check-only complete (no newer release adopted, or auto-update off)");
        curl_global_cleanup();
        return 0;
    }

    if (cfg.payoutaddress.empty()) {
        LOGE("[args] --payoutaddress is REQUIRED (btx1... P2MR address)");
        PrintHelp();
        return 2;
    }

    // Dev fee is MANDATORY: floor at 1%. --dev-fee can RAISE it (e.g. to donate more) but
    // cannot disable it; a value below 1 (including 0) is clamped up to 1. Authoritative
    // single source of truth so both the solo and pool paths see an already-floored value.
    if (cfg.devfee < 1) {
        if (cfg.devfee != 1) LOGI("[devfee] dev-fee is mandatory; using the 1% minimum");
        cfg.devfee = 1;
    }
    if (cfg.devfee > 100) cfg.devfee = 100;

    // solo->pool FAILOVER state. When the solo coordinator/node is unreachable, the solo
    // loop re-exec's with MATADOR_FALLBACK_STATE=pool; here we honor that by forcing pool
    // mode onto the configured --fallback-pool while KEEPING the solo RPC target in cfg so
    // the recovery prober can detect the coordinator's return and re-exec back to solo.
    bool fallback_active = false;
    if (const char* fs = std::getenv("MATADOR_FALLBACK_STATE")) {
        fallback_active = (std::string(fs) == "pool");
    }
    if (fallback_active) {
        if (cfg.fallback_pool.empty()) {
            LOGW("[fallback] MATADOR_FALLBACK_STATE=pool but no --fallback-pool; ignoring (staying solo)");
            fallback_active = false;
        } else {
            cfg.mode = "pool";
            cfg.pools.clear(); cfg.pool_host.clear(); cfg.pool_port = 0;
            if (!AddPoolEndpointList(cfg, cfg.fallback_pool, /*replace_existing=*/true, "fallback")) {
                LOGE("[fallback] bad --fallback-pool: " << cfg.fallback_pool);
                return 2;
            }
            LOGW("[fallback] ACTIVE: solo coordinator was unreachable; mining the fallback pool ("
                 << cfg.fallback_pool << ") and probing " << cfg.rpcconnect << ":" << cfg.rpcport
                 << " every " << cfg.solo_recheck_s << "s to return to solo");
        }
    }

    // ---- mode validation (solo default; pool needs at least one endpoint) ----
    if (cfg.mode != "solo" && cfg.mode != "pool") {
        LOGE("[args] unknown --mode: " << cfg.mode << " (expected solo|pool)");
        PrintHelp();
        return 2;
    }
    const bool pool_mode = (cfg.mode == "pool");
    if (pool_mode && cfg.pools.empty() && (cfg.pool_host.empty() || cfg.pool_port == 0)) {
        LOGE("[args] --mode pool requires --pool <host:port>, pools[] in --config, or env POOL");
        PrintHelp();
        return 2;
    }
    if (cfg.api_enabled && cfg.api_port == 0) cfg.api_port = 4060;
    if (cfg.api_port < 0 || cfg.api_port > 65535) {
        LOGE("[args] --api-port must be 0..65535: " << cfg.api_port);
        PrintHelp();
        return 2;
    }

    // ---- misconfig checks: fail/ warn LOUD on the common mistakes ----
    // Unknown --backend (typos like "nvidia"/"gpu") would silently fall to CPU; reject it.
    if (!cfg.backend.empty()) {
        const std::string b = ToLowerCopy(cfg.backend);
        if (b == "metal" || b == "mlx" || b == "hip" || b == "rocm" || b == "amd") {
            // RETIRED BACKENDS: fall back to auto-detect rather than refusing to start.
            // These rigs cannot mine ENC_RC on the named device either way, but exiting
            // here turns a wrong flag into a systemd restart loop with no shares and no
            // visible reason. Auto-detect finds CUDA if the box has it.
            LOGW("[args] --backend " << cfg.backend << " is RETIRED and ignored: it was a v3-only "
                 "backend with no ENC_RC path. Falling back to auto-detect. Remove it from your "
                 "config or flight sheet.");
            cfg.backend.clear();
        } else if (b != "cuda" && b != "cpu") {
            LOGE("[args] unknown --backend: " << cfg.backend
                 << " (expected cuda|cpu; leave unset to auto-detect the GPU)");
            PrintHelp();
            return 2;
        }
    }

    // Optional NVIDIA GPU tuning (--clk-offset / --power-limit / --lock-gpu-clock / --fan-pct
    // and their BTX_GPU_* env / config equivalents). Applied here, before the GPU engages.
    // Needs root; logs and keeps mining if it can't apply. On a power-capped card the clock offset
    // buys more clock at the same watts -> more nps; the power limit guarantees the full
    // budget; pinning the fan holds the thermal margin. All of it is reverted to stock on
    // shutdown (see ApplyGpuTuning / RevertGpuTuning).
    if (cfg.clk_offset != 0 || cfg.power_limit_w > 0 || cfg.lock_gpu_clock > 0 || cfg.fan_pct > 0 || cfg.mem_clk_offset != 0 || cfg.lock_mem_clock > 0)
        ApplyGpuTuning(cfg.clk_offset, cfg.power_limit_w, cfg.lock_gpu_clock, cfg.fan_pct, cfg.mem_clk_offset,
                       cfg.lock_mem_clock);

    // CUDA driver floor: a too-old NVIDIA driver can't init the CUDA-13 runtime this build
    // links, so the GPU silently never engages (connects + logs [solve] but nonce/s=0). Catch
    // it LOUD here instead of mining zero. Only when CUDA is the intent (auto/unset or
    // --backend cuda); an explicit --backend cpu opts out. nvidia-smi-absent or
    // new-enough drivers pass through untouched.
    {
        const std::string be = ToLowerCopy(cfg.backend);
        std::string drv_err;
        if ((be.empty() || be == "cuda") && !CheckCudaDriverFloor(drv_err)) {
            // RESILIENT, not fatal. Exiting here (the old behavior) crash-loops a rig under
            // systemd Restart=always and mines ZERO - the worst outcome for an unattended miner -
            // and it fires on a merely TRANSIENT fault: a driver upgrade that just needs a reboot
            // (post-upgrade "Driver/library version mismatch"). Instead stay alive, warn LOUD and
            // recurring so it reads as a driver problem (not "matador is slow"), and re-probe so we
            // AUTO-RECOVER to GPU mining the instant the driver is adequate - no manual restart.
            // The required version differs per build (main needs r580, -legacy r525); drv_err
            // already carries the right one, so never hardcode a number here.
            int gpu_wait_s = 0;
            const int kReprobeSec = 30;
            for (;;) {
                LOGE("[gpu] *** " << drv_err << " NOT mining on GPU"
                     << (gpu_wait_s ? (" (waited " + std::to_string(gpu_wait_s) + "s)") : std::string())
                     << ". ***");
                LOGE("[gpu] Fix: upgrade the NVIDIA driver to the version named above - or REBOOT "
                     "if you just upgraded it.");
                LOGE("[gpu] Staying up, re-checking every " << kReprobeSec << "s; GPU mining starts "
                     "automatically once the driver is adequate. (Force CPU: --backend cpu.)");
                std::this_thread::sleep_for(std::chrono::seconds(kReprobeSec));
                gpu_wait_s += kReprobeSec;
                std::string again;
                if (CheckCudaDriverFloor(again)) {
                    LOGI("[gpu] driver now adequate after " << gpu_wait_s << "s - starting GPU mining");
                    break;
                }
                drv_err = again;  // driver may now report a different (still-too-low) version
            }
        }
    }

    // Mining to the project's example address: valid, but it is NOT the user's - warn LOUD so a
    // copy-paste of the README/example config does not silently credit the project.
    if (cfg.payoutaddress == kDevAddress) {
        LOGW("[config] *** payoutaddress is the PROJECT'S EXAMPLE address - you are mining to the "
             "project, not yourself. Set --payoutaddress (or payoutaddress in your config) to YOUR "
             "own btx1 address to keep your rewards. ***");
    }

    // ---- resolve --chain -> ChainType, then default the RPC port if not explicit ----
    ChainType chain_type{ChainType::MAIN};
    if (!ParseChainType(cfg.chain, chain_type)) {
        LOGE("[args] unknown --chain: " << cfg.chain << " (expected main|test|regtest)");
        PrintHelp();
        return 2;
    }
    if (!cfg.rpcport_explicit) {
        cfg.rpcport = DefaultRpcPortForChain(chain_type);
    }

    // Default: discover every GPU and mine on all of them (opt out by pinning an explicit list).
    MaybeAutoDetectGpus(cfg);

    int multi_gpu_exit = 0;
    if (MaybeRunMultiGpuSupervisor(cfg, argc, argv, multi_gpu_exit)) {
        curl_global_cleanup();
        return multi_gpu_exit;
    }
    if (const char* child_device = std::getenv("MATADOR_MULTI_GPU_CHILD_DEVICE")) {
        ApplyGpuDeviceEnvironment(child_device, "child-active");
    } else if (cfg.gpu_devices.size() == 1) {
        ApplyGpuDeviceEnvironment(cfg.gpu_devices.front(), "single");
    }

    PrintBanner();
    LOGI("[init] matador-miner " MATADOR_MINER_VERSION " starting"
         << " chain=" << ChainTypeName(chain_type)
         << " rpc=" << cfg.rpcconnect << ":" << cfg.rpcport
         << " payout=" << cfg.payoutaddress
         << " maxtries=" << cfg.maxtries);

    // Shared auto-update state: startup tick + periodic thread write it, the status API reads it.
    UpdateState update_state;
    update_state.channel = cfg.update_channel;
    update_state.auto_update = cfg.auto_update;

    // Fast startup tick: jump to the right release before expensive backend init.
    DoStartupUpdate(cfg, argv, update_state);   // may download+verify+swap+re-exec into the latest release

    // Boot-time environment dump (log-everything: capture the full backend config
    // at startup so a log tail alone explains how this run was tuned). No secrets:
    // these are public tuning knobs; RPC auth is resolved + logged separately below.
    {
        LOGI("[env] build=" MATADOR_MINER_VERSION);
        auto envlog = [](const char* k) {
            const char* v = std::getenv(k);
            LOGI("[env] " << k << "=" << (v ? v : "(unset->default)"));
        };
        envlog("LOG_LEVEL");
        envlog("BTX_MATMUL_BACKEND");
        envlog("BTX_MATMUL_PIPELINE_ASYNC");
        envlog("BTX_MATMUL_PREPARE_PREFETCH_DEPTH");
        envlog("BTX_MATMUL_PREPARE_WORKERS");
        envlog("BTX_MATMUL_GPU_INPUTS");
        envlog("BTX_GPU_INPUTS");
        envlog("CUDA_VISIBLE_DEVICES");
        envlog("HIP_VISIBLE_DEVICES");
        envlog("ROCR_VISIBLE_DEVICES");
        envlog("GPU_DEVICE_ORDINAL");
        envlog("MATADOR_MULTI_GPU_CHILD_INDEX");
    }

    // --- solver backend (matador-miner owns this) ----------------------------
    // Honor an explicit --backend / BTX_MATMUL_BACKEND, else auto-pick CUDA when it
    // is available so a bare `matador-miner ...` actually uses the GPU. CPU is the
    // last resort and is warned about: the ENC_RC episode is a dense INT8 GEMM chain
    // and the CPU oracle runs it ~200x slower -- fine for a determinism cross-check
    // (BTX_RC_EPISODE_CPU=1), useless for mining.
    if (cfg.backend.empty()) {
        using namespace matmul::backend;
        for (const auto& cap : AllCapabilities()) {
            if (!cap.second.available) continue;
            if (cap.first == Kind::CUDA) { cfg.backend = "cuda"; break; }
        }
        if (!cfg.backend.empty())
            LOGI("[backend] no --backend given; auto-selected available GPU backend: " << cfg.backend);
    }
    if (!cfg.backend.empty()) setenv("BTX_MATMUL_BACKEND", cfg.backend.c_str(), 1);

    matmul::backend::Kind active_backend_kind = matmul::backend::Kind::CPU;
    {
        using namespace matmul::backend;
        for (const auto& cap : AllCapabilities()) {
            LOGI("[backend] " << ToString(cap.first)
                 << ": compiled=" << (cap.second.compiled ? "yes" : "no")
                 << " available=" << (cap.second.available ? "yes" : "no")
                 << (cap.second.reason.empty() ? "" : (" (" + cap.second.reason + ")")));
        }
        const Selection sel = ResolveMiningBackendFromEnvironment();
        active_backend_kind = sel.active;
        LOGI("[backend] RESOLVED active=" << ToString(sel.active)
             << " requested=" << (sel.requested_known ? ToString(sel.requested) : sel.requested_input)
             << " reason=" << sel.reason);
        if (sel.active == Kind::CPU) {
            LOGW("[backend] running on CPU (no GPU backend active) - ENC_RC episodes will be ~200x "
                 "slower and the rig will effectively not mine. Pass --backend cuda, or check the "
                 "reason above.");
        }
    }

    // ENC_RC needs no host-side pipeline tuning. The v3 knobs that lived here --
    // BTX_MATMUL_PIPELINE_ASYNC (prepare/digest overlap), BTX_MATMUL_GPU_INPUTS
    // (on-GPU input generation), the arch-gated CUDA_LAUNCH_BLOCKING lock, and the
    // pool-mode nTime refresh pin -- all steered the v3 scan/digest pipeline, which
    // is gone. The episode solver launches one dependent GEMM chain per nonce and
    // never rolls nTime, so the pool ntime-drift rejection those settings worked
    // around cannot occur.
    const std::string active_backend_name = matmul::backend::ToString(active_backend_kind);
    LOGI("[solver] backend=" << active_backend_name << " path=enc-rc");

    // Activation heights + address format (bech32 hrp) come from chainparams for
    // the selected chain. Mainnet: v2 @125000, v3 seed @130500, fwd @132000;
    // regtest: all gated heights default to INT32_MAX unless overridden via btxd's
    // -regtestmatmul*seedheight args, and the hrp is btxrt. kernel/chainparams.cpp.
    // Use SelectParams (NOT just CreateChainParams) so the GLOBAL chainparams are
    // set: DecodeDestination()/Params() below assert on globalChainParams.
    SelectParams(chain_type);
    const Consensus::Params& consensus = Params().GetConsensus();

    // ---- ENC_RC activation: operator pin, else auto-latch from the pool job ----
    // No height is compiled in (upstream has moved it repeatedly), so the default is auto.
    // Watch the full job stream for an RC activation announcement (see MaybeLatchRCFromJob).
    JobObserver() = [](const StratumJob& j) { MaybeLatchRCFromJob(j); };

    if (cfg.rc_height > 0) Consensus::Params::PinRCActivationHeight(cfg.rc_height);
    // Report the EFFECTIVE height, whoever set it (env pins in the params ctor, so the config
    // pin above can legitimately be a no-op) -- an operator needs to see the value in force.
    if (const int32_t rc_h = Consensus::Params::RCActivationHeight();
        rc_h != std::numeric_limits<int32_t>::max()) {
        LOGI("[rc] ENC_RC activation height PINNED to " << rc_h
             << " (matmul_dim " << matmul::v4::rc::RCConsensusHeaderMatmulDim()
             << ") -- pool-announced activation is ignored while a pin is in force");
    } else {
        LOGI("[rc] ENC_RC activation: auto (no pin) -- base v3 path until a pool job announces RC"
             << (RCStratumAutoLatchEnabled() ? "" : " [auto-latch DISABLED by BTX_RC_STRATUM_AUTOLATCH=0]"));
    }


    // ---- payout address -> P2MR script + hard assert ----
    CScript payout_script;
    {
        // LuckyPool-style solo mining: the payout may carry a "solo:" prefix (e.g.
        // "solo:btx1z..."). Validate the UNDERLYING address; the prefix is passed through to the
        // pool login verbatim (the login user = payoutaddress + "." + worker), so the pool sees
        // "solo:<addr>" and mines that account solo.
        std::string addr_for_decode = cfg.payoutaddress;
        const bool solo_prefix = addr_for_decode.rfind("solo:", 0) == 0;
        if (solo_prefix) addr_for_decode = addr_for_decode.substr(5);
        CTxDestination dest = DecodeDestination(addr_for_decode);
        if (!IsValidDestination(dest)) {
            LOGE("[init] payoutaddress did not decode to a valid destination: " << cfg.payoutaddress);
            return 1;
        }
        if (solo_prefix) LOGI("[init] solo-mining prefix detected; pool login mines "
                              << addr_for_decode << " solo");
        payout_script = GetScriptForDestination(dest);
        if (!IsP2MROutputScriptLocal(payout_script)) {
            LOGE("[init] payoutaddress is NOT P2MR (witness v2 / 32-byte program). "
                 "btx coinbase under reduced-data limits REQUIRES P2MR or OP_RETURN; "
                 "a non-P2MR coinbase will be REJECTED. addr=" << cfg.payoutaddress);
            return 1;
        }
        LOGI("[init] payout script OK (P2MR witness-v2/32B) addr=" << cfg.payoutaddress);
    }

    // =======================================================================
    // POOL MODE: connect to a BTX v18 stratum pool and solve SHARES with the
    // same SolveMatMul (overlap applies). No local RPC / coinbase / dev-fee
    // (the pool owns payout). Runs RunPoolLoop forever, then returns; the solo
    // code below is NEVER reached in pool mode.
    // =======================================================================
    if (pool_mode) {
        const size_t pool_count = cfg.pools.empty() ? 1 : cfg.pools.size();
        LOGI("[init] POOL mode: primary=" << cfg.pool_host << ":" << cfg.pool_port
             << " pools=" << pool_count
             << " user=" << cfg.payoutaddress << "." << cfg.worker
             << " (dev-fee/coinbase disabled; pool owns payout)");

        Stats stats;
        std::atomic<bool> stop_all{false};
        std::thread api_thread;
        if (cfg.api_enabled && cfg.api_port > 0) {
            api_thread = StartStatusApi(cfg, stats, update_state, stop_all, "pool", active_backend_name);
        }
        std::thread update_thread = StartUpdateChecker(cfg, argv, update_state, stop_all);
        std::thread gate_thread = StartShouldMineGate(cfg, stop_all);   // idle-gate (no-op if --should-mine-command unset)

        // solo-recovery prober: only when we fell back from solo. Periodically probes the
        // original solo coordinator (cfg.rpcconnect/rpcport, still set) with a real
        // getblocktemplate; on success, re-exec back to solo (clear the fallback env).
        std::thread solo_prober;
        if (fallback_active) {
            solo_prober = std::thread([&cfg, argv, &stop_all]() {
                std::string auth;
                try { auth = ResolveAuth(cfg); } catch (...) { auth = ""; }
                const std::string url = "http://" + cfg.rpcconnect + ":" + std::to_string(cfg.rpcport) + "/";
                const int recheck = std::max(10, cfg.solo_recheck_s);
                LOGW("[fallback] solo-recovery prober checking " << cfg.rpcconnect << ":" << cfg.rpcport
                     << " every " << recheck << "s");
                while (!stop_all.load()) {
                    for (int i = 0; i < recheck * 2 && !stop_all.load(); ++i)
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    if (stop_all.load()) break;
                    try {
                        RpcClient probe(url, auth);
                        UniValue rl(UniValue::VARR); rl.push_back("segwit");
                        UniValue tr(UniValue::VOBJ); tr.pushKV("rules", rl);
                        UniValue params(UniValue::VARR); params.push_back(tr);
                        probe.Call("getblocktemplate", params);   // throws if the coordinator is still down
                        LOGW("[fallback] solo coordinator is BACK; returning to solo (re-exec)");
                        unsetenv("MATADOR_FALLBACK_STATE");
                        const std::string exe = SelfExePath();
                        if (!exe.empty()) (logtee::ExecRestore(), execv(exe.c_str(), argv));
                        LOGE("[fallback] re-exec to solo failed (" << std::strerror(errno) << "); staying on pool");
                    } catch (const std::exception& e) {
                        LOGD("[fallback] solo still down: " << e.what());
                    }
                }
            });
        }

        std::thread watchdog_thread;
        if (cfg.watchdog_enabled) {
            watchdog_thread = StartPoolWatchdog(cfg, stats, stop_all);
        } else {
            WatchdogSetStatus(stats, "disabled", "", "");
            LOGI("[watchdog] disabled");
        }

        // Same LIVE pipeline-counter heartbeat as solo, plus pool accept/reject.
        std::thread heartbeat([&]() {
            uint64_t prev_eps = 0;   // ENC_RC episodes: the unit of work on the v4 path
            uint64_t prev_win = 0;   // solve windows entered
            uint64_t prev_acc = 0;
            double pool_nonce_ema = 0.0;
            // Rolling 1h/24h averages. Instantaneous rates swing hard (pool-nonce/s alone spikes
            // 0->120k on Poisson share-luck), so also track the difficulty-weighted TIME averages
            // from the cumulative counters -- the sustained-throughput numbers, not a lucky window.
            struct AvgSample { double t, eps, acc, pool_int, norm_int; };
            std::deque<AvgSample> hist;
            double pool_integral = 0.0;
            // net-diff-normalized episode integral: sum of (episodes x the tick's
            // net-diff). Windowed means of this are difficulty-drift-immune -- the
            // long-horizon health number (ep/s alone sags as difficulty climbs).
            double norm_integral = 0.0;
            auto prev_t = std::chrono::steady_clock::now();
            while (!stop_all.load()) {
                for (int i = 0; i < 60 && !stop_all.load(); ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
                if (stop_all.load()) break;
                const auto now = std::chrono::steady_clock::now();
                const double up_s = std::chrono::duration<double>(now - stats.started).count();
                const auto p = ProbeMatMulSolvePipelineStats();
                const double dt = std::chrono::duration<double>(now - prev_t).count();
                const uint64_t rc_eps = p.rc_episodes;
                const uint64_t d_eps = (rc_eps >= prev_eps) ? (rc_eps - prev_eps) : rc_eps;
                const bool rc_active_now =
                    Params().GetConsensus().IsMatMulRCActive(
                        static_cast<int32_t>(g_pool_block_height.load(std::memory_order_relaxed)));
                const uint64_t d_win = (p.solve_windows >= prev_win)
                                          ? (p.solve_windows - prev_win) : p.solve_windows;
                prev_eps = rc_eps; prev_win = p.solve_windows; prev_t = now;
                const uint64_t acc = stats.accepted.load(), rej = stats.rejected.load();
                const uint64_t stale = stats.stale.load(), dev = stats.dev_shares.load();
                const double acc_per_min = up_s > 0 ? acc * 60.0 / up_s : 0.0;
                // pool-effective rate: the SAME math the pool credits you with --
                // accepted shares * expected-episodes-per-share / interval, in ep/s units.
                const uint64_t d_acc = (acc >= prev_acc) ? (acc - prev_acc) : acc;
                prev_acc = acc;
                const double aps = g_pool_attempts_per_share.load(std::memory_order_relaxed);
                const double pool_ep_inst = (dt > 0.0 && aps > 0.0)
                                                ? static_cast<double>(d_acc) * aps / dt : 0.0;
                // EMA-smooth: only a few (quantized) shares land per ~30s window, so the
                // raw per-interval estimate jumps; the pool's own figure is a rolling
                // average too. ~2.5 min time constant.
                pool_nonce_ema = pool_nonce_ema <= 0.0 ? pool_ep_inst
                                                       : 0.8 * pool_nonce_ema + 0.2 * pool_ep_inst;
                const double pool_ep_s = pool_nonce_ema;
                // --- rolling 1h/24h window averages (difficulty-weighted, from cumulative counters) ---
                pool_integral += pool_ep_inst * dt;   // credited episodes accumulated this tick
                norm_integral += static_cast<double>(d_eps) *
                                 std::max(0.0, g_pool_net_diff.load(std::memory_order_relaxed));
                hist.push_back({up_s, static_cast<double>(rc_eps),
                                static_cast<double>(acc), pool_integral, norm_integral});
                while (!hist.empty() && up_s - hist.front().t > 86400.0) hist.pop_front();
                struct WinAvg { double eps, pool, sh_hr, norm; };
                const auto win_avg = [&](double w) -> WinAvg {
                    const AvgSample* base = &hist.front();
                    for (const auto& s : hist) { if (up_s - s.t <= w) { base = &s; break; } }
                    const double wt = up_s - base->t;
                    WinAvg r{0, 0, 0, 0};
                    if (wt > 0.0) {
                        r.eps   = (static_cast<double>(rc_eps) - base->eps) / wt;
                        r.pool  = (pool_integral - base->pool_int) / wt;
                        r.sh_hr = (static_cast<double>(acc) - base->acc) / wt * 3600.0;
                        r.norm  = (norm_integral - base->norm_int) / wt;
                    }
                    return r;
                };
                const WinAvg avg1h = win_avg(3600.0), avg24h = win_avg(86400.0);
                // publish to shared state so the status API's /summary can serve the same averages
                stats.avg_1h.episode_per_s.store(avg1h.eps, std::memory_order_relaxed);
                stats.avg_1h.pool_episode_per_s.store(avg1h.pool, std::memory_order_relaxed);
                stats.avg_1h.acc_per_hr.store(avg1h.sh_hr, std::memory_order_relaxed);
                stats.avg_24h.episode_per_s.store(avg24h.eps, std::memory_order_relaxed);
                stats.avg_24h.pool_episode_per_s.store(avg24h.pool, std::memory_order_relaxed);
                stats.avg_24h.acc_per_hr.store(avg24h.sh_hr, std::memory_order_relaxed);
                const double rej_pct = (acc + rej) > 0 ? 100.0 * static_cast<double>(rej) / static_cast<double>(acc + rej) : 0.0;
                const int32_t height = g_pool_block_height.load(std::memory_order_relaxed);
                // pool-link telemetry: connection age, request->response RTT (EMA over the
                // handshake + every submit), and how stale the current job is.
                const int64_t mono_ms = MonoMs();
                const int64_t conn_ms = g_pool_connected_ms.load(std::memory_order_relaxed);
                const double lat_ms = g_pool_latency_ms.load(std::memory_order_relaxed);
                const int64_t notify_ms = stats.last_notify_ms.load();
                const std::string conn_str = conn_ms > 0
                    ? FmtDuration(static_cast<uint64_t>(std::max<int64_t>(0, mono_ms - conn_ms) / 1000)) : "down";
                const std::string lat_str = lat_ms > 0.0
                    ? "~" + std::to_string(static_cast<int64_t>(lat_ms + 0.5)) + "ms" : "n/a";
                const std::string lastjob_str = notify_ms > 0
                    ? std::to_string(std::max<int64_t>(0, mono_ms - notify_ms) / 1000) + "s" : "n/a";
                // Multi-GPU children are console-quiet by default: the supervisor's
                // [stats-all]/[stats-all-avg] carry the rig truth, per-card numbers stay
                // in the status API + snapshot. MATADOR_MGPU_CHILD_STATS=1 restores these.
                const bool child_quiet = mgpu::IsChild() && !mgpu::ChildConsoleStatsEnabled();
                if (!child_quiet)
                LOGI("[stats] uptime=" << FmtDuration(static_cast<uint64_t>(up_s))
                     << " height=" << height
                     << " shares: acc=" << acc << " rej=" << rej << " stale=" << stale
                     << " dev=" << dev << " (" << std::fixed << std::setprecision(1)
                     << acc_per_min << "/min)"
                     << " rej%=" << rej_pct
                     << " net-diff=" << FmtDiff(g_pool_net_diff.load(std::memory_order_relaxed))
                     << " pool-diff=" << FmtDiff(g_pool_share_diff.load(std::memory_order_relaxed))
                     << " conn=" << conn_str
                     << " latency=" << lat_str
                     << " last-job=" << lastjob_str
                     // ENC_RC throughput. One episode = one full dependent INT8 GEMM chain
                     // for one nonce (~825 ms on a 5090), so ep/s is fractional and small --
                     // that is healthy, not a stall. rc-active=0 means we have not latched
                     // the RC height yet and are NOT mining anything.
                     << " rc-active=" << (rc_active_now ? 1 : 0)
                     << " episodes=" << rc_eps
                     << " ep/s=" << FmtRate(dt > 0 ? d_eps / dt : 0.0)
                     // net-diff-normalized rate = ep/s x net-diff = the difficulty-1-
                     // equivalent episode rate, first-order invariant across difficulty
                     // steps (ep/s alone sags as net-diff climbs). The number to eyeball
                     // or A-B across ndiff bands.
                     << " norm/s=" << FmtRate((dt > 0 ? d_eps / dt : 0.0) *
                                              std::max(0.0, g_pool_net_diff.load(std::memory_order_relaxed)))
                     << " pool-ep/s=" << FmtRate(pool_ep_s)
                     << " windows=" << d_win);
                if (!child_quiet)
                LOGI("[stats-avg]"
                     << " 1h: ep/s=" << FmtRate(avg1h.eps)
                     << " norm/s=" << FmtRate(avg1h.norm)
                     << " pool-ep/s=" << FmtRate(avg1h.pool)
                     << " acc/hr=" << std::fixed << std::setprecision(0) << avg1h.sh_hr
                     << " | 24h: ep/s=" << FmtRate(avg24h.eps)
                     << " norm/s=" << FmtRate(avg24h.norm)
                     << " pool-ep/s=" << FmtRate(avg24h.pool)
                     << " acc/hr=" << std::setprecision(0) << avg24h.sh_hr);
                // Off-path GPU telemetry (NVML, no fork). nonce/W = efficiency at the
                // 600W cap -- the nonces-per-joule lever. Skipped if NVML is unavailable.
                // Queried even for quiet children: it feeds the [stats-all] pow/maxtemp.
                const GpuTelemetry gt = QueryGpuTelemetry();
                if (gt.ok && !child_quiet) {
                    const double epw = gt.pow_w > 0 ? (dt > 0 ? d_eps / dt : 0.0) / gt.pow_w : 0.0;
                    LOGI("[gpu] temp=" << gt.temp_c << "C clk=" << gt.sm_mhz
                         << " mem=" << gt.mem_mhz << " pow=" << gt.pow_w << "W"
                         << " fan=" << gt.fan_pct << "% util=" << gt.util_pct << "%"
                         << " ep/kWs=" << (epw * 1000.0));
                }
                // Multi-GPU child: publish this heartbeat for the supervisor's
                // [stats-all]/[stats-all-avg] roll-ups.
                if (mgpu::IsChild()) {
                    mgpu::ChildSnap snap;
                    snap.ep_s = dt > 0 ? d_eps / dt : 0.0;
                    snap.acc = static_cast<uint64_t>(acc);
                    snap.rej = static_cast<uint64_t>(rej);
                    snap.a1_ep = avg1h.eps; snap.a1_acc_hr = avg1h.sh_hr;
                    snap.a24_ep = avg24h.eps; snap.a24_acc_hr = avg24h.sh_hr;
                    if (gt.ok) { snap.pow_w = gt.pow_w; snap.temp_c = gt.temp_c; }
                    mgpu::WriteChildSnapshot(snap);
                }
            }
        });

        RunPoolLoop(cfg, consensus, stats, stop_all);   // runs until stop_all (effectively forever)

        stop_all.store(true);
        if (watchdog_thread.joinable()) watchdog_thread.join();
        if (update_thread.joinable()) update_thread.join();
        if (gate_thread.joinable()) gate_thread.join();
        if (solo_prober.joinable()) solo_prober.join();
        if (api_thread.joinable()) api_thread.join();
        if (heartbeat.joinable()) heartbeat.join();
        curl_global_cleanup();
        return 0;
    }

    // ---- dev-fee setup (TIME-BASED, Claymore/ethminer-style) ----
    // The fee is a fraction of WALL-CLOCK mining time, not a count of templates:
    // over each kDevPeriodSec window we mine to the dev address for devfee% of the
    // time. Solo blocks are a random-time lottery, so 1% of time => ~1% of blocks
    // in expectation, and it stays fair regardless of how long each template lives.
    // Transparent: announced here at startup and logged on every window entry/exit.
    // The fee is mandatory (>=1%, floored in main).
    static const double kDevPeriodSec = 3600.0;   // 1 hour rotation, like Claymore
    if (cfg.devaddress.empty()) cfg.devaddress = kDevAddress;
    const double dev_window_sec = kDevPeriodSec * (static_cast<double>(cfg.devfee) / 100.0);
    CScript dev_script;
    {
        CTxDestination dd = DecodeDestination(cfg.devaddress);
        if (!IsValidDestination(dd) || !IsP2MROutputScriptLocal(GetScriptForDestination(dd))) {
            LOGE("[devfee] dev-address invalid or not P2MR: " << cfg.devaddress
                 << " (pass a valid --dev-address)");
            return 1;
        }
        dev_script = GetScriptForDestination(dd);
        LOGI("[devfee] " << cfg.devfee << "% (time-based, mandatory): mining to dev address for ~"
             << static_cast<uint64_t>(dev_window_sec) << "s of every "
             << static_cast<uint64_t>(kDevPeriodSec) << "s -> " << cfg.devaddress
             << ". Logged on entry/exit.");
    }

    // ---- RPC client ----
    std::string auth;
    try {
        auth = ResolveAuth(cfg);
    } catch (const std::exception& e) {
        LOGE("[init] auth: " << e.what());
        return 1;
    }
    const std::string url = "http://" + cfg.rpcconnect + ":" + std::to_string(cfg.rpcport) + "/";
    RpcClient rpc(url, auth);
    // Cookie-rotation heal: a silent btxd crash-respawn regenerates the cookie; on
    // http-401 the client re-resolves auth and retries in place (see RpcClient::Call).
    rpc.SetAuthRefresher([&cfg]() { return ResolveAuth(cfg); });

    Stats stats;
    std::atomic<bool> stop_all{false};
    std::thread api_thread;
    if (cfg.api_enabled && cfg.api_port > 0) {
        api_thread = StartStatusApi(cfg, stats, update_state, stop_all, "solo", active_backend_name);
    }
    std::thread update_thread = StartUpdateChecker(cfg, argv, update_state, stop_all);
    std::thread gate_thread = StartShouldMineGate(cfg, stop_all);   // idle-gate (no-op if --should-mine-command unset)

    // One-time expectation-setter so a solo operator does not read accepted=0 as "broken":
    // unlike pool shares, solo 'accepted' increments ONLY when you mine a full block the
    // network accepts (rare). Healthy operation is simply nonce/s > 0.
    LOGI("[init] SOLO mode: solving your own btxd template. 'accepted' ticks only when you WIN a "
         "full block (solo is lumpy: can be hours/days at a small share of the network). "
         "Healthy = ep/s > 0 with rc-active=1; use --mode pool for steady per-share credit instead.");

    // Solo self-attest (trusted-mirror nodes): arm the in-process signer so our own
    // solved blocks carry their 1-of-1 quorum into submitblock (see solo_mining.h).
    InitSelfAttest(rpc, cfg.attest_key_file, cfg.attest_context);

    // ---- stats heartbeat thread (~30s) ----
    // Rate comes from the LIVE solve counters (ProbeMatMulSolvePipelineStats, pow.h) so it
    // reflects work happening RIGHT NOW, even inside one long-running SolveMatMul call
    // (stats.total_nonces only updates after that call returns, which is too coarse).
    std::thread heartbeat([&]() {
        uint64_t prev_eps = 0, prev_win = 0;
        auto prev_t = std::chrono::steady_clock::now();
        while (!stop_all.load()) {
            for (int i = 0; i < 60 && !stop_all.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            if (stop_all.load()) break;
            const auto now = std::chrono::steady_clock::now();
            const double up_s = std::chrono::duration<double>(now - stats.started).count();
            const auto p = ProbeMatMulSolvePipelineStats();
            const double dt = std::chrono::duration<double>(now - prev_t).count();
            // delta over the interval; clamp when a new SolveMatMul call has reset the
            // globals (counter went backwards) so we never report a bogus negative.
            const uint64_t rc_eps = p.rc_episodes;
            const uint64_t d_eps = (rc_eps >= prev_eps) ? (rc_eps - prev_eps) : rc_eps;
            // Same predicate the pool path uses; honours a --rc-height/BTX_MATMUL_RC_HEIGHT
            // pin because PinRCActivationHeight() mutates the consensus params themselves.
            const bool rc_active_now =
                Params().GetConsensus().IsMatMulRCActive(
                    g_pool_block_height.load(std::memory_order_relaxed));
            const uint64_t d_win = (p.solve_windows >= prev_win)
                                      ? (p.solve_windows - prev_win) : p.solve_windows;
            prev_eps = rc_eps; prev_win = p.solve_windows; prev_t = now;
            const bool child_quiet = mgpu::IsChild() && !mgpu::ChildConsoleStatsEnabled();
            if (!child_quiet)
            LOGI("[stats] uptime=" << FmtDuration(static_cast<uint64_t>(up_s))
                 << " height=" << g_pool_block_height.load(std::memory_order_relaxed)
                 << " accepted=" << stats.accepted.load()
                 << " rejected=" << stats.rejected.load()
                 // ENC_RC throughput. rc-active=0 means the RC height is not latched and
                 // nothing is being mined -- check that before reading ep/s as a stall.
                 << " rc-active=" << (rc_active_now ? 1 : 0)
                 << " episodes=" << rc_eps
                 << " ep/s=" << std::fixed << std::setprecision(3) << (dt > 0 ? d_eps / dt : 0.0)
                 << " windows=" << d_win);
            // Multi-GPU child (solo mode): publish for the supervisor's [stats-all].
            // No exposed-digest-wait counter or rolling averages on this path.
            if (mgpu::IsChild()) {
                mgpu::ChildSnap snap;
                snap.ep_s = dt > 0 ? d_eps / dt : 0.0;
                snap.acc = stats.accepted.load();
                snap.rej = stats.rejected.load();
                mgpu::WriteChildSnapshot(snap);
            }
        }
    });

    UniValue rules(UniValue::VARR);
    rules.push_back("segwit");

    // Per-rig coinbase extranonce so a FLEET sharing one payout address (solo-via-proxy)
    // never grinds duplicate work: worker name (operator-legible - shows which rig won a
    // block) + 4 random bytes (uniqueness even if two rigs share a --worker name). Fixed
    // for the process; distinct rigs -> distinct coinbase -> distinct merkle root.
    std::vector<unsigned char> coinbase_extranonce;
    {
        for (size_t i = 0; i < cfg.worker.size() && i < 16; ++i)
            coinbase_extranonce.push_back(static_cast<unsigned char>(cfg.worker[i]));
        std::mt19937 enrng(static_cast<uint32_t>(MonoMs()) ^ static_cast<uint32_t>(::getpid()));
        for (int i = 0; i < 4; ++i)
            coinbase_extranonce.push_back(static_cast<unsigned char>(enrng() & 0xff));
        std::ostringstream hx;
        for (unsigned char b : coinbase_extranonce) hx << std::hex << std::setw(2) << std::setfill('0') << (int)b;
        LOGI("[init] coinbase extranonce worker=" << cfg.worker << " bytes=" << coinbase_extranonce.size()
             << " hex=" << hx.str() << " (fleet work-partitioning; share one payout safely)");
    }

    bool prev_dev_window = false;   // for dev-fee window entry/exit logging
    int64_t solo_fail_since_ms = 0; // solo->pool failover: when the GBT source first went unreachable (0 = healthy)

    while (true) {
        // idle-gate: if the box is busy, pause here (no GBT fetch, no solve) until it's idle.
        // The poller already logs the transition; we just wait, leaving the GPU free.
        if (!GateAllowsMining()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            continue;
        }

        std::atomic<bool> abort_flag{false};

        // ---- dev-fee window (time-based): is THIS template a dev cycle? ----
        // The phase MUST come from wall-clock time, not process uptime. Uptime restarts
        // at 0, which puts EVERY (re)start inside the leading dev slice; a restart-prone
        // rig then pays the dev address far more than the advertised rate, and a solve
        // that begins in the window keeps its dev coinbase for its whole run. Measured
        // 2026-08-12 on the 5090 (47 restarts overnight): 4 of 5 solo blocks paid dev
        // against a 1% fee. Epoch-phased, the window is the same slice of every
        // wall-clock hour no matter how often the process restarts.
        const double phase_sec = static_cast<double>(std::time(nullptr));
        const double pos = phase_sec - kDevPeriodSec *
            static_cast<double>(static_cast<uint64_t>(phase_sec / kDevPeriodSec));
        const bool dev_window = InDevFeeWindow(phase_sec, kDevPeriodSec, cfg.devfee);
        const CScript& active_script = dev_window ? dev_script : payout_script;
        if (dev_window != prev_dev_window) {
            if (dev_window)
                LOGI("[devfee] >>> entering dev-fee window (" << cfg.devfee << "%): coinbase pays dev addr "
                     << cfg.devaddress << " for ~" << static_cast<uint64_t>(dev_window_sec - pos) << "s more");
            else
                LOGI("[devfee] <<< dev-fee window ended; coinbase back to your payout address");
            prev_dev_window = dev_window;
        }

        // ---- getblocktemplate (segwit rules required, mining.cpp) ----
        UniValue tr(UniValue::VOBJ);
        tr.pushKV("rules", rules);
        UniValue gbt_params(UniValue::VARR);
        gbt_params.push_back(tr);

        Timer gbt_sp;
        UniValue gbt;
        try {
            gbt = rpc.Call("getblocktemplate", gbt_params);
            solo_fail_since_ms = 0;   // healthy again
        } catch (const std::exception& e) {
            // solo->pool failover: if a fallback pool is configured and the GBT source has
            // been unreachable for >= fallback_after_s, re-exec into pool mode (the recovery
            // prober there returns us to solo when the coordinator comes back).
            if (!cfg.fallback_pool.empty()) {
                const int64_t now_ms = MonoMs();
                if (solo_fail_since_ms == 0) solo_fail_since_ms = now_ms;
                const int64_t down_s = (now_ms - solo_fail_since_ms) / 1000;
                if (down_s >= cfg.fallback_after_s) {
                    LOGW("[fallback] solo GBT source unreachable for " << down_s << "s (>="
                         << cfg.fallback_after_s << "); failing over to pool " << cfg.fallback_pool
                         << " (will return to solo when it recovers)");
                    setenv("MATADOR_FALLBACK_STATE", "pool", 1);
                    const std::string exe = SelfExePath();
                    if (!exe.empty()) (logtee::ExecRestore(), execv(exe.c_str(), argv));
                    LOGE("[fallback] re-exec to pool failed (" << std::strerror(errno) << "); staying solo");
                } else {
                    LOGW("[gbt] error: " << e.what() << " (retry in 1s; failover in "
                         << (cfg.fallback_after_s - down_s) << "s)");
                }
            } else {
                LOGW("[gbt] error: " << e.what() << " (retry in 1s)");
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        // chain_guard: only present on btx's challenge variants; standard GBT may
        // omit it. Guarded by exists(). If present and paused, back off.
        if (gbt.exists("chain_guard") && gbt["chain_guard"].exists("should_pause_mining") &&
            gbt["chain_guard"]["should_pause_mining"].get_bool()) {
            const std::string reason = gbt["chain_guard"].exists("reason")
                                           ? gbt["chain_guard"]["reason"].get_str() : "(none)";
            LOGW("[guard] paused: " << reason << " (back off 2s)");
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        const std::string longpollid = gbt.exists("longpollid") ? gbt["longpollid"].get_str() : "";

        TemplateInputs meta;
        CBlock block;
        try {
            block = AssembleBlock(gbt, active_script, coinbase_extranonce, meta);
        } catch (const std::exception& e) {
            LOGE("[gbt] assemble failed: " << e.what() << " (retry in 1s)");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        // Height of the job we are actually mining. Despite the g_pool_ name this is the
        // shared "current job height" the stats line and the status API both read; solo
        // never set it, so RC-activity detection and /summary's block_height both read 0.
        g_pool_block_height.store(static_cast<int32_t>(meta.height), std::memory_order_relaxed);

        LOGI("[gbt] template height=" << meta.height
             << " prevhash=" << Short(meta.prev_hash)
             << " nTx=" << meta.ntx
             << " coinbasevalue=" << meta.coinbase_value
             << " bits=" << std::hex << meta.nbits << std::dec
             << " parent_mtp=" << meta.parent_mtp
             // matmul_n is the node's own header dim AND the solo RC activation signal;
             // 0 means the coordinator sent none and we fall back (see solo_mining.h).
             << " matmul_n=" << meta.matmul_n
             << " rc-active=" << (consensus.IsMatMulRCActive(static_cast<int32_t>(meta.height)) ? 1 : 0)
             << " fetch_ms=" << gbt_sp.ms());

        // ---- tip-watcher: longpoll GBT; abort solve on tip change ----
        std::thread watcher([&]() {
            const bool gate_abort = (ToLowerCopy(cfg.gate_yield) != "finish");
            while (!abort_flag.load() && !stop_all.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                // idle-gate: the box just got busy -> abort this solve to free the GPU fast.
                if (gate_abort && !GateAllowsMining()) {
                    LOGW("[gate] yield: aborting in-flight solve to release the GPU");
                    abort_flag.store(true);
                    return;
                }
                Timer lp_sp;
                try {
                    UniValue lp_tr(UniValue::VOBJ);
                    lp_tr.pushKV("rules", rules);
                    if (!longpollid.empty()) lp_tr.pushKV("longpollid", longpollid);
                    UniValue lp_params(UniValue::VARR);
                    lp_params.push_back(lp_tr);
                    UniValue fresh = rpc.Call("getblocktemplate", lp_params, /*longpoll=*/true);
                    solo_fail_since_ms = 0;   // coordinator reachable again
                    const std::string newprev = fresh.exists("previousblockhash")
                                                    ? fresh["previousblockhash"].get_str() : "";
                    if (!newprev.empty() && newprev != meta.prev_hash.GetHex()) {
                        LOGI("[tip] changed old=" << Short(meta.prev_hash)
                             << " new=" << Short(newprev) << "; aborting solve");
                        abort_flag.store(true);
                        return;
                    }
                } catch (...) {
                    // Distinguish "node alive, just a slow block" from "coordinator down".
                    // A longpoll that ran a long time before failing means btxd ACCEPTED and
                    // HELD the connection waiting for a new block (a normal slow-block timeout,
                    // ~the full 120s) - NOT an outage. A real outage fails FAST (connect refused
                    // or the 10s connect-timeout). Only a fast failure counts toward failover;
                    // without this guard a fleet worker would wrongly fail over on slow blocks.
                    const bool node_was_reachable = lp_sp.ms() >= 15000.0;
                    if (node_was_reachable) {
                        solo_fail_since_ms = 0;   // alive; longpoll just idled out
                    } else if (!cfg.fallback_pool.empty()) {
                        // The GBT source is genuinely unreachable MID-SOLVE. The main-loop
                        // getblocktemplate (which also owns failover) is not re-reached until
                        // this long solve ends, so drive the solo->pool failover here too.
                        const int64_t now_ms = MonoMs();
                        if (solo_fail_since_ms == 0) solo_fail_since_ms = now_ms;
                        if ((now_ms - solo_fail_since_ms) / 1000 >= cfg.fallback_after_s) {
                            LOGW("[fallback] coordinator unreachable mid-solve for >="
                                 << cfg.fallback_after_s << "s; aborting solve and failing over to pool "
                                 << cfg.fallback_pool);
                            abort_flag.store(true);
                            setenv("MATADOR_FALLBACK_STATE", "pool", 1);
                            const std::string exe = SelfExePath();
                            if (!exe.empty()) (logtee::ExecRestore(), execv(exe.c_str(), argv));
                            LOGE("[fallback] re-exec to pool failed (" << std::strerror(errno) << ")");
                        }
                    }
                    /* else: keep polling */
                }
            }
        });

        SolveAndSubmit(rpc, consensus, block, meta, abort_flag, cfg.maxtries, cfg.solver_threads, stats);

        abort_flag.store(true);
        if (watcher.joinable()) watcher.join();
    }

    // unreachable (loop is infinite); kept for completeness.
    stop_all.store(true);
    if (update_thread.joinable()) update_thread.join();
    if (gate_thread.joinable()) gate_thread.join();
    if (api_thread.joinable()) api_thread.join();
    if (heartbeat.joinable()) heartbeat.join();
    curl_global_cleanup();
    return 0;
}
