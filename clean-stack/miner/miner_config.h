// matador-miner: the runtime Config struct (+ PoolEndpoint and the dev-fee payout
// constant). Defined before the pool client because RunPoolLoop takes a const
// Config&. Filled by config_parse.h (file/CLI), consumed across the miner.
// SECTION of the single miner translation unit. Extracted verbatim.
#pragma once

// 5a. Runtime config (defined here, before the pool client, because RunPoolLoop
//     below takes a const Config&; the solo arg/env parsing in section 6 fills it).
// ===========================================================================

// Baked-in dev-fee payout address (AM2 LLC). P2MR (witness v2 / 32-byte program).
// Used when --dev-fee > 0 and no --dev-address override is given. Defined here
// (before RunPoolLoop + main) so both the pool and solo dev-fee paths can use it.
static constexpr const char* kDevAddress =
    "btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv";

struct PoolEndpoint {
    std::string host;
    int port{0};
    std::string label;
    bool use_tls{false};   // ssl:// / tls:// / stratum+ssl:// / stratum+tls:// scheme (see ParsePoolEndpoint)
};

struct Config {
    std::string config_path;             // optional JSON config file (env MATADOR_CONFIG / --config)
    std::string log_file_path;           // optional: also mirror all stderr output to this file (env MATADOR_LOG_FILE / --log-file)

    // ---- mode (solo default, unchanged behavior; pool = stratum client) ----
    std::string mode{"solo"};             // solo|pool (env MODE)
    std::string pool_host;                // stratum host (pool mode; from --pool host:port)
    int pool_port{0};                     // stratum port (pool mode)
    std::vector<PoolEndpoint> pools;      // ordered failover list (first entry = primary)
    std::string worker{"matador"};        // stratum worker suffix -> "<addr>.<worker>" (env WORKER)
    std::string operator_label;           // pool dashboard label (env OPERATOR_LABEL / --operator-label);
                                          // empty -> defaults to worker. Multi-GPU fan-out auto-shares the
                                          // base label across children so the rig groups as ONE "hive".
    std::string pool_pass{"x"};           // stratum password (env POOL_PASS)
    // TLS pool connections (ssl://, tls:// scheme -- see PoolEndpoint::use_tls) verify the
    // server certificate against the system CA store by default. This drops to no verification
    // (still encrypted, just no identity/MITM check) for a pool on a self-signed cert.
    bool pool_tls_insecure{false};         // env MATADOR_POOL_TLS_INSECURE / --pool-tls-insecure

    // ---- optional SOCKS5 proxy for the pool connection (RFC 1928/1929) ----
    // Routes the stratum socket through a SOCKS5 proxy so a pool/node reachable
    // ONLY via a proxy works without a local route: e.g. Tailscale userspace-
    // networking (the 100.64/10 CGNAT range, no kernel tun) or `ssh -D`. The proxy
    // resolves the pool host, so MagicDNS/tailnet names resolve proxy-side. Empty
    // host = direct connect (default, unchanged behavior).
    std::string socks5_host;              // proxy host (env SOCKS5 / ALL_PROXY=socks5://host:port)
    int socks5_port{0};                   // proxy port
    std::string socks5_user;              // optional RFC 1929 username (env SOCKS5_USER)
    std::string socks5_pass;              // optional RFC 1929 password (env SOCKS5_PASS)

    std::string rpcconnect{"127.0.0.1"};
    std::string chain{"main"};            // main|test|regtest (drives SelectParams + default port)
    int rpcport{19334};                   // mainnet default (chainparamsbase.cpp:82)
    bool rpcport_explicit{false};         // true once --rpcport / RPCPORT seen
    std::string datadir;                  // for default cookie path
    std::string rpccookiefile;            // explicit cookie path (overrides datadir/.cookie)
    std::string rpcuser;
    std::string rpcpassword;
    std::string payoutaddress;            // required, btx1 P2MR
    uint64_t maxtries{1ULL << 40};        // effectively "until tip changes"
    int rc_height{0};                     // ENC_RC activation height PIN: 0 = auto (latch from the
                                          // pool job; see MaybeLatchRCFromJob). A non-zero value
                                          // overrides any pool-announced activation, forever.
                                          // --rc-height / BTX_MATMUL_RC_HEIGHT / rc_height
    std::string attest_key_file;          // SOLO self-attest: path to a 64-hex trusted-mirror signing
                                          // key file. Empty (the default) = feature OFF. When set and
                                          // the node runs matmulvalidation=trusted with our pubkey,
                                          // every solved block is signed + submitted via
                                          // submitmatmulattestations BEFORE submitblock, so it
                                          // connects instantly instead of parking in trusted-wait.
    std::string attest_context;           // SOLO self-attest: 64-hex replay_authority_context override.
                                          // Empty = fetch from the node's getmatmultrustedstatus.
    int devfee{1};                        // dev-fee percent 0..100; ~devfee of every 100 cycles -> dev addr
    std::string devaddress;               // dev-fee payout (defaults to kDevAddress when empty)
    std::string backend;                  // BTX_MATMUL_BACKEND override: cuda|cpu (empty = auto/env)
    std::vector<std::string> gpu_devices; // basic multi-GPU fan-out device ids (e.g. [0,1]); no solver-level optimization
    bool gpu_worker_suffix{false};        // multi-GPU: append a per-card "-gpuN" suffix to the stratum worker.
                                          // DEFAULT OFF since v0.8.46: one rig = ONE pool worker name, matching
                                          // what every single-process miner shows (a 5-card rig used to list as
                                          // 5 workers on pools without operator-label grouping, e.g. ninjaraider).
                                          // Cosmetic either way: each card keeps its own pool connection + nonce
                                          // lane + share credit; the pool just aggregates under <addr>.<worker>.
                                          // --gpu-suffix / gpu_worker_suffix:true restores per-card worker rows.
    bool update_check{true};              // check GitHub releases at startup (--no-update-check skips it)
    bool auto_update{true};               // auto download+verify+swap+re-exec to the latest (--no-auto-update = notify only)
    int update_interval_s{1800};          // periodic re-check cadence after startup (default 30min); <=0 = startup-only
    int update_jitter_s{300};             // randomized 0..N s delay before each PERIODIC check (de-syncs a fleet; 0=off)
    int min_version_age_s{3600};          // bake-time: a release must be >=N s old before auto-adoption (0=off)
    std::string update_channel{"stable"}; // "stable" = /releases/latest (non-prerelease); "prerelease" = newest incl. prereleases
    bool update_check_only{false};        // run ONE update check (may swap+re-exec) then exit; the systemd .timer entrypoint
    bool poolcore{false};                 // wrapper-driven compute-core mode (JSON-lines on stdio); no stratum/updater/fee
    std::string fallback_pool;            // solo->pool failover: pool endpoint(s) to mine when the solo coordinator/proxy is down (empty=disabled)
    int fallback_after_s{60};             // solo RPC must fail continuously this long before failing over to the pool
    int solo_recheck_s{120};              // while on the fallback pool, probe the solo coordinator this often to return to solo
    std::string should_mine_command;      // idle-gate: run this every should_mine_interval; exit 0=mine, non-zero=yield (empty=always mine)
    int should_mine_interval{2};          // idle-gate poll cadence (s)
    std::string gate_yield{"abort"};      // on yield: "abort" (kill in-flight solve, fast GPU release) or "finish" (let it complete)
    bool api_enabled{true};               // local read-only HTTP status API (default ON, loopback:4060; --no-api disables). hub-ready out of the box
    std::string api_listen{"127.0.0.1"};  // bind address for status API
    int api_port{0};                      // 0 disables; common rigs use e.g. 4060
    bool watchdog_enabled{true};          // observe + safe reconnect/failover actions in pool mode
    int watchdog_check_s{15};             // watchdog poll interval
    int watchdog_reject_streak{20};       // reconnect/failover after this many consecutive rejects (0 disables)
    int watchdog_no_share_s{0};           // optional: reconnect if hashing but no accepted share for N seconds (0 disables)
    bool thermal_enabled{true};           // warning-only GPU temp/power watchdog; no tuning/restart side effects
    int thermal_warn_temp_c{86};          // warn when any GPU reaches this temp (0 disables)
    int thermal_critical_temp_c{90};      // louder warning level only; still observe-only (0 disables)
    double thermal_warn_power_w{0.0};     // optional warn when any GPU reaches this power draw (0 disables)
    unsigned solver_threads{1};           // concurrent in-flight GPU solves (disjoint nonce ranges). 1 =
                                          // single stream (CUDA saturates a 5090 with one). Metal needs
                                          // ~16-32: one stream leaves the GPU ~85% idle on submit/sync
                                          // latency; concurrency recovers ~5.5-6.5x (M4 Max bench). Each
                                          // thread solves a full independent candidate; first valid wins.
    int clk_offset{0};                    // NVIDIA GPC clock V/F offset (MHz): --clk-offset /
                                          // BTX_GPU_CLK_OFFSET / config "clk_offset". 0 = leave clocks
                                          // stock. Applied via NVML at startup (needs root). On a
                                          // power-capped card a positive offset shifts the V/F curve up
                                          // so the same watts buy more clock -> more nps. Clamped to the
                                          // driver's range; reversible (cleared on reboot).
    int power_limit_w{0};                 // board power limit in W: --power-limit / BTX_GPU_POWER_LIMIT /
                                          // config "power_limit". 0 = leave as-is. Applied via NVML at
                                          // startup (needs root); clamped to the card's range. At a power
                                          // cap, watts == nonces, so this guarantees the full budget.
    int lock_gpu_clock{0};                // pin core clock (MHz): --lock-gpu-clock / BTX_GPU_LOCK_CLOCK /
                                          // config "lock_gpu_clock". 0 = unlocked. For stability/consistency
                                          // or to cap the boost so a too-high offset can't wedge the GPU.
    int fan_pct{0};                       // pin GPU fan duty (1..100%): --fan-pct / BTX_GPU_FAN_PCT /
                                          // config "fan_pct". 0 = leave on the driver's auto curve. Applied
                                          // via NVML at startup (needs root). Holding fans high keeps the
                                          // thermal margin so the boost can't step down -> flatter nps in a
                                          // warm room. All tuning (offset/power/lock/fan) is reverted to
                                          // stock on shutdown (SIGTERM/SIGINT or clean exit).
    int mem_clk_offset{0};                // NVIDIA memory (GDDR7) clock V/F offset (MHz): --mem-clk-offset /
                                          // BTX_GPU_MEM_CLK_OFFSET / config "mem_clk_offset". 0 = leave mem
                                          // stock. Applied via NVML (nvmlDeviceSetMemClkVfOffset) at startup
                                          // (needs root). On a power-capped card the digest is HBM-traffic-
                                          // bound, so faster memory delivers more effective bandwidth at the
                                          // same board watts -> more nps. Clamped to the driver's range;
                                          // reverted on shutdown. CAUTION: a too-high mem offset corrupts
                                          // digests SILENTLY (no crash) -> pool rejects; raise small + watch rej.
    int lock_mem_clock{0};                // pin memory clock (MHz): --lock-mem-clock / BTX_GPU_LOCK_MEM_CLOCK /
                                          // config "lock_mem_clock". 0 = unlocked. The INVERSE lever of
                                          // mem_clk_offset: pinning GDDR to a LOW supported point frees board
                                          // power at the cap -> the core boosts higher. The driver snaps to
                                          // supported points (5090: 14551/7001/810/405). MEASURED DEAD on the
                                          // 5090 at ndiff~6.5 (2026-07-08 live A/B): 810 = -64% (digest goes
                                          // memory-starved, board drops to 267W), 7001 = -1..-2% (core +58MHz
                                          // doesn't cover the digest's bandwidth loss). Kept as a per-card
                                          // offline-tuning knob; guard = batch fill + rej. Overrides
                                          // mem_clk_offset. NVML SetMemoryLockedClocks; reverted on shutdown.
};
