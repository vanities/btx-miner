// matador-miner: SOLO mode. JSON-RPC getblocktemplate client, the P2MR payout
// predicate, coinbase + full-CBlock assembly, and the solo solve+submit path.
// Also declares the shared external-HIP sidecar structs (used by solo and pool;
// the bridge impl is in external_hip.h). CONSENSUS-CRITICAL block construction.
// SECTION of the single miner translation unit (uses mlog, the vendored btx
// consensus types, and SolveMatMul). Extracted verbatim from matador-miner.cpp.
#pragma once

#include <chainparams.h>   // Params().GenesisBlock() (self-attest chain_id)
#include <hash.h>          // HashWriter (self-attest statement hash)
#include <key.h>           // CKey / ECC_Context (self-attest signing)
#include <matmul/matmul_v4_rc.h>  // RCConsensusHeaderMatmulDim (RC header dim + solo latch signal)
#include <pubkey.h>        // CPubKey

#include <atomic>          // one-shot warn latch on the v3-dim fallback
#include <cstdlib>         // getenv (BTX_RC_STRATUM_AUTOLATCH)
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>          // INT32_MAX sentinel on RCActivationHeight()
#include <mutex>
#include <random>

// ===========================================================================
// 1. Minimal JSON-RPC over HTTP to localhost btxd via libcurl.
//    Body: {"jsonrpc":"1.0","id":"matador-miner","method":..,"params":..}
//    Auth: HTTP Basic from cookie ("__cookie__:<hex>") or rpcuser:rpcpassword.
// ===========================================================================
class RpcClient {
public:
    // auth_userpass is the literal "user:password" (cookie line counts as one).
    RpcClient(std::string url, std::string auth_userpass)
        : m_url(std::move(url)), m_auth(std::move(auth_userpass))
    {
        m_auth_b64 = EncodeBase64(m_auth);
    }

    // Make LONGPOLL calls abortable. A getblocktemplate longpoll holds the HTTP
    // request open up to the full curl timeout waiting for a new block, and the
    // solo loop JOINS its tip-watcher thread right after every solve -- so before
    // this hook, a rejected submitblock (or any fast solve on a slow chain) left
    // the main thread stuck in that join for up to the whole timeout window.
    // SolveAndSubmit registers the shared abort flag here (the same atomic the
    // watcher loops on; it is function-scope in the solo main loop, so it outlives
    // every call); the curl progress callback then aborts the in-flight transfer
    // within ~1s of the flag flipping. ONLY longpoll calls consult it -- aborting
    // a submitblock mid-flight could lose a solved block, which is far worse than
    // a slow join. nullptr (the default) restores plain blocking behavior.
    void SetLongpollAbortFlag(std::atomic<bool>* flag) { m_longpoll_abort.store(flag); }

    // longpoll=true marks a getblocktemplate longpoll, where a timeout is EXPECTED (the
    // node held the request open waiting for a new block and none arrived). Such timeouts
    // are logged quietly, not as ERROR, and the caller can tell "node alive, slow block"
    // from "node down" by the elapsed time (a real outage fails fast at connect time).
    // A silently-respawned btxd rotates the RPC cookie; every later call then dies
    // http-401 until the process restarts (lost solve at h=186887, 2026-08-12: an
    // 8-min solve straddled a crash-respawn and submitted with dead auth). On 401,
    // re-resolve auth once via the refresher and retry the SAME call -- every call
    // path (GBT, submit, attest) heals without knowing.
    void SetAuthRefresher(std::function<std::string()> f) { m_auth_refresher = std::move(f); }

    UniValue Call(const std::string& method, const UniValue& params, bool longpoll = false)
    {
        try {
            return CallOnce(method, params, longpoll);
        } catch (const std::exception& e) {
            if (m_auth_refresher && std::string(e.what()).find("(http 401)") != std::string::npos) {
                std::string fresh;
                try { fresh = m_auth_refresher(); } catch (...) {}
                if (!fresh.empty()) {
                    {
                        std::lock_guard<std::mutex> lk(m_auth_mu);
                        if (fresh != m_auth) { m_auth = fresh; m_auth_b64 = EncodeBase64(m_auth); }
                    }
                    LOGW("[rpc] http 401 -> auth refreshed from cookie; retrying " << method);
                    return CallOnce(method, params, longpoll);
                }
            }
            throw;
        }
    }

    UniValue CallOnce(const std::string& method, const UniValue& params, bool longpoll = false)
    {
        Timer sp;
        UniValue req(UniValue::VOBJ);
        req.pushKV("jsonrpc", "1.0");
        req.pushKV("id", "matador-miner");
        req.pushKV("method", method);
        req.pushKV("params", params);
        const std::string body = req.write();

        std::string response;
        long http_code = 0;

        CURL* curl = curl_easy_init();
        if (curl == nullptr) {
            LOGE("[rpc] method=" << method << " curl_easy_init failed");
            throw std::runtime_error("curl_easy_init failed");
        }

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        std::string auth_b64_snapshot;
        {
            std::lock_guard<std::mutex> lk(m_auth_mu);   // watcher thread longpolls concurrently
            auth_b64_snapshot = m_auth_b64;
        }
        const std::string auth_header = "Authorization: Basic " + auth_b64_snapshot;
        headers = curl_slist_append(headers, auth_header.c_str());

        curl_easy_setopt(curl, CURLOPT_URL, m_url.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &RpcClient::WriteCb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        // Longpoll: 30s (down from 120s) as belt-and-braces on the join stall, plus
        // the cooperative abort below as the real fix. 30s still comfortably clears
        // the ">=15s means the node was reachable" failover discriminator at the
        // watcher call site, so slow-block timeouts keep reading as "node alive".
        // Normal calls (submitblock!) keep the full 120s and are never aborted.
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, longpoll ? 30L : 120L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        std::atomic<bool>* lp_abort = longpoll ? m_longpoll_abort.load() : nullptr;
        if (lp_abort != nullptr) {
            // Progress callback fires ~1/s even while the transfer is idle waiting
            // on the held-open longpoll; returning nonzero aborts the transfer with
            // CURLE_ABORTED_BY_CALLBACK (handled quietly below).
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION,
                +[](void* u, curl_off_t, curl_off_t, curl_off_t, curl_off_t) -> int {
                    return static_cast<std::atomic<bool>*>(u)->load() ? 1 : 0;
                });
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, lp_abort);
        }

        const CURLcode rc = curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        if (rc != CURLE_OK) {
            // NOTE: do not log m_auth/m_auth_b64 (contains the cookie).
            if (longpoll && rc == CURLE_OPERATION_TIMEDOUT) {
                // Expected: the longpoll held open waiting for a new block, none came within
                // the timeout. The node is alive (the connection was established + held), so
                // this is not an error - just a quiet heartbeat.
                LOGD("[rpc] longpoll getblocktemplate idle " << sp.ms()
                     << "ms (no new block yet; node alive)");
            } else if (longpoll && rc == CURLE_ABORTED_BY_CALLBACK) {
                // Expected: the solve ended (solved/exhausted/aborted) and flipped the
                // shared abort flag, so the progress callback cancelled the longpoll to
                // let the watcher join immediately. Not an outage signal.
                LOGD("[rpc] longpoll aborted after " << sp.ms()
                     << "ms (solve ended; watcher unblocking)");
            } else {
                LOGE("[rpc] method=" << method << " transport_error=" << curl_easy_strerror(rc)
                     << " url=" << m_url << " in " << sp.ms() << "ms");
            }
            throw std::runtime_error(std::string("rpc transport: ") + curl_easy_strerror(rc));
        }

        UniValue root;
        if (!root.read(response)) {
            LOGE("[rpc] method=" << method << " http=" << http_code
                 << " unparseable response len=" << response.size() << " in " << sp.ms() << "ms");
            throw std::runtime_error("rpc: unparseable response (http " + std::to_string(http_code) + ")");
        }

        const UniValue& err = root["error"];
        if (!err.isNull()) {
            std::string emsg = err.isObject() && err.exists("message")
                                   ? err["message"].get_str() : err.write();
            // Embed the numeric code so callers can classify (e.g. -28 = RPC warmup,
            // transient by definition: "Loading wallet...", "Verifying blocks...").
            std::string ecode;
            if (err.isObject() && err.exists("code") && err["code"].isNum())
                ecode = " [code " + std::to_string(err["code"].getInt<int>()) + "]";
            LOGW("[rpc] method=" << method << " http=" << http_code
                 << " rpc_error=\"" << emsg << "\"" << ecode << " in " << sp.ms() << "ms");
            throw std::runtime_error("rpc error (" + method + "): " + emsg + ecode);
        }

        LOGD("[rpc] method=" << method << " http=" << http_code
             << " resp_len=" << response.size() << " in " << sp.ms() << "ms");
        return root["result"];
    }

private:
    static size_t WriteCb(char* ptr, size_t size, size_t nmemb, void* userdata)
    {
        auto* out = static_cast<std::string*>(userdata);
        out->append(ptr, size * nmemb);
        return size * nmemb;
    }
    std::string m_url;
    std::mutex m_auth_mu;     // guards m_auth/m_auth_b64 (401-refresh vs concurrent longpoll)
    std::function<std::string()> m_auth_refresher;   // re-resolves auth (cookie re-read) on 401
    std::string m_auth;       // user:password (never logged)
    std::string m_auth_b64;   // base64(user:password) (never logged)
    // Longpoll abort hook (see SetLongpollAbortFlag). Atomic pointer: the WATCHER
    // thread longpolls on this client while the solve thread (re)registers the flag.
    std::atomic<std::atomic<bool>*> m_longpoll_abort{nullptr};
};

// ===========================================================================
// 2. P2MR predicate, replicated inline.
//    btx's IsP2MROutputScript (node/miner.cpp:53) is NOT exported in any header,
//    so we re-implement the exact predicate here: witness v2 + 32-byte program.
//    An OP_RETURN coinbase payout is also consensus-legal (PassesReducedData...)
//    but pointless for a solo miner (burns the reward), so we require P2MR.
// ===========================================================================
static bool IsP2MROutputScriptLocal(const CScript& spk)
{
    int witness_version{-1};
    std::vector<unsigned char> witness_program;
    if (!spk.IsWitnessProgram(witness_version, witness_program)) return false;
    return witness_version == 2 && witness_program.size() == 32;
}

// ===========================================================================
// 3b. Chain selection helpers.
//    Maps a --chain string to btx's ChainType and the matching default RPC port.
//    Ports verified against btx chainparamsbase.cpp:80-92 (215170f2 / v0.32.11):
//      main 19334, test(net3) 29334, regtest 18443. SelectParams() then drives
//      the GLOBAL chainparams so the right consensus activation heights and the
//      right bech32 address format (mainnet btx1.., regtest btxrt1..) are used.
// ===========================================================================
static bool ParseChainType(const std::string& s, ChainType& out)
{
    if (s == "main" || s == "mainnet")               { out = ChainType::MAIN;    return true; }
    if (s == "test" || s == "testnet" || s == "testnet3") { out = ChainType::TESTNET; return true; }
    if (s == "regtest")                              { out = ChainType::REGTEST; return true; }
    return false;
}

static const char* ChainTypeName(ChainType c)
{
    switch (c) {
        case ChainType::MAIN:    return "main";
        case ChainType::TESTNET: return "test";
        case ChainType::REGTEST: return "regtest";
        default:                 return "other";
    }
}

// Default RPC port per chain (chainparamsbase.cpp). Used only when the operator
// does not pass --rpcport / RPCPORT explicitly.
static int DefaultRpcPortForChain(ChainType c)
{
    switch (c) {
        case ChainType::MAIN:    return 19334;
        case ChainType::TESTNET: return 29334;
        case ChainType::REGTEST: return 18443;
        default:                 return 19334;
    }
}

// ===========================================================================
// 3. Build the coinbase transaction from GBT fields.
//    Mirrors BlockAssembler::CreateNewBlock (node/miner.cpp:982-991) +
//    GenerateCoinbaseCommitment's witness reserved value (validation.cpp:9728).
// ===========================================================================
static CMutableTransaction BuildCoinbase(int32_t height,
                                         int64_t coinbase_value,
                                         const CScript& payout_script,
                                         const std::vector<unsigned char>& default_witness_commitment,
                                         const std::vector<unsigned char>& coinbase_extranonce)
{
    CMutableTransaction cb;
    // CMutableTransaction default version == CTransaction::CURRENT_VERSION (==2);
    // CreateNewBlock leaves this default. Verified primitives/transaction.h:318.
    // (The field is named `version` on CMutableTransaction, not `nVersion`.)
    cb.version = 2;
    cb.vin.resize(1);
    cb.vin[0].prevout.SetNull();
    // BIP34 height (miner.cpp:988), then a Stratum-style EXTRANONCE. The extranonce is
    // what lets a fleet of rigs share ONE payout address without grinding duplicate
    // work: a distinct extranonce -> distinct coinbase -> distinct merkle root ->
    // disjoint PoW search space (exactly how a pool partitions work across workers).
    // Empty extranonce keeps the legacy single-miner coinbase (height + OP_0).
    cb.vin[0].scriptSig = coinbase_extranonce.empty()
        ? (CScript() << height << OP_0)
        : (CScript() << height << coinbase_extranonce);
    // BIP141 coinbase witness reserved value: a single 32-byte zero vector,
    // exactly what GenerateCoinbaseCommitment sets (validation.cpp:9728-9729).
    cb.vin[0].scriptWitness.stack.assign(1, std::vector<unsigned char>(32, 0x00));

    cb.vout.resize(1);
    cb.vout[0].scriptPubKey = payout_script;             // MUST be P2MR (v2/32B)
    cb.vout[0].nValue = coinbase_value;                  // GBT coinbasevalue (subsidy+fees)

    // Witness commitment output (BIP141). GBT's default_witness_commitment is the
    // FULL scriptPubKey (OP_RETURN + 0x24 + 0xaa21a9ed + 32-byte commitment) over
    // the UNMODIFIED template tx set. We keep the exact GBT tx set and the standard
    // zero witness reserved value, so the commitment is unchanged -> append verbatim.
    if (!default_witness_commitment.empty()) {
        CTxOut commit;
        commit.nValue = 0;
        commit.scriptPubKey = CScript(default_witness_commitment.begin(),
                                      default_witness_commitment.end());
        cb.vout.push_back(commit);
    }
    return cb;
}

// ===========================================================================
// 4. Assemble the full CBlock from a getblocktemplate result.
// ===========================================================================
struct TemplateInputs {
    int32_t height{0};
    int64_t parent_mtp{0};          // GBT time_policy.tip_mediantime (mining.cpp:341)
    uint256 prev_hash;
    uint32_t version{0};
    uint32_t curtime{0};
    uint32_t nbits{0};
    int64_t coinbase_value{0};
    size_t ntx{0};
    uint16_t matmul_n{0};           // GBT top-level "matmul_n" = the node's own header
                                    // matmul_dim (mining.cpp:9608). 0 = field absent.
};

// ---- RC auto-latch on the SOLO path -------------------------------------------------------
// The stratum path latches ENC_RC activation from the pool job (MaybeLatchRCFromJob) and
// poolcore routes off the job profile, but SOLO had NO equivalent: JobObserver only fires from
// the stratum reader, so a solo rig without --rc-height / BTX_MATMUL_RC_HEIGHT left
// RCActivationHeight() at INT32_MAX. IsMatMulRCActive() then returned false at every height and
// the dim fallback below stamped the v3 nMatMulDimension (512) while ActiveMatMulSolveVariant
// picked the BASE v3 solver -- every solve unacceptable, with nothing in the log to say why.
// This is the same failure the pool path hit with 1058 rejected shares, and it is the reported
// "matador was using 512" sighting.
//
// The signal is GBT's own "matmul_n", which the node sets from the header it would mine
// (consensus REQUIRES nMatMulV4Dimension there and rejects anything else). Unlike a pool this is
// OUR OWN trusted node, so there is no false-positive story to debounce against: latch on first
// sight. No height is compiled in, and an operator pin always wins (PinRCActivationHeight makes
// LatchRCActivationHeight a no-op). Opt out with BTX_RC_STRATUM_AUTOLATCH=0, same switch as the
// stratum path.
static void MaybeLatchRCFromTemplate(int32_t height, uint16_t matmul_n)
{
    if (height <= 0 || matmul_n == 0) return;
    if (const char* v = std::getenv("BTX_RC_STRATUM_AUTOLATCH");
        v != nullptr && v[0] == '0' && v[1] == '\0') return;
    if (matmul_n != matmul::v4::rc::RCConsensusHeaderMatmulDim()) return;
    if (Consensus::Params::RCActivationHeight() != std::numeric_limits<int32_t>::max()) return;

    if (Consensus::Params::LatchRCActivationHeight(height)) {
        LOGW("[solo] ENC_RC ACTIVATION detected from our node's block template at height "
             << height << " (matmul_n=" << matmul_n << " == the RC consensus dimension)"
             << " -- switching to the RC episode solver from this template on."
             << " Pin BTX_MATMUL_RC_HEIGHT (or rc_height in config) to override;"
             << " BTX_RC_STRATUM_AUTOLATCH=0 disables this detection.");
    }
}

static CBlock AssembleBlock(const UniValue& gbt,
                            const CScript& payout_script,
                            const std::vector<unsigned char>& coinbase_extranonce,
                            TemplateInputs& meta /*out*/)
{
    Timer sp;
    CBlock block;

    // ---- header scalars straight from GBT (mining.cpp:7895-7964) ----
    meta.height     = gbt["height"].getInt<int>();
    meta.version    = static_cast<uint32_t>(gbt["version"].getInt<int64_t>());
    meta.curtime    = static_cast<uint32_t>(gbt["curtime"].getInt<int64_t>());
    meta.prev_hash  = uint256::FromHex(gbt["previousblockhash"].get_str()).value();
    meta.nbits      = static_cast<uint32_t>(std::stoul(gbt["bits"].get_str(), nullptr, 16));
    meta.parent_mtp = gbt["time_policy"]["tip_mediantime"].getInt<int64_t>(); // v3 seed input
    meta.coinbase_value = gbt["coinbasevalue"].getInt<int64_t>();

    block.nVersion       = static_cast<int32_t>(meta.version);
    block.hashPrevBlock  = meta.prev_hash;
    block.nTime          = meta.curtime;
    block.nBits          = meta.nbits;
    block.nNonce64       = 0;
    block.nNonce         = 0;
    block.mix_hash.SetNull();
    block.matmul_digest.SetNull();
    // The node tells us the dim it would mine (GBT top-level "matmul_n", mining.cpp:9608 --
    // a documented backward-compatible field). Take it verbatim: it IS consensus for this
    // height, it feeds the seed-V3 and sigma preimages, and it is the RC activation signal.
    // Absent/0 leaves the RC-aware fallback in SolveAndSubmit to fill it.
    meta.matmul_n = gbt.exists("matmul_n")
                        ? static_cast<uint16_t>(gbt["matmul_n"].getInt<int64_t>())
                        : uint16_t{0};
    block.matmul_dim = meta.matmul_n;
    // Latch RC BEFORE the solve variant is chosen (ActiveMatMulSolveVariant runs inside
    // SolveMatMul), otherwise we would stamp the RC dim but grind the base v3 solver.
    MaybeLatchRCFromTemplate(meta.height, meta.matmul_n);
    // seed_a/seed_b are NOT taken from GBT. They are re-derived by SolveMatMul ->
    // SetDeterministicMatMulSeeds over the FINAL header (depends on our merkle root
    // and parent_mtp). GBT's seed_a/seed_b are only valid for the node's own coinbase.

    // ---- coinbase ----
    std::vector<unsigned char> dwc;
    if (gbt.exists("default_witness_commitment")) {
        dwc = ParseHex(gbt["default_witness_commitment"].get_str());
    }
    CMutableTransaction cb = BuildCoinbase(meta.height, meta.coinbase_value, payout_script, dwc, coinbase_extranonce);
    block.vtx.push_back(MakeTransactionRef(std::move(cb)));

    // ---- mempool transactions (verbatim, in GBT order; do NOT reorder/drop) ----
    if (gbt.exists("transactions")) {
        for (const UniValue& t : gbt["transactions"].getValues()) {
            CMutableTransaction tx;
            if (!DecodeHexTx(tx, t["data"].get_str())) {
                LOGE("[assemble] undecodable tx in GBT height=" << meta.height
                     << " txdata_len=" << t["data"].get_str().size());
                throw std::runtime_error("getblocktemplate: undecodable tx");
            }
            block.vtx.push_back(MakeTransactionRef(std::move(tx)));
        }
    }
    meta.ntx = block.vtx.size();

    // ---- merkle root (consensus/merkle.h) ----
    block.hashMerkleRoot = BlockMerkleRoot(block);

    LOGI("[assemble] height=" << meta.height
         << " nTx=" << meta.ntx
         << " coinbase_value=" << meta.coinbase_value
         << " merkle=" << Short(block.hashMerkleRoot)
         << " dwc=" << (dwc.empty() ? "none" : "appended")
         << " in " << sp.ms() << "ms");
    return block;
}

// ===========================================================================
// 5. Solve + submit. Mirrors GenerateBlock (rpc/mining.cpp:4696-4768).
// ===========================================================================
static int64_t MonoMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stats {
    std::atomic<uint64_t> accepted{0};
    std::atomic<uint64_t> rejected{0};
    std::atomic<uint64_t> stale{0};        // pool: shares found after a newer job arrived
    std::atomic<uint64_t> dev_shares{0};   // pool: shares submitted under the dev address
    std::atomic<uint64_t> total_nonces{0};
    std::atomic<uint64_t> reject_streak{0};
    std::atomic<int64_t> last_notify_ms{0};
    std::atomic<int64_t> last_share_ms{0};
    std::atomic<int64_t> last_accept_ms{0};
    std::atomic<int64_t> last_reject_ms{0};
    std::atomic<int64_t> last_nonce_ms{0};
    std::atomic<bool> watchdog_reconnect_requested{false};
    mutable std::mutex watchdog_mu;
    std::string watchdog_status{"ok"};
    std::string watchdog_last_warning;
    std::string watchdog_last_action;
    double total_solve_ms{0.0};
    int64_t started_ms{MonoMs()};
    std::chrono::steady_clock::time_point started{std::chrono::steady_clock::now()};
    // Rolling-window throughput averages: written by the heartbeat (which owns the sample
    // history), read by the status API's /summary. Telemetry only -> relaxed atomics.
    // ENC_RC unit of work is the EPISODE, not the nonce: one episode = one full
    // dependent INT8 GEMM chain for one nonce, ~825 ms of GPU on a 5090. The v3
    // rates that used to live here (digest candidates/s, pre-hash scan nonces/s)
    // measured a pipeline that no longer exists.
    struct WindowAvg {
        std::atomic<double> episode_per_s{0.0};
        std::atomic<double> pool_episode_per_s{0.0};   // pool-credited rate, from accepted shares
        std::atomic<double> acc_per_hr{0.0};
    };
    WindowAvg avg_1h;
    WindowAvg avg_24h;
};

static void WatchdogSetStatus(Stats& stats,
                              const std::string& status,
                              const std::string& warning,
                              const std::string& action)
{
    std::lock_guard<std::mutex> lk(stats.watchdog_mu);
    stats.watchdog_status = status;
    if (!warning.empty()) stats.watchdog_last_warning = warning;
    if (!action.empty()) stats.watchdog_last_action = action;
}

// Live auto-update visibility, shared between the periodic update-checker thread and
// the status API (/summary "update" block). Lets a fleet hub see each rig's running
// version + the newest tag it has observed without scraping logs.
struct UpdateState {
    mutable std::mutex mu;
    std::string current{MATADOR_MINER_VERSION};  // version THIS process is running
    std::string latest_seen;                     // newest tag observed from GitHub ("" until first check)
    int64_t last_check_ms{0};                    // MonoMs of the last completed check (0 = never)
    std::string channel{"stable"};               // stable|prerelease (mirrors cfg.update_channel)
    bool auto_update{true};                       // mirrors cfg.auto_update
};

// Idle-gate ("mine when the box is idle, yield when it's needed"). A poller thread runs
// the operator's --should-mine-command on an interval (exit 0 = mine, non-zero = yield,
// matching the node's `test ! -f .pause-mining` convention) and publishes the verdict
// here. The solo + pool loops read it to pause; /summary + the fleet hub surface it.
static std::atomic<bool> g_gate_enabled{false};  // a --should-mine-command was configured
static std::atomic<bool> g_gate_mine{true};      // current verdict: true = mine, false = gated (yield)

// ---- live pool difficulty / share-rate telemetry -------------------------------
// Published from the stratum path (set_difficulty + per-job decode) and read by the
// [stats] heartbeat. TELEMETRY ONLY: none of this is ever read back into the solve
// or the digest, so the byte-exact PoW math is completely untouched.
static std::atomic<double> g_pool_net_diff{0.0};           // network difficulty (from job nbits)
// Difficulty of the share target we ACTUALLY solve against, derived from the per-job
// target (mining.notify[6] / the login dialect's shareTarget). This is the only share
// difficulty that means anything: it is the number the pool grades our submits by.
static std::atomic<double> g_pool_share_diff{0.0};
// The raw number a pool announces via mining.set_difficulty, kept SEPARATE because it is
// NOT in the same units as the per-job target and must never be mixed with it. Only
// minebtx sends one, and it does not gate our shares: minebtx has accepted 21.5k shares
// solved against notify[6] with zero low-difficulty rejects, so the announced value is
// vestigial. Diagnostic readout only -- never compared against, or substituted for, the
// per-job target.
static std::atomic<double> g_pool_announced_diff{0.0};
static std::atomic<double> g_pool_attempts_per_share{0.0}; // expected digests per share = 2^256 / share_target
static std::atomic<int32_t> g_pool_block_height{0};        // height of the job currently being mined
static std::atomic<int64_t> g_pool_connected_ms{0};        // MonoMs when the CURRENT pool connection came up (0 = down)
static std::atomic<double>  g_pool_latency_ms{0.0};        // EMA of request->response RTTs (handshake + submits); 0 = unmeasured
static std::mutex        g_gate_mu;
static std::string       g_gate_reason;          // last stdout line of the gate command (human reason)

static inline bool GateAllowsMining() { return !g_gate_enabled.load() || g_gate_mine.load(); }
static std::string GateReason() { std::lock_guard<std::mutex> lk(g_gate_mu); return g_gate_reason; }
static std::string MiningStateStr() {
    if (!g_gate_enabled.load()) return "mining";
    return g_gate_mine.load() ? "mining" : "gated";
}

// ===========================================================================
// SELF-ATTEST (solo + trusted-mirror node): sign our OWN solved block and submit
// the attestation BEFORE submitblock, so the node holds 1-of-1 quorum the instant
// the block arrives and connects it synchronously -- instead of parking it in the
// trusted-wait deferral queue (~1 retry/min) until the external keeper's next 10s
// tick. In the solo height race every second of deferral is orphan exposure; this
// closes that window to ~0.
//
// TRUST MODEL -- no replay here, ON PURPOSE: the digest in this header was
// produced by our own byte-exact GPU episode and beat the target inside
// SolveMatMul moments ago; the solve IS the ExactReplay. rc_attest's re-replay
// gate exists for FOREIGN blocks; we only ever sign a block this process solved.
// Wire format mirrors harness/rc_attest.cpp == the node's
// src/matmul/trusted_exact_replay_attestation.{h,cpp} (V2, 103-byte statement).
// OFF unless attest_key_file is configured (fleet default: off, zero cost).
// ===========================================================================
struct SelfAttestState {
    bool armed{false};
    CKey key;
    CPubKey pub;
    uint256 context;                 // replay_authority_context (null until known)
    uint256 chain_id;                // genesis hash of the active chain
    std::optional<ECC_Context> ecc;  // must outlive every Sign call
};
static SelfAttestState g_self_attest;

// Fetch replay_authority_context from the node. Soft-fail: logs and returns false;
// callers retry at the next solve.
static bool SelfAttestFetchContext(RpcClient& rpc)
{
    Timer sp;
    try {
        UniValue res = rpc.Call("getmatmultrustedstatus", UniValue(UniValue::VARR));
        if (!res.exists("configured") || !res["configured"].get_bool()) {
            LOGW("[self-attest] node reports trusted mode NOT configured; attestations would be ignored");
            return false;
        }
        const std::string ctx = res.exists("replay_authority_context")
                                    ? res["replay_authority_context"].get_str() : "";
        const auto parsed = uint256::FromHex(ctx);
        if (!parsed) {
            LOGW("[self-attest] getmatmultrustedstatus returned no usable replay_authority_context"
                 " (got \"" << ctx << "\")");
            return false;
        }
        g_self_attest.context = *parsed;
        LOGI("[self-attest] authority context " << ctx << " (fetched in " << sp.ms() << "ms)");
        return true;
    } catch (const std::exception& e) {
        LOGW("[self-attest] context fetch failed: " << e.what() << " (will retry at next solve)");
        return false;
    }
}

static void SelfAttestOwnBlock(RpcClient& rpc, const uint256& block_hash, int32_t height);

// Arm self-attest from config at solo startup. Loudly refuses (and stays OFF) on
// any key problem so a misconfiguration is visible at start, not at the first win.
// Plain-string params because this header precedes miner_config.h in the TU.
static void InitSelfAttest(RpcClient& rpc, const std::string& key_file, const std::string& context_hex)
{
    if (key_file.empty()) {
        LOGD("[self-attest] disabled (no attest_key_file configured)");
        return;
    }
    std::ifstream kf(key_file);
    if (!kf) {
        LOGE("[self-attest] cannot read attest_key_file=" << key_file
             << " -- self-attest STAYS OFF (external keeper still covers solves)");
        return;
    }
    std::string hex;
    kf >> hex;   // first token; tolerates a trailing newline
    const auto keybytes = TryParseHex<unsigned char>(hex);
    if (!keybytes || keybytes->size() != 32) {
        LOGE("[self-attest] attest_key_file is not 64 hex chars -- self-attest STAYS OFF");
        return;
    }
    g_self_attest.ecc.emplace();
    g_self_attest.key.Set(keybytes->begin(), keybytes->end(), /*fCompressedIn=*/true);
    if (!g_self_attest.key.IsValid()) {
        LOGE("[self-attest] invalid signing key -- self-attest STAYS OFF");
        g_self_attest.ecc.reset();
        return;
    }
    g_self_attest.pub = g_self_attest.key.GetPubKey();
    g_self_attest.chain_id = Params().GenesisBlock().GetHash();
    if (!context_hex.empty()) {
        const auto ctx = uint256::FromHex(context_hex);
        if (ctx) g_self_attest.context = *ctx;
        else LOGW("[self-attest] attest_context is not 64-hex; falling back to RPC fetch");
    }
    if (g_self_attest.context.IsNull()) SelfAttestFetchContext(rpc);   // soft-fail: retried per solve
    g_self_attest.armed = true;
    LOGI("[self-attest] ARMED pubkey=" << HexStr(g_self_attest.pub)
         << " chain_id=" << Short(g_self_attest.chain_id.GetHex())
         << (g_self_attest.context.IsNull() ? " context=PENDING" : " context=ok")
         << " -- solved blocks are attested BEFORE submitblock");

    // BTX_ATTEST_SELFCHECK=<height>: attest the (already-attested) block at that
    // height once at startup. The node answering "duplicate" (or "accepted") proves
    // the whole path -- statement bytes, key, RPC -- end-to-end without waiting for
    // a block win. Deploy-verification hook; not set in normal operation.
    if (const char* sc = std::getenv("BTX_ATTEST_SELFCHECK")) {
        try {
            const int h = std::max(0, std::atoi(sc));
            UniValue p(UniValue::VARR);
            p.push_back(h);
            const UniValue bh = rpc.Call("getblockhash", p);
            const auto hash = uint256::FromHex(bh.get_str());
            if (!hash) { LOGE("[self-attest] selfcheck: bad getblockhash result"); return; }
            LOGI("[self-attest] SELFCHECK attesting h=" << h << " hash=" << Short(bh.get_str())
                 << " (expect accepted/duplicate)");
            SelfAttestOwnBlock(rpc, *hash, static_cast<int32_t>(h));
        } catch (const std::exception& e) {
            LOGE("[self-attest] selfcheck failed: " << e.what());
        }
    }
}

// Sign + submit the attestation for OUR OWN solved block. Fail-open at every step:
// any failure logs and returns, submitblock proceeds regardless, and the external
// keeper remains the safety net.
static void SelfAttestOwnBlock(RpcClient& rpc, const uint256& block_hash, int32_t height)
{
    if (!g_self_attest.armed) return;
    Timer sp;
    if (g_self_attest.context.IsNull() && !SelfAttestFetchContext(rpc)) {
        LOGW("[self-attest] no authority context; SKIPPING h=" << height << " (keeper will cover)");
        return;
    }

    // 103-byte V2 statement: ver | chain_id | block_hash | height(LE) | major | profile | context
    std::vector<unsigned char> st;
    st.reserve(103);
    st.push_back(2);
    st.insert(st.end(), g_self_attest.chain_id.begin(), g_self_attest.chain_id.end());
    st.insert(st.end(), block_hash.begin(), block_hash.end());
    for (int i = 0; i < 4; ++i) st.push_back(static_cast<unsigned char>((height >> (8 * i)) & 0xff));
    st.push_back(4);   // matmul major
    st.push_back(1);   // profile
    st.insert(st.end(), g_self_attest.context.begin(), g_self_attest.context.end());
    if (st.size() != 103) { LOGE("[self-attest] statement " << st.size() << " != 103; skipping"); return; }

    // Domain-separated dSHA256: CompactSize(39) || domain || statement
    static const char* kAttestDomain = "BTX_TRUSTED_EXACT_REPLAY_ATTESTATION_V2";
    HashWriter hasher;
    const unsigned char dlen = static_cast<unsigned char>(std::strlen(kAttestDomain));
    hasher.write(Span<const std::byte>(reinterpret_cast<const std::byte*>(&dlen), 1));
    hasher.write(Span<const std::byte>(reinterpret_cast<const std::byte*>(kAttestDomain), dlen));
    hasher.write(Span<const std::byte>(reinterpret_cast<const std::byte*>(st.data()), st.size()));
    const uint256 msg = hasher.GetHash();

    std::vector<unsigned char> sig;
    if (!g_self_attest.key.Sign(msg, sig)) { LOGE("[self-attest] sign failed h=" << height); return; }

    std::vector<unsigned char> att = st;
    att.push_back(static_cast<unsigned char>(g_self_attest.pub.size()));   // 33
    att.insert(att.end(), g_self_attest.pub.begin(), g_self_attest.pub.end());
    att.push_back(static_cast<unsigned char>(sig.size()));                 // DER < 253 = 1-byte CompactSize
    att.insert(att.end(), sig.begin(), sig.end());

    UniValue inner(UniValue::VARR);
    inner.push_back(HexStr(att));
    UniValue att_params(UniValue::VARR);
    att_params.push_back(inner);
    try {
        UniValue res = rpc.Call("submitmatmulattestations", att_params);
        const std::string rs = res.write();
        const bool ok = rs.find("accepted") != std::string::npos ||
                        rs.find("duplicate") != std::string::npos;
        if (ok) {
            LOGI("[self-attest] quorum pre-staged h=" << height
                 << " hash=" << Short(block_hash.GetHex())
                 << " in " << sp.ms() << "ms (block connects without trusted-wait)");
        } else {
            LOGW("[self-attest] node did not accept attestation h=" << height
                 << " res=" << rs.substr(0, 200) << " (submitblock proceeds; keeper covers)");
        }
    } catch (const std::exception& e) {
        LOGW("[self-attest] submit failed h=" << height << ": " << e.what()
             << " (submitblock proceeds; keeper covers)");
    }
}

// ===========================================================================
// PARKED SOLVES: a won block must never exist only in RAM.
// h=186889 (2026-08-12) was solved, survived 70 s of submit retries, then died
// on the node's warmup error -- and with it ~20 BTX, because the only copy of
// the bytes was a local. Worse, its HEADER had already registered node-side, so
// the unresolvable branch then poisoned best-header until hand-invalidated.
// Now: any solve that exhausts the retry window is written to disk, and every
// later solve first tries to push whatever is parked. Cheap insurance -- the
// file is ~1 KB and the retry is one RPC when the directory is empty.
// ===========================================================================
static std::filesystem::path UnsubmittedDir()
{
    if (const char* d = std::getenv("BTX_SOLO_SAVE_DIR")) return std::filesystem::path(d);
    const char* home = std::getenv("HOME");
    return std::filesystem::path(home ? home : ".") / ".local/share/matador-miner/unsubmitted";
}

static void ParkUnsubmittedBlock(int32_t height, const std::string& hash, const std::string& hex)
{
    std::error_code ec;
    const auto dir = UnsubmittedDir();
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / (std::to_string(height) + "-" + hash + ".hex");
    std::ofstream f(path, std::ios::trunc);
    if (!f) { LOGE("[submit] could NOT park unsubmitted block at " << path.string()); return; }
    f << hex;
    f.close();
    LOGW("[submit] PARKED unsubmitted block h=" << height << " -> " << path.string()
         << " (retried before every later solve; `btx-cli submitblock $(cat <file>)` also works)");
}

// Push anything parked by an earlier failed submit. Runs before each solve.
static void RetryParkedBlocks(RpcClient& rpc)
{
    std::error_code ec;
    const auto dir = UnsubmittedDir();
    if (!std::filesystem::exists(dir, ec)) return;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) return;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        const auto dash = name.find('-');
        const auto dot = name.rfind(".hex");
        if (dash == std::string::npos || dot == std::string::npos || dot <= dash) continue;
        int32_t height = 0;
        try { height = static_cast<int32_t>(std::stol(name.substr(0, dash))); } catch (...) { continue; }
        const std::string hash = name.substr(dash + 1, dot - dash - 1);
        std::string hex;
        { std::ifstream f(entry.path()); f >> hex; }
        if (hex.empty()) { std::filesystem::remove(entry.path(), ec); continue; }

        LOGI("[submit] retrying parked block h=" << height << " hash=" << Short(hash));
        if (const auto h = uint256::FromHex(hash)) SelfAttestOwnBlock(rpc, *h, height);
        UniValue p(UniValue::VARR);
        p.push_back(hex);
        std::string why;
        bool done = false;
        try {
            const UniValue r = rpc.Call("submitblock", p);
            if (r.isNull()) {
                LOGI("[submit] parked block h=" << height << " ACCEPTED on retry");
                done = true;
            } else {
                why = r.write();
            }
        } catch (const std::exception& e) { why = e.what(); }
        // "duplicate" = the chain already has it (we or the network connected it).
        if (!done && why.find("duplicate") != std::string::npos) {
            LOGI("[submit] parked block h=" << height << " already in the chain; clearing");
            done = true;
        }
        if (done) { std::filesystem::remove(entry.path(), ec); continue; }
        // Give up on blocks that can no longer win: a height that far behind is dead.
        LOGW("[submit] parked block h=" << height << " still not accepted (\"" << why << "\")");
    }
}

static bool SolveAndSubmit(RpcClient& rpc,
                           const Consensus::Params& consensus,
                           CBlock& block,
                           const TemplateInputs& meta,
                           std::atomic<bool>& abort_flag,
                           uint64_t max_tries_in,
                           unsigned solver_threads,
                           Stats& stats)
{
    RetryParkedBlocks(rpc);   // no-op when nothing is parked
    // Wire the shared abort flag into the RPC helper: the tip-watcher thread
    // longpolls getblocktemplate on this SAME RpcClient while we solve, and the
    // solo loop joins it right after this function returns -- registering the flag
    // lets that in-flight longpoll abort within ~1s of the flag flipping instead of
    // pinning the join for the full curl timeout (was up to 120s after a rejected
    // submitblock). abort_flag is function-scope in the solo main loop, so the
    // pointer outlives every call on this client.
    rpc.SetLongpollAbortFlag(&abort_flag);

    // ---- reconcile the header matmul_dim against consensus (single decision point) ----
    // AssembleBlock stamped whatever GBT's matmul_n said (0 = the node sent none). The dim is in
    // BOTH the seed-V3 and sigma preimages, so getting it wrong does not merely mis-shape the
    // grind -- it produces a digest no node can ever accept.
    {
        const bool rc_active = consensus.IsMatMulRCActive(static_cast<int32_t>(meta.height));
        const uint16_t rc_dim = matmul::v4::rc::RCConsensusHeaderMatmulDim();

        if (rc_active && block.matmul_dim != 0 && block.matmul_dim != rc_dim) {
            // Same posture as the pool path: at RC heights consensus does not accept a choice,
            // so an operator pin plus a stale node (still serving the v3 dim) must not be able
            // to send us grinding a shape that can never validate. Consensus wins, loudly.
            static std::atomic<int32_t> s_warned_dim{0};
            const int32_t seen = static_cast<int32_t>(block.matmul_dim);
            int32_t prev = s_warned_dim.load(std::memory_order_relaxed);
            if (prev != seen && s_warned_dim.compare_exchange_strong(prev, seen,
                                                                    std::memory_order_relaxed)) {
                LOGW("[solve] node template at RC height " << meta.height << " carries matmul_n="
                     << block.matmul_dim << " but ENC_RC consensus requires " << rc_dim
                     << " -- using " << rc_dim << ". Solves would be rejected otherwise;"
                     << " your node is serving a pre-RC dimension for an RC height.");
            }
            block.matmul_dim = rc_dim;
        } else if (block.matmul_dim == 0) {
            // RC-aware fallback: at RC heights consensus requires matmul_dim ==
            // nMatMulV4Dimension (4096 mainnet). Only reached when GBT carried no matmul_n
            // (pre-v4 node, or a non-btx coordinator).
            block.matmul_dim = rc_active ? rc_dim
                                         : static_cast<uint16_t>(consensus.nMatMulDimension);
            if (!rc_active) {
                // About to grind the v3 dim with no corroboration from anyone. If the chain is
                // in fact at an RC height every solve here is unacceptable and the only symptom
                // would be silence. Say so once rather than mining garbage quietly.
                static std::atomic<bool> s_warned{false};
                if (!s_warned.exchange(true)) {
                    LOGW("[solve] height " << meta.height << ": template carried no matmul_n and"
                         << " ENC_RC is not latched, so this solve uses the v3 dimension "
                         << consensus.nMatMulDimension << " and the base v3 solver."
                         << " If the chain is at or above the ENC_RC activation height this block"
                         << " can NEVER be accepted -- pin --rc-height (or rc_height in config)"
                         << " and confirm your node serves matmul_n in getblocktemplate.");
                }
            }
        }
    }
    block.mix_hash.SetNull();

    const unsigned nthreads = std::max(1u, solver_threads);
    LOGI("[solve] start height=" << meta.height
         << " max_tries=" << max_tries_in
         << " solver_threads=" << nthreads
         << " matmul_dim=" << block.matmul_dim
         << " parent_mtp=" << meta.parent_mtp);

    Timer sp;

    // Fan out nthreads concurrent GPU solves over DISJOINT nonce ranges. SolveMatMul
    // scans upward from block.nNonce64 (pow.cpp:4060), so each thread starts a budget
    // apart and never overlaps the others. Each solves a full, independent candidate
    // (seed_a/seed_b are re-derived from the nonce), so the first VALID block wins and
    // signals the rest via the shared abort_flag. One stream saturates a 5090; Metal
    // needs the concurrency to fill submit/sync latency (~6x on the M4 Max).
    // Randomize the solo starting nonce (BTX_SOLO_NONCE_RANDOM=0 restores the
    // deterministic base). Lived failure 2026-08-11: a stuck tip serves a CACHED,
    // byte-identical template, so every service bounce re-ground the SAME leading
    // ~600 nonces -- five bounced solves, ~40 min of full-power GPU, zero fresh
    // work, while the winning nonce sat beyond the restart horizon. A random u64
    // base makes every (re)start cover virgin nonce space; the field is 2^64 so
    // overlap odds are negligible.
    if (const char* nr = std::getenv("BTX_SOLO_NONCE_RANDOM"); nr == nullptr || std::strcmp(nr, "0") != 0) {
        std::random_device rd;
        const uint64_t rbase = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd()) ^
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        block.nNonce64 = rbase;
        block.nNonce = static_cast<uint32_t>(rbase);
        LOGI("[solve] nonce base randomized to " << rbase << " (BTX_SOLO_NONCE_RANDOM=0 disables)");
    }

    const uint64_t base_nonce = block.nNonce64;
    const uint64_t per_thread = std::max<uint64_t>(1, max_tries_in / nthreads);

    std::mutex win_mu;
    bool have_winner = false;
    CBlock winning_block;
    std::atomic<uint64_t> attempted_total{0};

    auto solve_range = [&](unsigned i) {
        CBlock b = block;                                  // coinbase/txs/merkle identical
        b.nNonce64 = base_nonce + static_cast<uint64_t>(i) * per_thread;
        b.nNonce   = static_cast<uint32_t>(b.nNonce64);
        const uint64_t budget = (i == nthreads - 1)
                                    ? (max_tries_in - per_thread * (nthreads - 1))
                                    : per_thread;
        if (budget == 0) return;
        uint64_t tries = budget;                            // [in/out] remaining after return
        const bool solved = SolveMatMul(
            b, consensus, tries, meta.height, &abort_flag,
            /*share_target_override=*/nullptr,
            /*parent_median_time_past=*/std::optional<int64_t>(meta.parent_mtp));
        const uint64_t attempted = budget >= tries ? (budget - tries) : 0;
        attempted_total.fetch_add(attempted);
        if (solved) {
            std::lock_guard<std::mutex> lk(win_mu);
            if (!have_winner) {
                have_winner = true;
                winning_block = std::move(b);
                abort_flag.store(true);                     // stop the other solver threads
            }
        }
    };

    if (nthreads == 1) {
        solve_range(0);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(nthreads);
        for (unsigned i = 0; i < nthreads; ++i) workers.emplace_back(solve_range, i);
        for (auto& w : workers) w.join();
    }

    const double solve_ms = sp.ms();
    const uint64_t attempted = attempted_total.load();
    stats.total_nonces.fetch_add(attempted);
    if (attempted > 0) stats.last_nonce_ms.store(MonoMs());
    const double nps = solve_ms > 0.0 ? (attempted / (solve_ms / 1000.0)) : 0.0;

    LOGI("[solve] done height=" << meta.height
         << " solved=" << (have_winner ? "true" : "false")
         << " nonces_attempted=" << attempted
         << " nonces_per_s=" << static_cast<uint64_t>(nps)
         << " aborted=" << (abort_flag.load() ? "true" : "false")
         << " in " << solve_ms << "ms");

    if (!have_winner) {
        // every range exhausted, aborted (tip change), or sentinel.
        return false;
    }

    // Submit the winning candidate; everything below operates on it.
    block = std::move(winning_block);

    // No Freivalds carrier at v4 heights: the ENC_RC episode digest IS the proof
    // (Epoch A / Profile 1, ExactReplay authority), so the block carries no product
    // matrix. Clear the field so a template that arrived with one cannot smuggle it
    // into the submitted block.
    block.matrix_c_data.clear();


    // ---- serialize the FULL block (TX_WITH_WITNESS) and submitblock ----
    DataStream ss;
    ss << TX_WITH_WITNESS(block);
    const std::string block_hex = HexStr(ss);
    const std::string block_hash = block.GetHash().GetHex();

    LOGD("[submit] serializing height=" << meta.height
         << " hash=" << Short(block_hash)
         << " nonce64=" << block.nNonce64
         << " serialized_bytes=" << ss.size());

    // NO pre-staging is possible for a fresh solve (both proven live 2026-08-11):
    // submitheader is REFUSED on MatMul chains ("submit full blocks with
    // submitblock", h=186858), and the node rejects attestations for blocks it
    // has never seen ("Unknown or non-Profile-1 block", h=186855). The working
    // path is below: submitblock (bounces on quorum but STORES our header) ->
    // self-attest (accepted now) -> resubmit the same bytes (connects, ~180 ms
    // solve-to-connected on the h=186858 win). Upstream ask on file: give RPC
    // submitblock the same short trusted-wait P2P blocks get, or accept
    // store-ahead attestations for unknown blocks -- then the first submit
    // connects directly.

    UniValue params(UniValue::VARR);
    params.push_back(block_hex);

    Timer sub_sp;
    // submitblock returns null on accept; a reject reason otherwise. Wrapped so the
    // quorum-bounce path below can retry the SAME serialized bytes exactly once.
    auto submit_once = [&](std::string& why) -> bool {
        try {
            UniValue r = rpc.Call("submitblock", params);
            if (r.isNull()) { why.clear(); return true; }
            why = r.write();
            return false;
        } catch (const std::exception& e) {
            why = std::string("rpc threw: ") + e.what();
            return false;
        }
    };

    std::string why;
    bool ok = false;
    // Ride out a btxd wedge/heal: a solve landing while the node is mid-restart
    // used to be DISCARDED on the transport error. The watchdog heal cycle is
    // ~90-120 s (confirm + forensics + SIGTERM flush + respawn), so retry
    // transport failures for up to 4 min. Consensus rejections exit immediately.
    const int64_t submit_deadline = MonoMs() + 360000;
    for (;;) {
        ok = submit_once(why);
        if (!ok && (why.find("quorum") != std::string::npos ||
                    why.find("attestation") != std::string::npos)) {
            // The failed submit stored our header node-side, so the attestation is
            // now acceptable even if the pre-submit attest failed. Attest + resubmit.
            LOGW("[submit] quorum bounce h=" << meta.height << " (\"" << why
                 << "\"); attesting + resubmitting the same block");
            SelfAttestOwnBlock(rpc, block.GetHash(), static_cast<int32_t>(meta.height));
            ok = submit_once(why);
        }
        if (ok) break;
        // Retryable: curl transport errors, http-level failures (503 while the
        // supervisor respawns btxd; 401 self-heals inside Call but a failed refresh
        // mid-respawn surfaces here), and RPC warmup (code -28: "Loading wallet...",
        // lost solve h=186889 -- the node answered mid-initialization and the old
        // taxonomy read it as fatal).
        const bool transport = why.find("rpc transport:") != std::string::npos ||
                               why.find("unparseable response (http") != std::string::npos ||
                               why.find("[code -28]") != std::string::npos;
        if (!transport || MonoMs() >= submit_deadline) break;
        LOGW("[submit] transport failure h=" << meta.height << " (\"" << why
             << "\"); node likely mid-heal -- retrying in 5s (deadline "
             << (submit_deadline - MonoMs()) / 1000 << "s away)");
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    if (ok) {
        stats.accepted.fetch_add(1);
        stats.reject_streak.store(0);
        const int64_t now_ms = MonoMs();
        stats.last_accept_ms.store(now_ms);
        stats.last_share_ms.store(now_ms);
        LOGI("[submit] ACCEPTED height=" << meta.height
             << " hash=" << block_hash
             << " nonce64=" << block.nNonce64
             << " solve_ms=" << solve_ms << " submit_ms=" << sub_sp.ms());
        return true;
    }

    stats.rejected.fetch_add(1);
    stats.reject_streak.fetch_add(1);
    stats.last_reject_ms.store(MonoMs());
    LOGE("[submit] REJECTED height=" << meta.height
         << " hash=" << Short(block_hash)
         << " nonce64=" << block.nNonce64
         << " reason=\"" << why << "\""
         << " solve_ms=" << solve_ms << " submit_ms=" << sub_sp.ms());
    // The bytes are the only copy of a won block; never let them die with this frame.
    ParkUnsubmittedBlock(static_cast<int32_t>(meta.height), block_hash, block_hex);
    return false;
}

