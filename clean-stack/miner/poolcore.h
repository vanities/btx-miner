// poolcore.h -- wrapper-driven compute-core mode (--poolcore) for pool integrations
// (first consumer: minebtx / Atticus wrapper, spec 2026-07-24).
//
// CONTRACT (poolcore-v0, see docs/poolcore-protocol-v0.md): the miner becomes a supervised
// subprocess. The WRAPPER owns the pool connection, session/nonce policy, share submission,
// physical GPU telemetry (NVML on its side), and the update channel. Matador owns solving and
// solver-internal telemetry only. Transport: JSON-lines -- one message per line, wrapper->core
// on stdin, core->wrapper on stdout. stdout is EXCLUSIVELY the protocol channel; every log
// stays on stderr (mlog/log-tee are stderr-only, so normal logging is already safe).
//
// In this mode we deliberately run NONE of: stratum/pool client, auto-updater, dev-fee lane
// (fee is collected pool-side per the deployment spec), solo prober, status HTTP API.
//
// STATUS: protocol loop + init/device identity + stats + preempt/shutdown are real; "btx-live"
// job DISPATCH is WIRED (job -> CBlockHeader + share target -> SolveMatMul on a worker thread
// with the share-sink -> Emit {type:solution,kind:share} the moment a share hits; preempt/new-job
// abort). "enc-rc-v46"/"enc-rc-v47" route to the RC episode solver, DEFAULT ON since v0.9.2
// (BTX_RC_ENABLE_EPISODE_SOLVE=0 force-disables).
// Debug play-by-play behind BTX_POOLCORE_DEBUG (off by default). Unknown messages -> {type:error}.

#pragma once

// Version string: normally defined by the build (-DMATADOR_MINER_VERSION=...) before this
// header is included from matador-miner.cpp; the fallback mirrors the miner's own default so
// the header also compiles standalone (lint/tests).
#ifndef MATADOR_MINER_VERSION
#define MATADOR_MINER_VERSION "v0.1.0-dev"
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <unistd.h>          // dup/dup2: fence the protocol stream off from stray stdout writes

#include <pow.h>                 // SolveMatMul (share sink overload) + SolveMatMulRCEpisode
#include <primitives/block.h>    // CBlockHeader
#include <chainparams.h>         // SelectParams / Params()
#include <util/chaintype.h>
#include <consensus/params.h>
#include <uint256.h>
#include <arith_uint256.h>
#include <matmul/matmul_v4_rc.h> // EpisodeParams, DatacenterEpisodeParams, ComputeEpisodeRoundRoots
#include <matmul/matmul_v4_rc_coupled.h> // CoupParamsV3, Production/MediumV3CoupParams, RCBankTemplateHash
#include <matmul/matmul_pow.h>   // DeriveSigma (for the witness round_roots)
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace poolcore {

constexpr int kSchemaVersion = 1;
constexpr const char* kProtocol = "poolcore-v0";

// ---- tiny JSON helpers (scaffold-grade) --------------------------------------------------
// Input is a trusted same-host wrapper, one object per line. These extract top-level string /
// u64 fields without a full parser; the real dispatch commit swaps in proper parsing. Output
// escaping matches status_api.h's JsonEscape (duplicated locally so this header stays
// self-contained and testable).

inline std::string PcJsonEscape(const std::string& in)
{
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

inline std::string PcJsonStr(const std::string& s) { return "\"" + PcJsonEscape(s) + "\""; }

// Extract "key":"value" (string) from a single-line JSON object. Returns false if absent.
inline bool PcGetString(const std::string& line, const std::string& key, std::string& out)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return false;
    p = line.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    p = line.find('"', p);
    if (p == std::string::npos) return false;
    const size_t q = line.find('"', p + 1);
    if (q == std::string::npos) return false;
    out = line.substr(p + 1, q - p - 1);
    return true;
}

// Extract "key":123 (unsigned) from a single-line JSON object.
inline bool PcGetU64(const std::string& line, const std::string& key, uint64_t& out)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return false;
    p = line.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < line.size() && (line[p] == ' ')) ++p;
    uint64_t v = 0;
    bool any = false;
    while (p < line.size() && line[p] >= '0' && line[p] <= '9') {
        v = v * 10 + static_cast<uint64_t>(line[p] - '0');
        ++p;
        any = true;
    }
    if (!any) return false;
    out = v;
    return true;
}

inline uint64_t PcNowMs()
{
    return static_cast<uint64_t>(std::time(nullptr)) * 1000ULL;
}

// One line out, immediately flushed: the wrapper polls; a buffered reply is a hung wrapper.
inline std::mutex& EmitMutex() { static std::mutex m; return m; }

// The protocol stream, held as a PRIVATE dup of the original stdout.
//
// "stdout is EXCLUSIVELY the protocol channel" was a rule we merely intended to follow, and it
// broke in the field: a `[rc-tune-cache] loaded N tuned algo(s)` line from the CUDA GEMM tuner
// went to stdout, landed in the middle of the JSON stream, and the pool answered
// {"status":"rejected","reason":"bad-json"} and threw away the whole connection. That one file
// alone has ~49 bare printf calls, and any vendored code we link can add more, so auditing call
// sites is not a fix -- the next stray line is one dependency bump away.
//
// Instead make it structural: dup the real stdout for Emit's private use, then point fd 1 at
// stderr. After this, ANY printf / std::cout / third-party write in the process lands in the log
// where it is harmless, and the only thing that can reach the wrapper is Emit.
inline FILE*& PcProtoOut() { static FILE* f = nullptr; return f; }

inline void PcBindProtocolStdout()
{
    if (PcProtoOut() != nullptr) return;          // idempotent
    std::fflush(stdout);
    const int fd = ::dup(1);
    if (fd < 0) { PcProtoOut() = stdout; return; }   // no dup: degrade to old behaviour
    FILE* f = ::fdopen(fd, "w");
    if (f == nullptr) { ::close(fd); PcProtoOut() = stdout; return; }
    ::setvbuf(f, nullptr, _IOLBF, 0);
    PcProtoOut() = f;
    ::dup2(2, 1);                                  // stray stdout writes -> stderr
}

inline void Emit(const std::string& json_line)
{
    std::lock_guard<std::mutex> lk(EmitMutex());
    FILE* f = PcProtoOut() != nullptr ? PcProtoOut() : stdout;
    std::fwrite(json_line.data(), 1, json_line.size(), f);
    std::fputc('\n', f);
    std::fflush(f);
}

inline void EmitError(const std::string& what, const std::string& detail)
{
    Emit(std::string("{\"type\":\"error\",\"what\":") + PcJsonStr(what) +
         ",\"detail\":" + PcJsonStr(detail) + ",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
}

// ---- device identity ---------------------------------------------------------------------
// DeviceBinding join keys (spec §3.1): NVML gpu UUID (primary) + PCI bus id (fallback).
// Same source of truth as status_api.h's identity latch: one nvidia-smi query at init.
// Metal/HIP: no NVML UUID exists; those backends will emit a stable platform id with a
// uuid_kind tag (documented in the protocol doc) -- not wired in this scaffold.

struct PcDevice {
    uint32_t solver_index = 0;
    std::string gpu_uuid;
    std::string pci_bus_id;
    std::string name;
    std::string backend = "cuda";
};

inline std::vector<PcDevice> QueryDevices()
{
    std::vector<PcDevice> out;
#if !defined(__APPLE__)
    // Absolute-path fallbacks: under `socat EXEC:` (the wrapper transport shib uses) the child
    // inherits a minimal PATH, bare `nvidia-smi` is not found, and every device came back as
    // gpu_uuid "unknown" -- which is precisely the DeviceBinding join key the wrapper needs.
    static const char* kSmiCandidates[] = {
        "nvidia-smi",
        "/usr/bin/nvidia-smi",
        "/usr/local/bin/nvidia-smi",
    };
    FILE* p = nullptr;
    for (const char* smi : kSmiCandidates) {
        const std::string cmd =
            std::string(smi) + " --query-gpu=uuid,pci.bus_id,name --format=csv,noheader 2>/dev/null";
        p = popen(cmd.c_str(), "r");
        if (p == nullptr) continue;
        const int c = fgetc(p);
        if (c != EOF) { ungetc(c, p); break; }   // produced output: use this one
        pclose(p);
        p = nullptr;
    }
    if (p == nullptr) return out;
    char buf[512];
    uint32_t idx = 0;
    while (fgets(buf, sizeof(buf), p) != nullptr) {
        std::string row(buf);
        while (!row.empty() && (row.back() == '\n' || row.back() == '\r')) row.pop_back();
        if (row.empty()) continue;
        PcDevice d;
        d.solver_index = idx++;
        // csv,noheader: "GPU-xxxx, 00000000:01:00.0, NVIDIA GeForce RTX 5090"
        size_t c1 = row.find(", ");
        if (c1 == std::string::npos) continue;
        size_t c2 = row.find(", ", c1 + 2);
        if (c2 == std::string::npos) continue;
        d.gpu_uuid = row.substr(0, c1);
        d.pci_bus_id = row.substr(c1 + 2, c2 - c1 - 2);
        d.name = row.substr(c2 + 2);
        out.push_back(d);
    }
    pclose(p);
#endif
    return out;
}

inline std::string DeviceJson(const PcDevice& d)
{
    std::ostringstream o;
    o << "{\"solver_index\":" << d.solver_index
      << ",\"gpu_uuid\":" << PcJsonStr(d.gpu_uuid)
      << ",\"pci_bus_id\":" << PcJsonStr(d.pci_bus_id)
      << ",\"name\":" << PcJsonStr(d.name)
      << ",\"backend\":" << PcJsonStr(d.backend) << "}";
    return o.str();
}

// ---- btx-live dispatch -------------------------------------------------------------------
// job -> CBlockHeader (80-byte classic template, raw serialized order) + share target ->
// worker thread looping SolveMatMul with the share-sink (shares emitted the moment the solver
// finds them, same hot pipeline as the stratum path). Preempt/new-job/clean aborts the solve.

inline bool PcHexToBytes(const std::string& h, uint8_t* out, size_t n)
{
    if (h.size() != n * 2) return false;
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < n; ++i) {
        const int hi = nib(h[2 * i]), lo = nib(h[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

struct PcSolveState {
    std::thread th;
    std::atomic<bool> abort{false};
    std::atomic<uint64_t> shares_found{0};
    uint64_t job_id = 0;
    bool running = false;
    // LIVE share target, re-read by the solve loop every iteration rather than captured once.
    // A wrapper raises difficulty mid-job (vardiff) with "set_target"; when we ignored that and
    // kept grading against the job's original easier target, the pool rejected the resulting
    // submissions "below-target" -- 5 of 8 in the first live run after the target moved.
    std::mutex tgt_mu;
    uint256 share_target;
    void SetTarget(const uint256& t) { std::lock_guard<std::mutex> lk(tgt_mu); share_target = t; }
    uint256 Target() { std::lock_guard<std::mutex> lk(tgt_mu); return share_target; }
    ~PcSolveState() { stop(); }   // never destroy a joinable thread (terminate)
    void stop() {
        if (!running) return;
        abort.store(true);
        if (th.joinable()) th.join();
        running = false;
        abort.store(false);
    }
};

inline PcSolveState& SolveState() { static PcSolveState st; return st; }

// Parse the 80-byte classic header template (raw serialized byte order: LE ints, internal-
// order hashes). nonce fields + matmul_digest are the solver's to fill.
inline bool PcHeaderFromHex(const std::string& header_hex, CBlockHeader& b)
{
    uint8_t raw[80];
    if (!PcHexToBytes(header_hex, raw, sizeof(raw))) return false;
    uint32_t v, t, bits;
    std::memcpy(&v, raw + 0, 4);
    std::memcpy(&t, raw + 68, 4);
    std::memcpy(&bits, raw + 72, 4);
    b.nVersion = static_cast<int32_t>(v);
    std::memcpy(b.hashPrevBlock.data(), raw + 4, 32);
    std::memcpy(b.hashMerkleRoot.data(), raw + 36, 32);
    b.nTime = t;
    b.nBits = bits;
    b.nNonce = 0;
    b.nNonce64 = 0;
    b.matmul_digest.SetNull();
    return true;
}

inline void StartBtxLiveJob(uint64_t job_id, const CBlockHeader& tmpl, const uint256& share_target,
                            int32_t height, int64_t parent_mtp, uint16_t matmul_dim,
                            const std::string& gpu_uuid, const std::string& profile)
{
    PcSolveState& st = SolveState();
    st.stop();
    st.job_id = job_id;
    st.shares_found.store(0);
    st.SetTarget(share_target);   // live target; "set_target" updates it mid-job
    st.running = true;
    st.th = std::thread([job_id, tmpl, share_target, height, parent_mtp, matmul_dim, gpu_uuid, profile, &st] {
        const Consensus::Params& consensus = Params().GetConsensus();
        CBlockHeader b = tmpl;
        b.matmul_dim = matmul_dim != 0
                           ? matmul_dim
                           : static_cast<uint16_t>(consensus.nMatMulDimension);
        uint64_t nonce_cursor = 0;
        const std::function<bool(const CBlockHeader&)> sink = [&](const CBlockHeader& sh) {
            // Final check against the target as it stands RIGHT NOW. A solution found in the
            // window between a set_target arriving and the solver picking it up was graded
            // against the old target; submitting it just earns a "below-target" rejection. One
            // such share showed up per difficulty change in the live run. Drop it instead.
            if (UintToArith256(sh.matmul_digest) > UintToArith256(st.Target())) {
                if (std::getenv("BTX_POOLCORE_DEBUG")) {
                    std::fprintf(stderr, "[poolcore-dbg] stale share dropped (target moved) nonce=%llu\n",
                                 (unsigned long long)sh.nNonce64);
                    std::fflush(stderr);
                }
                return true;   // keep solving; this one simply missed the new bar
            }
            char nonce_hex[17];
            std::snprintf(nonce_hex, sizeof(nonce_hex), "%016llx",
                          static_cast<unsigned long long>(sh.nNonce64));
            std::ostringstream o;
            o << "{\"type\":\"solution\",\"job_id\":" << job_id
              << ",\"kind\":\"share\",\"gpu_uuid\":" << PcJsonStr(gpu_uuid)
              << ",\"nonce_hex\":\"" << nonce_hex << "\""
              << ",\"digest_hex\":" << PcJsonStr(sh.matmul_digest.GetHex())
              << ",\"ntime\":" << sh.nTime
              << ",\"profile\":" << PcJsonStr(profile)
              << ",\"ts_unix_ms\":" << PcNowMs() << "}";
            if (std::getenv("BTX_POOLCORE_DEBUG")) {
                std::fprintf(stderr, "[poolcore-dbg] SINK FIRED nonce=%llu digest=%s\n",
                             (unsigned long long)sh.nNonce64, sh.matmul_digest.GetHex().c_str());
                std::fflush(stderr);
            }
            Emit(o.str());
            st.shares_found.fetch_add(1);
            return true;   // keep mining on the same pipeline
        };
        const bool pcdbg = std::getenv("BTX_POOLCORE_DEBUG") != nullptr;
        if (pcdbg) {
            const bool rc_active = consensus.IsMatMulRCActive(height);
            auto blk_tgt = DeriveTarget(b.nBits, consensus.powLimit);
            std::fprintf(stderr, "[poolcore-dbg] solve start: matmul_dim=%u nBits=%08x height=%d "
                         "fMatMulPOW=%d rc_active=%d rc_height=%d deriveTarget_ok=%d blk_target=%s share_target=%s\n",
                         (unsigned)b.matmul_dim, (unsigned)b.nBits, height,
                         consensus.fMatMulPOW ? 1 : 0, rc_active ? 1 : 0,
                         (int)Consensus::Params::RCActivationHeight(), blk_tgt.has_value() ? 1 : 0,
                         blk_tgt.has_value() ? blk_tgt->GetHex().c_str() : "none",
                         share_target.GetHex().c_str());
            std::fflush(stderr);
        }
        uint64_t iters = 0;
        while (!st.abort.load()) {
            b.nNonce64 = nonce_cursor;
            b.nNonce = static_cast<uint32_t>(nonce_cursor);
            // Large window; preemption is the abort flag (checked inside the solver), not
            // tries exhaustion. A small window wastes time in per-call pipeline setup.
            uint64_t tries = 1'000'000'000'000ULL;
            // Re-read every iteration: a mid-job "set_target" must take effect here, not at the
            // next job, or we keep grading against a stale target and the pool rejects the result.
            const uint256 live_target = st.Target();
            const bool solved = SolveMatMul(b, consensus, tries, height, &st.abort,
                                            /*share_target_override=*/&live_target,
                                            std::optional<int64_t>(parent_mtp), &sink);
            ++iters;
            if (pcdbg && (iters <= 4 || solved)) {
                std::fprintf(stderr, "[poolcore-dbg] iter=%llu solved=%d end_nonce=%llu digest=%s\n",
                             (unsigned long long)iters, solved ? 1 : 0,
                             (unsigned long long)b.nNonce64, b.matmul_digest.GetHex().c_str());
                std::fflush(stderr);
            }
            if (solved) {
                // Block-target hit outside the sink path: emit it too, then continue.
                sink(b);
            }
            nonce_cursor = b.nNonce64 + 1;
        }
        if (pcdbg) {
            std::fprintf(stderr, "[poolcore-dbg] solve exit: iters=%llu shares=%llu\n",
                         (unsigned long long)iters, (unsigned long long)st.shares_found.load());
            std::fflush(stderr);
        }
    });
}

// ---- enc-rc-v46 / enc-rc-v47 (ENC_RC episode) dispatch -------------------------------------
// Routes an RC-profile job to SolveMatMulRCEpisode (the byte-exact episode digest solve loop).
// On a winner it emits BOTH a {kind:block_winner} solution AND the {type:witness} the pool's
// prover needs (header/nonce/digest/round_roots), per poolcore-witness-v0.md. Episode shape
// follows the ACTIVE launch profile (PR#97 v4.7 Epoch A = PROFILE 1; BTX_MATMUL_RC_PROFILE=2
// selects the Epoch D datacenter shape); BTX_RC_EPISODE_SHAPE=toy runs the fast toy dims for
// wiring/witness tests. DEFAULT ON since v0.9.2 -- the pool's enc-rc profile string is the
// activation signal; the caller only skips dispatch on BTX_RC_ENABLE_EPISODE_SOLVE=0. The
// received profile string is echoed back in every emission so the pool's label (v46 legacy or
// v47) round-trips unchanged.
inline void StartRCEpisodeJob(uint64_t job_id, const CBlockHeader& tmpl, const std::string& header_hex,
                              const uint256& share_target, int32_t height, int64_t parent_mtp,
                              const std::string& gpu_uuid, const std::string& profile,
                              uint64_t nonce64_start = 0)
{
    PcSolveState& st = SolveState();
    st.stop();
    st.job_id = job_id;
    st.shares_found.store(0);
    st.SetTarget(share_target);   // live target; "set_target" updates it mid-job
    st.running = true;
    st.th = std::thread([job_id, tmpl, header_hex, share_target, height, parent_mtp, gpu_uuid,
                         profile, nonce64_start, &st] {
        const Consensus::Params& consensus = Params().GetConsensus();
        matmul::v4::rc::EpisodeParams ep = matmul::v4::rc::ActiveProfileEpisodeParams();
        if (const char* s = std::getenv("BTX_RC_EPISODE_SHAPE"); s && std::string(s) == "toy")
            ep = matmul::v4::rc::EpisodeParams{};   // toy dims (fast; the golden 5b1bff3c shape)
        const bool pcdbg = std::getenv("BTX_POOLCORE_DEBUG") != nullptr;
        CBlockHeader b = tmpl;
        // Start where the WRAPPER told us to. The pool shards the nonce space across workers, so
        // defaulting every instance to 0 makes multi-GPU / multi-instance rigs grind identical
        // nonces. 0 keeps the previous behaviour for a wrapper that does not send the field.
        uint64_t cursor = nonce64_start;
        while (!st.abort.load()) {
            b.nNonce64 = cursor;
            b.nNonce = static_cast<uint32_t>(cursor);
            uint64_t tries = 1'000'000ULL;   // window per call; abort checked inside the solver
            // Live target, re-read per window so a mid-job set_target applies immediately.
            const arith_uint256 tgt = UintToArith256(st.Target());
            const bool won = SolveMatMulRCEpisode(b, consensus, tries, height, &st.abort,
                                                  std::optional<int64_t>(parent_mtp), tgt, ep);
            if (won) {
                // Re-check against the target as it stands now: this episode was graded against
                // the value read at the top of the window, and a set_target may have landed
                // since. Submitting the stale winner only earns a "below-target" rejection.
                if (UintToArith256(b.matmul_digest) > UintToArith256(st.Target())) {
                    if (pcdbg) {
                        std::fprintf(stderr, "[poolcore-dbg] stale RC share dropped (target moved) nonce=%llu\n",
                                     (unsigned long long)b.nNonce64);
                        std::fflush(stderr);
                    }
                    cursor = b.nNonce64 + 1;
                    continue;
                }
                char nonce_hex[17];
                std::snprintf(nonce_hex, sizeof(nonce_hex), "%016llx",
                              static_cast<unsigned long long>(b.nNonce64));
                const std::string digest_hex = b.matmul_digest.GetHex();
                // Classify: a hit is a BLOCK CANDIDATE iff the digest also meets the header's
                // own nBits-derived block target (inclusive, same comparison consensus runs);
                // otherwise it is an ordinary pool share. The pre-hash gate is retired at v4
                // heights, so digest-vs-target is the whole lottery -- no other check exists.
                bool is_block = false;
                if (auto blk = DeriveTarget(b.nBits, consensus.powLimit))
                    is_block = UintToArith256(b.matmul_digest) <= *blk;
                {
                    std::ostringstream o;
                    o << "{\"type\":\"solution\",\"job_id\":" << job_id
                      << ",\"kind\":\"" << (is_block ? "block_winner" : "share")
                      << "\",\"gpu_uuid\":" << PcJsonStr(gpu_uuid)
                      << ",\"nonce_hex\":\"" << nonce_hex << "\""
                      << ",\"digest_hex\":" << PcJsonStr(digest_hex)
                      << ",\"profile\":" << PcJsonStr(profile)
                      << ",\"ts_unix_ms\":" << PcNowMs() << "}";
                    Emit(o.str());
                }
                // witness (enc-rc-v0): header/nonce/digest + round_roots (round 0..R-1), BLOCK
                // WINNERS ONLY (it costs one extra episode to materialize; shares are cheap by
                // contract). Emitted on the line right after the solution, same
                // (job_id, nonce_hex, digest_hex) triple.
                if (is_block) {
                    const uint256 sigma = matmul::DeriveSigma(b);
                    // Roots via the GPU episode when available: the CPU reference is HOURS at
                    // Profile-1 dims and would wedge the solver on every winner (caught live in
                    // the 2026-08-04 byron demo). Same guard as the solve backend selection.
                    std::vector<uint256> roots;
#ifdef MATADOR_ENABLE_CUDA
                    if (std::getenv("BTX_RC_EPISODE_CPU") == nullptr &&
                        matmul::v4::rc::RCEpisodeGpuAvailable())
                        roots = matmul::v4::rc::ComputeEpisodeRoundRootsGPU(sigma, ep);
                    else
#endif
                        roots = matmul::v4::rc::ComputeEpisodeRoundRoots(sigma, ep);
                    std::ostringstream o;
                    o << "{\"type\":\"witness\",\"witness_version\":\"enc-rc-v0\",\"job_id\":" << job_id
                      << ",\"header_hex\":" << PcJsonStr(header_hex)
                      << ",\"nonce_hex\":\"" << nonce_hex << "\""
                      << ",\"digest_hex\":" << PcJsonStr(digest_hex)
                      << ",\"round_roots\":[";
                    for (size_t i = 0; i < roots.size(); ++i)
                        o << (i ? "," : "") << PcJsonStr(roots[i].GetHex());
                    o << "],\"profile\":" << PcJsonStr(profile)
                      << ",\"derivation\":\"v0\",\"ts_unix_ms\":" << PcNowMs() << "}";
                    Emit(o.str());
                }
                st.shares_found.fetch_add(1);
                if (pcdbg) {
                    std::fprintf(stderr, "[poolcore-dbg] RC winner nonce=%llu digest=%s\n",
                                 (unsigned long long)b.nNonce64, digest_hex.c_str());
                    std::fflush(stderr);
                }
            }
            cursor = b.nNonce64 + 1;   // SolveMatMulRCEpisode advances nNonce64 past its window
        }
    });
}

// ---- enc-rc-v46-coupled (ENC_RC coupled V3) dispatch -------------------------------------
// Routes a coupled-profile job to SolveMatMulRCCoupled (the byte-exact coupled V3 digest solve
// loop). Coupled binds an episode leg to the ~48 GiB bank puzzle; here we grind the coupled digest
// against the share target and, on a winner, emit BOTH a {kind:block_winner} solution AND a coupled
// witness (header/nonce/digest + the nonce-independent bank_template_hash the pool re-derives).
// Production V3 dims by default; BTX_RC_COUPLED_SHAPE=toy runs the fast Medium-V3 dims (golden
// a4bb0cc4 shape) for wiring/witness tests. GATED by the caller on BTX_RC_ENABLE_COUPLED_SOLVE.
inline void StartRCCoupledJob(uint64_t job_id, const CBlockHeader& tmpl, const std::string& header_hex,
                              const uint256& share_target, int32_t height, int64_t parent_mtp,
                              const std::string& gpu_uuid)
{
    PcSolveState& st = SolveState();
    st.stop();
    st.job_id = job_id;
    st.shares_found.store(0);
    st.SetTarget(share_target);   // live target; "set_target" updates it mid-job
    st.running = true;
    st.th = std::thread([job_id, tmpl, header_hex, share_target, height, parent_mtp, gpu_uuid, &st] {
        const Consensus::Params& consensus = Params().GetConsensus();
        matmul::v4::rc::CoupParamsV3 cp = matmul::v4::rc::ProductionV3CoupParams();
        if (const char* s = std::getenv("BTX_RC_COUPLED_SHAPE"); s && std::string(s) == "toy")
            cp = matmul::v4::rc::MediumV3CoupParams();   // fast dims (the golden a4bb0cc4 shape)
        const bool pcdbg = std::getenv("BTX_POOLCORE_DEBUG") != nullptr;
        const arith_uint256 tgt = UintToArith256(share_target);
        CBlockHeader b = tmpl;
        uint64_t cursor = 0;
        while (!st.abort.load()) {
            b.nNonce64 = cursor;
            b.nNonce = static_cast<uint32_t>(cursor);
            uint64_t tries = 1'000'000ULL;   // window per call; abort checked inside the solver
            const bool won = SolveMatMulRCCoupled(b, consensus, tries, height, &st.abort,
                                                  std::optional<int64_t>(parent_mtp), tgt, cp);
            if (won) {
                // Re-check against the target as it stands now: this episode was graded against
                // the value read at the top of the window, and a set_target may have landed
                // since. Submitting the stale winner only earns a "below-target" rejection.
                if (UintToArith256(b.matmul_digest) > UintToArith256(st.Target())) {
                    if (pcdbg) {
                        std::fprintf(stderr, "[poolcore-dbg] stale RC share dropped (target moved) nonce=%llu\n",
                                     (unsigned long long)b.nNonce64);
                        std::fflush(stderr);
                    }
                    cursor = b.nNonce64 + 1;
                    continue;
                }
                char nonce_hex[17];
                std::snprintf(nonce_hex, sizeof(nonce_hex), "%016llx",
                              static_cast<unsigned long long>(b.nNonce64));
                const std::string digest_hex = b.matmul_digest.GetHex();
                // Same share-vs-block-candidate classification as the episode path.
                bool is_block = false;
                if (auto blk = DeriveTarget(b.nBits, consensus.powLimit))
                    is_block = UintToArith256(b.matmul_digest) <= *blk;
                {
                    std::ostringstream o;
                    o << "{\"type\":\"solution\",\"job_id\":" << job_id
                      << ",\"kind\":\"" << (is_block ? "block_winner" : "share")
                      << "\",\"gpu_uuid\":" << PcJsonStr(gpu_uuid)
                      << ",\"nonce_hex\":\"" << nonce_hex << "\""
                      << ",\"digest_hex\":" << PcJsonStr(digest_hex)
                      << ",\"profile\":\"enc-rc-v46-coupled\",\"ts_unix_ms\":" << PcNowMs() << "}";
                    Emit(o.str());
                }
                // witness (enc-rc-coupled-v0): header/nonce/digest + the nonce-independent
                // bank_template_hash (the pool re-derives the ~48 GiB bank from it). Same
                // (job_id, nonce_hex, digest_hex) triple as the solution line. Winner-only.
                if (is_block) {
                    const uint256 bank_tmpl = matmul::v4::rc::RCBankTemplateHash(b);
                    std::ostringstream o;
                    o << "{\"type\":\"witness\",\"witness_version\":\"enc-rc-coupled-v0\",\"job_id\":"
                      << job_id
                      << ",\"header_hex\":" << PcJsonStr(header_hex)
                      << ",\"nonce_hex\":\"" << nonce_hex << "\""
                      << ",\"digest_hex\":" << PcJsonStr(digest_hex)
                      << ",\"bank_template_hash\":" << PcJsonStr(bank_tmpl.GetHex())
                      << ",\"profile\":\"enc-rc-v46-coupled\",\"derivation\":\"v0\",\"ts_unix_ms\":"
                      << PcNowMs() << "}";
                    Emit(o.str());
                }
                st.shares_found.fetch_add(1);
                if (pcdbg) {
                    std::fprintf(stderr, "[poolcore-dbg] coupled winner nonce=%llu digest=%s\n",
                                 (unsigned long long)b.nNonce64, digest_hex.c_str());
                    std::fflush(stderr);
                }
            }
            cursor = b.nNonce64 + 1;   // SolveMatMulRCCoupled advances nNonce64 past its window
        }
    });
}

// ---- the loop ----------------------------------------------------------------------------

template <typename ConfigT>
int RunPoolcoreLoop(const ConfigT& /*cfg*/)
{
    // Hello first so the wrapper can gate on protocol/schema before sending anything.
    Emit(std::string("{\"type\":\"hello\",\"protocol\":") + PcJsonStr(kProtocol) +
         ",\"schema_version\":" + std::to_string(kSchemaVersion) +
         ",\"solver_version\":" + PcJsonStr(MATADOR_MINER_VERSION) +
         ",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");

    // Query devices UP FRONT, not only on "init". gpu_uuid rides on every solution as the
    // wrapper's DeviceBinding join key, but "init" is an optional handshake step -- the minebtx
    // endpoint goes straight to "job" -- and when it is skipped this stayed empty and every
    // single solution went out tagged gpu_uuid "unknown". A field we attach unconditionally must
    // not depend on a message the wrapper is free not to send.
    std::vector<PcDevice> devices = QueryDevices();
    uint64_t current_job = 0;
    bool have_job = false;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::string type;
        if (!PcGetString(line, "type", type)) {
            EmitError("bad_message", "missing type field");
            continue;
        }

        if (type == "init") {
            devices = QueryDevices();
            std::ostringstream o;
            o << "{\"type\":\"init_ok\",\"schema_version\":" << kSchemaVersion
              << ",\"solver_version\":" << PcJsonStr(MATADOR_MINER_VERSION) << ",\"devices\":[";
            for (size_t i = 0; i < devices.size(); ++i)
                o << (i ? "," : "") << DeviceJson(devices[i]);
            o << "],\"ts_unix_ms\":" << PcNowMs() << "}";
            Emit(o.str());
            if (devices.empty())
                EmitError("no_devices", "nvidia-smi identity query returned no rows "
                                        "(non-NVIDIA backends not wired in scaffold)");
        } else if (type == "job") {
            uint64_t jid = 0;
            std::string profile;
            if (!PcGetU64(line, "job_id", jid) || !PcGetString(line, "profile", profile)) {
                EmitError("bad_job", "job requires job_id (u64) and profile (string)");
                continue;
            }
            current_job = jid;
            have_job = true;
            if (profile == "btx-live") {
                std::string header_hex, target_hex;
                uint64_t height = 0, pmtp = 0;
                CBlockHeader tmpl;
                if (!PcGetString(line, "header_hex", header_hex) ||
                    !PcGetString(line, "target_hex", target_hex) ||
                    !PcGetU64(line, "height", height) ||
                    !PcGetU64(line, "parent_mtp", pmtp)) {
                    EmitError("bad_job", "btx-live v0 needs header_hex, target_hex, height, parent_mtp");
                    continue;
                }
                if (!PcHeaderFromHex(header_hex, tmpl)) {
                    EmitError("bad_job", "header_hex must be exactly 160 hex chars (80 bytes)");
                    continue;
                }
                const auto tgt = uint256::FromHex(target_hex);
                if (!tgt.has_value()) {
                    EmitError("bad_job", "target_hex must be 64 hex chars (32 bytes, BE display)");
                    continue;
                }
                uint64_t matmul_n = 0;
                PcGetU64(line, "matmul_n", matmul_n);   // optional; 0 -> consensus default for height
                const std::string uuid = devices.empty() ? std::string("unknown") : devices[0].gpu_uuid;
                StartBtxLiveJob(jid, tmpl, tgt.value(), static_cast<int32_t>(height),
                                static_cast<int64_t>(pmtp), static_cast<uint16_t>(matmul_n), uuid, profile);
                Emit(std::string("{\"type\":\"job_ack\",\"job_id\":") + std::to_string(jid) +
                     ",\"profile\":" + PcJsonStr(profile) +
                     ",\"note\":\"dispatching\",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
            } else if (profile == "enc-rc-v46" || profile == "enc-rc-v47") {
                // v47 = the upstream v4.7 relabel of v4.6-v3 (PR#97) -- same job fields, same
                // solver, same digest bytes; the label is echoed back verbatim in emissions.
                std::string header_hex, target_hex;
                uint64_t height = 0, pmtp = 0;
                CBlockHeader tmpl;
                if (!PcGetString(line, "header_hex", header_hex) ||
                    !PcGetString(line, "target_hex", target_hex) ||
                    !PcGetU64(line, "height", height) ||
                    !PcGetU64(line, "parent_mtp", pmtp)) {
                    EmitError("bad_job", "enc-rc v0 needs header_hex, target_hex, height, parent_mtp");
                    continue;
                }
                if (!PcHeaderFromHex(header_hex, tmpl)) {
                    EmitError("bad_job", "header_hex must be exactly 160 hex chars (80 bytes)");
                    continue;
                }
                const auto tgt = uint256::FromHex(target_hex);
                if (!tgt.has_value()) {
                    EmitError("bad_job", "target_hex must be 64 hex chars (32 bytes, BE display)");
                    continue;
                }
                // DEFAULT ON since v0.9.2: a pool sending profile enc-rc-v46/v47 IS the deliberate
                // activation signal (height-agnostic -- the activation height moved three times and
                // is never compiled in), and Epoch A (Profile 1 + ExactReplay) needs no succinct
                // carrier: digest <= target is the whole proof. BTX_RC_ENABLE_EPISODE_SOLVE=0
                // force-disables (pre-0.9.2 fail-closed posture). Loud, never silently idle.
                if (const char* g = std::getenv("BTX_RC_ENABLE_EPISODE_SOLVE");
                    g != nullptr && g[0] == '0' && g[1] == '\0') {
                    Emit(std::string("{\"type\":\"job_ack\",\"job_id\":") + std::to_string(jid) +
                         ",\"profile\":" + PcJsonStr(profile) +
                         ",\"note\":\"rc_solve_disabled (BTX_RC_ENABLE_EPISODE_SOLVE=0; unset to enable)\""
                         ",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
                    continue;
                }
                // Header matmul_dim MUST be stamped before seed/sigma derivation: consensus at RC
                // heights rejects headers whose matmul_dim != nMatMulV4Dimension, and the field is
                // in both the seed-V3 and sigma preimages. Optional job field "matmul_n" (pool's
                // template value) wins; absent/0 falls back to the consensus RC header dim (4096
                // mainnet). Grinding with the pre-fix dim=0 produced sigmas no node can accept.
                uint64_t rc_matmul_n = 0;
                PcGetU64(line, "matmul_n", rc_matmul_n);
                tmpl.matmul_dim = rc_matmul_n != 0
                                      ? static_cast<uint16_t>(rc_matmul_n)
                                      : matmul::v4::rc::RCConsensusHeaderMatmulDim();
                // Optional "nonce64_start": where THIS instance begins its scan. Pools shard the
                // nonce space across workers (extranonce-style); without it every instance starts
                // at 0 and multi-GPU / multi-instance rigs duplicate each other's work outright.
                // Absent/0 keeps the old behaviour, so it is backward compatible with a wrapper
                // that does not send it.
                uint64_t rc_nonce_start = 0;
                PcGetU64(line, "nonce64_start", rc_nonce_start);
                const std::string uuid = devices.empty() ? std::string("unknown") : devices[0].gpu_uuid;
                StartRCEpisodeJob(jid, tmpl, header_hex, tgt.value(), static_cast<int32_t>(height),
                                  static_cast<int64_t>(pmtp), uuid, profile, rc_nonce_start);
                Emit(std::string("{\"type\":\"job_ack\",\"job_id\":") + std::to_string(jid) +
                     ",\"profile\":" + PcJsonStr(profile) +
                     ",\"note\":\"dispatching-rc\",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
            } else if (profile == "enc-rc-v46-coupled") {
                std::string header_hex, target_hex;
                uint64_t height = 0, pmtp = 0;
                CBlockHeader tmpl;
                if (!PcGetString(line, "header_hex", header_hex) ||
                    !PcGetString(line, "target_hex", target_hex) ||
                    !PcGetU64(line, "height", height) ||
                    !PcGetU64(line, "parent_mtp", pmtp)) {
                    EmitError("bad_job",
                              "enc-rc-v46-coupled v0 needs header_hex, target_hex, height, parent_mtp");
                    continue;
                }
                if (!PcHeaderFromHex(header_hex, tmpl)) {
                    EmitError("bad_job", "header_hex must be exactly 160 hex chars (80 bytes)");
                    continue;
                }
                const auto tgt = uint256::FromHex(target_hex);
                if (!tgt.has_value()) {
                    EmitError("bad_job", "target_hex must be 64 hex chars (32 bytes, BE display)");
                    continue;
                }
                // Fail-closed like enc-rc-v46: a mined coupled digest still needs the carrier/pool RC
                // format on a public net, so do not grind the ~48 GiB bank puzzle unless deliberately
                // enabled. Loud, never silently idle.
                if (std::getenv("BTX_RC_ENABLE_COUPLED_SOLVE") == nullptr) {
                    Emit(std::string("{\"type\":\"job_ack\",\"job_id\":") + std::to_string(jid) +
                         ",\"profile\":" + PcJsonStr(profile) +
                         ",\"note\":\"coupled_solve_disabled (set BTX_RC_ENABLE_COUPLED_SOLVE=1)\""
                         ",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
                    continue;
                }
                // Same matmul_dim stamping contract as the episode path (seed/sigma preimage).
                uint64_t coup_matmul_n = 0;
                PcGetU64(line, "matmul_n", coup_matmul_n);
                tmpl.matmul_dim = coup_matmul_n != 0
                                      ? static_cast<uint16_t>(coup_matmul_n)
                                      : matmul::v4::rc::RCConsensusHeaderMatmulDim();
                const std::string uuid = devices.empty() ? std::string("unknown") : devices[0].gpu_uuid;
                StartRCCoupledJob(jid, tmpl, header_hex, tgt.value(), static_cast<int32_t>(height),
                                  static_cast<int64_t>(pmtp), uuid);
                Emit(std::string("{\"type\":\"job_ack\",\"job_id\":") + std::to_string(jid) +
                     ",\"profile\":" + PcJsonStr(profile) +
                     ",\"note\":\"dispatching-coupled\",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
            } else {
                Emit(std::string("{\"type\":\"job_ack\",\"job_id\":") + std::to_string(jid) +
                     ",\"profile\":" + PcJsonStr(profile) +
                     ",\"note\":\"profile_not_wired\",\"ts_unix_ms\":" +
                     std::to_string(PcNowMs()) + "}");
            }
        } else if (type == "stats") {
            std::ostringstream o;
            o << "{\"type\":\"stats\",\"schema_version\":" << kSchemaVersion << ",\"devices\":[";
            for (size_t i = 0; i < devices.size(); ++i) {
                const PcDevice& d = devices[i];
                o << (i ? "," : "")
                  << "{\"gpu_uuid\":" << PcJsonStr(d.gpu_uuid)
                  << ",\"pci_bus_id\":" << PcJsonStr(d.pci_bus_id)
                  << ",\"solver_index\":" << d.solver_index
                  << ",\"nps\":0.0"   // 10s rolling mean once dispatch is wired
                  << ",\"backend\":" << PcJsonStr(d.backend)
                  << ",\"solver_version\":" << PcJsonStr(MATADOR_MINER_VERSION)
                  << ",\"shares_found\":" << SolveState().shares_found.load()
                  << ",\"solving\":" << (SolveState().running ? "true" : "false")
                  << ",\"errors\":[]}";
            }
            o << "],\"ts_unix_ms\":" << PcNowMs() << "}";
            Emit(o.str());
        } else if (type == "preempt") {
            uint64_t jid = 0;
            PcGetU64(line, "job_id", jid);
            if (have_job && jid == current_job) have_job = false;
            SolveState().stop();
            Emit(std::string("{\"type\":\"preempt_ok\",\"job_id\":") + std::to_string(jid) +
                 ",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
        } else if (type == "shutdown") {
            SolveState().stop();
            Emit(std::string("{\"type\":\"shutdown_ok\",\"ts_unix_ms\":") +
                 std::to_string(PcNowMs()) + "}");
            return 0;
        } else if (type == "set_target") {
            // Mid-job difficulty change (vardiff). Applies to the RUNNING job: the solve loops
            // re-read the live target every iteration, so this takes effect immediately rather
            // than at the next job. Ignoring it cost 5 of 8 submissions "below-target" against
            // the minebtx endpoint -- we kept grading against the job's original easier target.
            std::string t_hex;
            if (!PcGetString(line, "target_hex", t_hex)) {
                EmitError("bad_message", "set_target without target_hex");
                continue;
            }
            const auto t = uint256::FromHex(t_hex);
            if (!t.has_value()) {
                EmitError("bad_message", "set_target target_hex not 32-byte hex");
                continue;
            }
            SolveState().SetTarget(t.value());
            Emit(std::string("{\"type\":\"set_target_ok\",\"target_hex\":") + PcJsonStr(t_hex) +
                 ",\"ts_unix_ms\":" + std::to_string(PcNowMs()) + "}");
        } else if (type == "result") {
            // Per-solution ack from the wrapper (accepted / rejected / stale). Purely
            // informational for us: the wrapper owns submission and grading, we own solving.
            // Do NOT answer it -- replying {"type":"error","what":"unknown_type"} to a routine
            // ack, as we did against the minebtx endpoint, makes every graded share look like a
            // protocol fault on their side. Surfaced in the log only.
            if (std::getenv("BTX_POOLCORE_DEBUG") != nullptr) {
                std::string status, reason;
                PcGetString(line, "status", status);
                PcGetString(line, "reason", reason);
                std::fprintf(stderr, "[poolcore-dbg] result status=%s reason=%s\n",
                             status.empty() ? "?" : status.c_str(),
                             reason.empty() ? "-" : reason.c_str());
            }
        } else {
            EmitError("unknown_type", type);
        }
    }
    // stdin closed = wrapper died; stop any in-flight solve, then exit cleanly
    // (a joinable thread at static destruction is std::terminate).
    SolveState().stop();
    return 0;
}

}  // namespace poolcore
