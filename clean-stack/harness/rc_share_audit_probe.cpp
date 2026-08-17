// rc_share_audit_probe -- offline ENC_RC share auditor: the pool audit-farm tool for the
// Epoch A (v4.7 Profile 1, ExactReplay) sampled-replay regime.
//
// A pool admits shares cheaply (digest vs share target) and replays a SAMPLE to catch
// fabricated digests. This is that replay, standalone: feed it JSONL share records (the same
// fields the poolcore job/solution flow carries) and it recomputes the consensus episode
// digest from scratch -- deterministic seeds, sigma, full episode -- and reports whether the
// claimed digest is honest and whether it meets the share target. GPU backend when a CUDA
// device is present (sub-second per share at Epoch A dims on a 5090), CPU oracle otherwise.
//
// stdin, one JSON object per line:
//   {"header_hex":"<160 hex>","nonce_hex":"<16 hex>","height":N,"parent_mtp":M,
//    "share_target_hex":"<64 hex BE>","digest_hex":"<64 hex, optional claimed digest>"}
// stdout, one verdict per line:
//   {"digest_hex":...,"meets_target":0|1,"matches_claim":0|1|null,"elapsed_ms":...}
// exit code: 0 if every line parsed and every claimed digest matched; 1 otherwise.
//
// --selftest: audits one self-mined toy-dims share end-to-end (solve -> audit -> verdict 1/1)
// so a pool can prove the exact binary it deploys is self-consistent, without source.
//
// Profile: ActiveProfileEpisodeParams() -- Profile 1 by default (the Epoch A launch tuple);
// BTX_MATMUL_RC_PROFILE=2 audits the Epoch D datacenter shape; BTX_RC_EPISODE_SHAPE=toy for
// wiring tests. Same knobs as the solver, so auditor and miner can never disagree on shape.

#include <pow.h>
#include <primitives/block.h>
#include <consensus/params.h>
#include <matmul/matmul_pow.h>
#include <matmul/matmul_v4_rc.h>
#include <uint256.h>
#include <arith_uint256.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>

// ---- tiny single-line JSON field extractors (byte-identical semantics to poolcore's) -------
static bool GetStr(const std::string& line, const std::string& key, std::string& out)
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
static bool GetU64(const std::string& line, const std::string& key, uint64_t& out)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) return false;
    p = line.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < line.size() && line[p] == ' ') ++p;
    uint64_t v = 0; bool any = false;
    while (p < line.size() && line[p] >= '0' && line[p] <= '9') {
        v = v * 10 + uint64_t(line[p] - '0'); ++p; any = true;
    }
    if (!any) return false;
    out = v; return true;
}
static bool HexToBytes(const std::string& hex, uint8_t* out, size_t n)
{
    if (hex.size() != n * 2) return false;
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        c = char(c | 32);
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < n; ++i) {
        const int hi = nib(hex[i*2]), lo = nib(hex[i*2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = uint8_t((hi << 4) | lo);
    }
    return true;
}
// 80-byte classic header prefix -> CBlockHeader (same field map as poolcore's PcHeaderFromHex)
static bool HeaderFromHex(const std::string& header_hex, CBlockHeader& b)
{
    uint8_t raw[80];
    if (!HexToBytes(header_hex, raw, sizeof(raw))) return false;
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

static uint256 AuditDigest(CBlockHeader& b, const Consensus::Params& params, int32_t height,
                           int64_t parent_mtp, const matmul::v4::rc::EpisodeParams& ep,
                           bool use_gpu)
{
    if (!SetDeterministicMatMulSeeds(b, params, height, std::optional<int64_t>(parent_mtp))) {
        return uint256{};
    }
    const uint256 sigma = matmul::DeriveSigma(b);
    // BTX_RC_AUDIT_DEBUG=1: emit the derivation intermediates (stderr, so the stdout JSONL
    // contract is untouched). For cross-implementation freeze checks: if two implementations
    // disagree on the final digest, these localize WHERE. GetHex is the usual reversed display.
    if (std::getenv("BTX_RC_AUDIT_DEBUG") != nullptr) {
        fprintf(stderr,
                "{\"seed_a\":\"%s\",\"seed_b\":\"%s\",\"sigma\":\"%s\","
                "\"matmul_dim\":%u,\"nNonce64\":%llu,\"nTime\":%u,\"nBits\":\"%08x\"}\n",
                b.seed_a.GetHex().c_str(), b.seed_b.GetHex().c_str(), sigma.GetHex().c_str(),
                unsigned(b.matmul_dim), (unsigned long long)b.nNonce64, unsigned(b.nTime),
                unsigned(b.nBits));
    }
#ifdef MATADOR_ENABLE_CUDA
    if (use_gpu) return matmul::v4::rc::ComputeEpisodeDigestGPU(sigma, ep);
#else
    (void)use_gpu;
#endif
    return matmul::v4::rc::ComputeEpisodeDigest(sigma, ep);
}

int main(int argc, char** argv)
{
    Consensus::Params params{};
    params.powLimit = uint256::FromHex(
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff").value();
    // Mainnet seed-derivation schedule (sealed upstream tip 909aa703). Without these a
    // default-constructed Params derives LEGACY V1 seeds (prevhash-only, nonce-independent)
    // at every height, which is NOT what consensus runs at RC heights: mainnet has used
    // nonce-bound seeds since 125,000 and parent-MTP-bound V3 seeds since 130,500, and the
    // RC activation height (181,894) is above both. The audit is time-correct: a record's
    // height selects the derivation the chain actually used at that height.
    params.fMatMulPOW = true;
    params.nMatMulNonceSeedHeight = 125'000;
    params.nMatMulParentMtpSeedHeight = 130'500;

    matmul::v4::rc::EpisodeParams ep = matmul::v4::rc::ActiveProfileEpisodeParams();
    if (const char* s = std::getenv("BTX_RC_EPISODE_SHAPE"); s && std::string(s) == "toy")
        ep = matmul::v4::rc::EpisodeParams{};

    bool use_gpu = false;
#ifdef MATADOR_ENABLE_CUDA
    use_gpu = std::getenv("BTX_RC_EPISODE_CPU") == nullptr && matmul::v4::rc::RCEpisodeGpuAvailable();
#endif

    // BTX_RC_SIGMA_HEX=<64hex>: run the episode DIRECTLY on a supplied sigma and print the
    // digest, skipping header parsing and seed derivation entirely. This is the cross-
    // implementation freeze check: sigma -> episode digest is the consensus core, while the
    // header->seed plumbing is transport that differs between pool dialects. Hex is the usual
    // reversed (GetHex) display order.
    if (const char* sh = std::getenv("BTX_RC_SIGMA_HEX")) {
        uint8_t raw[32];
        if (!HexToBytes(std::string(sh), raw, sizeof(raw))) {
            std::printf("{\"error\":\"bad_sigma_hex\"}\n");
            return 2;
        }
        uint256 sigma;
        for (int i = 0; i < 32; ++i) sigma.data()[i] = raw[31 - i];
        const uint256 d =
#ifdef MATADOR_ENABLE_CUDA
            use_gpu ? matmul::v4::rc::ComputeEpisodeDigestGPU(sigma, ep) :
#endif
                      matmul::v4::rc::ComputeEpisodeDigest(sigma, ep);
        std::printf("{\"sigma\":\"%s\",\"digest_hex\":\"%s\",\"backend\":\"%s\"}\n",
                    sigma.GetHex().c_str(), d.GetHex().c_str(), use_gpu ? "gpu" : "cpu");
        return 0;
    }

    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        // Mine one toy share, audit it through the exact production code path, expect 1/1.
        matmul::v4::rc::EpisodeParams toy{};
        CBlockHeader b{};
        b.nVersion = 0x20000004;
        b.hashPrevBlock = uint256::FromHex(
            "5151515151515151515151515151515151515151515151515151515151515151").value();
        b.hashMerkleRoot = uint256::FromHex(
            "a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3").value();
        b.nTime = 1770000000u; b.nBits = 0x207fffffu;
        // RC-representative: mainnet activation height (V3 MTP-bound seeds) + the consensus
        // header dim, so the selftest exercises the exact production derivation path.
        b.matmul_dim = matmul::v4::rc::RCConsensusHeaderMatmulDim();
        const int32_t h = 181'894; const int64_t mtp = 1769999000;
        uint64_t tries = 64;
        const arith_uint256 easy = ~arith_uint256(0);
        const bool won = SolveMatMulRCEpisode(b, params, tries, h, nullptr,
                                              std::optional<int64_t>(mtp),
                                              easy, toy);
        CBlockHeader audit = b;   // re-audit from scratch: seeds re-derived, digest recomputed
        const uint256 rec = AuditDigest(audit, params, h, mtp, toy, use_gpu);
        const bool ok = won && !rec.IsNull() && rec == b.matmul_digest;
        std::printf("rc_share_audit_probe selftest: mined=%d recompute==claim=%d (%s)\n",
                    won, ok, rec.GetHex().substr(0, 16).c_str());
        return ok ? 0 : 1;
    }

    std::fprintf(stderr, "[rc-share-audit] backend=%s profile: rounds=%u L=%u b_seq=%u\n",
                 use_gpu ? "gpu" : "cpu", ep.rounds, ep.L_lyr, ep.b_seq);

    int failures = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::string header_hex, nonce_hex, target_hex, claim_hex;
        uint64_t height = 0, pmtp = 0;
        CBlockHeader b;
        uint8_t nb[8];
        if (!GetStr(line, "header_hex", header_hex) || !GetStr(line, "nonce_hex", nonce_hex) ||
            !GetStr(line, "share_target_hex", target_hex) || !GetU64(line, "height", height) ||
            !GetU64(line, "parent_mtp", pmtp) || !HeaderFromHex(header_hex, b) ||
            !HexToBytes(nonce_hex, nb, 8)) {
            std::printf("{\"error\":\"bad_record\"}\n");
            ++failures;
            continue;
        }
        uint64_t nonce = 0;                      // nonce_hex is %016llx (BE hex of the u64)
        for (int i = 0; i < 8; ++i) nonce = (nonce << 8) | nb[i];
        b.nNonce64 = nonce;
        b.nNonce = static_cast<uint32_t>(nonce);
        // matmul_dim is IN the seed preimage (DeterministicMatMulSeedV3) and in the header
        // hash, so it must come from the job, not a default. Optional in the record; when
        // absent we keep the consensus dim rather than silently deriving with 0.
        {
            uint64_t dim = 0;
            if (GetU64(line, "matmul_dim", dim) && dim != 0 && dim <= 0xFFFF) {
                b.matmul_dim = static_cast<uint16_t>(dim);
            }
        }
        // Header matmul_dim: consensus at RC heights requires nMatMulV4Dimension (4096
        // mainnet) and it feeds the seed/sigma preimages. Optional per-record "matmul_n"
        // wins; absent/0 falls back to the consensus RC header dim. Mirrors the poolcore
        // job contract exactly, so auditor and miner can never disagree on the preimage.
        uint64_t matmul_n = 0;
        GetU64(line, "matmul_n", matmul_n);
        b.matmul_dim = matmul_n != 0 ? static_cast<uint16_t>(matmul_n)
                                     : matmul::v4::rc::RCConsensusHeaderMatmulDim();
        const auto tgt = uint256::FromHex(target_hex);
        if (!tgt.has_value()) { std::printf("{\"error\":\"bad_target\"}\n"); ++failures; continue; }
        const bool has_claim = GetStr(line, "digest_hex", claim_hex);

        const auto t0 = std::chrono::steady_clock::now();
        const uint256 digest = AuditDigest(b, params, static_cast<int32_t>(height),
                                           static_cast<int64_t>(pmtp), ep, use_gpu);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        if (digest.IsNull()) { std::printf("{\"error\":\"seed_derivation_failed\"}\n"); ++failures; continue; }

        const bool meets = UintToArith256(digest) <= UintToArith256(tgt.value());
        int matches = -1;
        if (has_claim) {
            matches = (digest.GetHex() == claim_hex) ? 1 : 0;
            if (matches == 0) ++failures;        // fabricated digest = the thing we hunt
        }
        std::printf("{\"digest_hex\":\"%s\",\"meets_target\":%d,\"matches_claim\":%s,"
                    "\"elapsed_ms\":%.1f}\n",
                    digest.GetHex().c_str(), meets ? 1 : 0,
                    matches < 0 ? "null" : (matches ? "1" : "0"), ms);
        std::fflush(stdout);
    }
    return failures == 0 ? 0 : 1;
}
