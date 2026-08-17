// rc_attest -- verify a block with matador's ENC_RC ExactReplay, then emit a signed
// trusted-mirror attestation the node will accept via submitmatmulattestations.
//
// The GPU replay is the GATE: we only ever sign a block our own byte-exact episode
// backend proved. The signature is not trust, it is how the node is told what our GPU
// already verified. (Node v0.33.2's own CUDA RC self-qual fails on this same 5090 with
// episode_digest_mismatch_backend_vs_cpu, which is why it cannot follow the RC chain.)
//
// Wire format (github.com/btxchain/btx src/matmul/trusted_exact_replay_attestation.h):
//   statement(103) = ver(1)=2 | chain_id(32) | block_hash(32) | height(4,LE)
//                    | major(1)=4 | profile(1)=1 | authority_context(32)
//   hash           = dSHA256( CompactSize(39) || "BTX_TRUSTED_EXACT_REPLAY_ATTESTATION_V2"
//                             || statement )
//   attestation    = statement || 0x21 || pubkey(33) || CompactSize(len) || DER-sig

#include <chainparams.h>
#include <hash.h>
#include <key.h>
#include <matmul/matmul_pow.h>
#include <matmul/matmul_v4_rc.h>
#include <pow.h>
#include <primitives/block.h>
#include <pubkey.h>
#include <streams.h>
#include <uint256.h>
#include <util/strencodings.h>

#include <cstdio>
#include <functional>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

// Every btx binary defines this itself; the archives reference it.
#include <util/translation.h>
const TranslateFn G_TRANSLATION_FUN{nullptr};

namespace {

const char* kDomain = "BTX_TRUSTED_EXACT_REPLAY_ATTESTATION_V2";

void PutU256(std::vector<unsigned char>& out, const uint256& v)
{
    out.insert(out.end(), v.begin(), v.end());
}

const char* Arg(int argc, char** argv, const char* name)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return nullptr;
}

} // namespace

int main(int argc, char** argv)
{
    const char* header_hex = Arg(argc, argv, "--header");
    const char* height_s   = Arg(argc, argv, "--height");
    const char* mtp_s      = Arg(argc, argv, "--parent-mtp");
    const char* chain_s    = Arg(argc, argv, "--chain-id");
    const char* ctx_s      = Arg(argc, argv, "--context");
    const char* key_s      = Arg(argc, argv, "--privkey");
    // --pubkey-only: derive and print the compressed pubkey for -matmultrustedpubkey.
    if (key_s && Arg(argc, argv, "--pubkey-only")) {
        ECC_Context ecc0{};
        const auto kb = TryParseHex<unsigned char>(key_s);
        if (!kb || kb->size() != 32) { std::fprintf(stderr, "bad privkey\n"); return 2; }
        CKey k0; k0.Set(kb->begin(), kb->end(), true);
        if (!k0.IsValid()) { std::fprintf(stderr, "invalid privkey\n"); return 2; }
        std::printf("%s\n", HexStr(k0.GetPubKey()).c_str());
        return 0;
    }
    if (!header_hex || !height_s || !mtp_s || !chain_s || !ctx_s || !key_s) {
        std::fprintf(stderr,
            "usage: rc_attest --header <hex> --height <n> --parent-mtp <n>\n"
            "                 --chain-id <hex64> --context <hex64> --privkey <hex64>\n");
        return 2;
    }

    SelectParams(ChainType::MAIN);
    const Consensus::Params& consensus = Params().GetConsensus();

    // ---- decode the header -------------------------------------------------
    const auto raw = TryParseHex<unsigned char>(header_hex);
    if (!raw) { std::fprintf(stderr, "rc_attest: --header is not hex\n"); return 2; }
    CBlockHeader header;
    try {
        DataStream ss{*raw};
        ss >> header;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "rc_attest: header deserialize failed: %s\n", e.what());
        return 2;
    }

    const int32_t height = static_cast<int32_t>(std::stol(height_s));
    const int64_t parent_mtp = std::stoll(mtp_s);

    // ---- THE GATE: matador's own ExactReplay -------------------------------
    CBlockHeader probe = header;
    if (!SetDeterministicMatMulSeeds(probe, consensus, height, std::optional<int64_t>(parent_mtp))) {
        std::fprintf(stderr, "rc_attest: SetDeterministicMatMulSeeds failed\n");
        return 3;
    }
    const uint256 sigma = matmul::DeriveSigma(probe);
    if (!matmul::v4::rc::RCEpisodeGpuAvailable()) {
        std::fprintf(stderr, "rc_attest: no CUDA episode backend; refusing to attest\n");
        return 3;
    }
    const uint256 digest =
        matmul::v4::rc::ComputeEpisodeDigestGPU(sigma, matmul::v4::rc::ActiveProfileEpisodeParams());

    const auto target = DeriveTarget(header.nBits, consensus.powLimit);
    if (!target) { std::fprintf(stderr, "rc_attest: bad nBits\n"); return 3; }

    const bool digest_matches = (digest == header.matmul_digest);
    const bool beats_target   = (UintToArith256(digest) <= *target);
    std::fprintf(stderr, "rc_attest: h=%d sigma=%s\n", height, sigma.GetHex().c_str());
    std::fprintf(stderr, "rc_attest: replay=%s header=%s match=%d beats_target=%d\n",
                 digest.GetHex().c_str(), header.matmul_digest.GetHex().c_str(),
                 digest_matches ? 1 : 0, beats_target ? 1 : 0);
    if (!digest_matches || !beats_target) {
        std::fprintf(stderr, "rc_attest: REFUSING to sign (replay did not verify this block)\n");
        return 4;
    }

    // ---- build the 103-byte statement --------------------------------------
    const auto chain_id = uint256::FromHex(chain_s);
    const auto context  = uint256::FromHex(ctx_s);
    if (!chain_id || !context) { std::fprintf(stderr, "rc_attest: bad chain-id/context\n"); return 2; }
    const uint256 block_hash = header.GetHash();

    std::vector<unsigned char> st;
    st.reserve(103);
    st.push_back(2);                       // version
    PutU256(st, *chain_id);
    PutU256(st, block_hash);
    for (int i = 0; i < 4; ++i) st.push_back(static_cast<unsigned char>((height >> (8 * i)) & 0xff));
    st.push_back(4);                       // matmul major
    st.push_back(1);                       // profile 1
    PutU256(st, *context);
    if (st.size() != 103) { std::fprintf(stderr, "rc_attest: statement %zu != 103\n", st.size()); return 5; }

    // ---- domain-separated hash ---------------------------------------------
    HashWriter hasher;
    const size_t dlen = std::strlen(kDomain);
    const unsigned char dlen_b = static_cast<unsigned char>(dlen);   // 39 -> single-byte CompactSize
    hasher.write(Span<const std::byte>(reinterpret_cast<const std::byte*>(&dlen_b), 1));
    hasher.write(Span<const std::byte>(reinterpret_cast<const std::byte*>(kDomain), dlen));
    hasher.write(Span<const std::byte>(reinterpret_cast<const std::byte*>(st.data()), st.size()));
    const uint256 msg = hasher.GetHash();

    // ---- sign ---------------------------------------------------------------
    ECC_Context ecc{};
    const auto keybytes = TryParseHex<unsigned char>(key_s);
    if (!keybytes || keybytes->size() != 32) { std::fprintf(stderr, "rc_attest: --privkey must be 32-byte hex\n"); return 2; }
    CKey key;
    key.Set(keybytes->begin(), keybytes->end(), /*fCompressedIn=*/true);
    if (!key.IsValid()) { std::fprintf(stderr, "rc_attest: invalid privkey\n"); return 2; }
    std::vector<unsigned char> sig;
    if (!key.Sign(msg, sig)) { std::fprintf(stderr, "rc_attest: sign failed\n"); return 5; }
    const CPubKey pub = key.GetPubKey();

    std::vector<unsigned char> att = st;
    att.push_back(static_cast<unsigned char>(pub.size()));            // 33
    att.insert(att.end(), pub.begin(), pub.end());
    att.push_back(static_cast<unsigned char>(sig.size()));            // DER < 253
    att.insert(att.end(), sig.begin(), sig.end());

    std::fprintf(stderr, "rc_attest: pubkey=%s\n", HexStr(pub).c_str());
    std::printf("%s\n", HexStr(att).c_str());
    return 0;
}
