// v4.6 ENC_RC (Resident Curriculum) episode -- byte-exact CPU reference.
//
// Reproduces btx PR#89's frozen ENC_RC_V1 golden (v4.6 fused-FFN re-golden 5b1bff3c; the
// pre-fused b339d0ff is DEAD upstream). Used two ways:
//   1. rc_probe gates it in CI (5b1bff3c... for the toy params below), so a future edit to the
//      Extract / GEMM / Merkle path cannot silently fork from consensus.
//   2. It is the oracle the GPU episode (research rc_gpu_solver.cu, byte-exact on 5090) is
//      diffed against -- correctness first, speed second.
//
// INERT: nMatMulRCHeight is INT32_MAX on every public net -- but v4.6 wired SINGLE-SWITCH
// activation (one height turns episode+coupled on together, profile default = datacenter).
// The DatacenterEpisodeParams shape below (Config W sharing + row-block X0) is what mainnet
// would run; keep this module ready to wire into the solve path the day the height goes finite.
#ifndef MATADOR_MATMUL_V4_RC_H
#define MATADOR_MATMUL_V4_RC_H

#include <uint256.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace matmul::v4::rc {

/** Episode shape. Defaults are the ENC_RC_V1 toy params (the frozen golden's shape). */
struct EpisodeParams {
    uint32_t rounds{1};
    uint32_t d_head{32};
    uint32_t n_q{32};
    uint32_t n_ctx{64};
    uint32_t L_lyr{2};
    uint32_t d_model{32};
    uint32_t d_ff{128}; // fused-FFN inner width (4x d_model)
    uint32_t b_seq{32};
    uint32_t T_leaf{64};
    // Datacenter (profile 2) semantics -- both false for the toy/base golden.
    bool share_ep{false};    // Config W: K/V + one (W_up,W_down) pair sigma-derived episode-wide
    bool rowblock_x0{false}; // X0 expanded as independent 32-row blocks (X0_ROW_BLOCK tag)
};

/** Epoch-0 base production dims (rounds=4, L=16, b_seq=16384, d_ff=16384, T_leaf=1024). */
[[nodiscard]] EpisodeParams ProductionEpisodeParams();

/** Episode shape for the ACTIVE launch profile. Upstream PR#97 (v4.7) fixes the Epoch A
 *  activation tuple to PROFILE 1 = ProductionEpisodeParams (rounds=4, L=16, b_seq=16384,
 *  T_leaf=1024, dim 4096; ExactReplay authority); Profile 2 (DatacenterEpisodeParams) is the
 *  later, separately-activated Epoch D shape. Selection: env BTX_MATMUL_RC_PROFILE ("1"
 *  default / "2" datacenter) -- mirrors Consensus::Params::nMatMulRCProfile{1} upstream
 *  without vendored-struct ABI drift (same reasoning as CoupledActivationHeight). */
[[nodiscard]] EpisodeParams ActiveProfileEpisodeParams();

/** Datacenter profile-2 dims -- THE mainnet activation shape (v4.6 default profile):
 *  rounds=8, L=24, b_seq=87552, T_leaf=4096, Config W sharing + row-block X0. ~16x base MAC. */
[[nodiscard]] EpisodeParams DatacenterEpisodeParams();

/** Header matmul_dim consensus REQUIRES at RC heights. Upstream's
 *  CheckMatMulProofOfWork_RCOutcome rejects any header whose matmul_dim !=
 *  nMatMulV4Dimension (4096 on mainnet; sealed tip 909aa703, activation height 181'894),
 *  and the field feeds BOTH the V3 seed preimage and the sigma preimage -- so the solver
 *  must stamp it BEFORE deriving seeds, or the ground sigma (and thus every digest) can
 *  never validate. The vendored consensus struct predates the v4 field (same ABI reason
 *  as RCActivationHeight), so it lives here clean-stack-side. Env override
 *  BTX_MATMUL_RC_HEADER_DIM for regtest/foreign nets; latched once. */
[[nodiscard]] inline uint16_t RCConsensusHeaderMatmulDim()
{
    static const uint16_t dim = [] {
        if (const char* e = std::getenv("BTX_MATMUL_RC_HEADER_DIM")) {
            const long v = std::atol(e);
            if (v > 0 && v <= 65535) return static_cast<uint16_t>(v);
        }
        return static_cast<uint16_t>(4096);
    }();
    return dim;
}

/** ENC_RC episode digest = SHA256d("BTX_RC_EPISODE_V1" || round_roots...).
 *  `sigma` is the v4 DeriveSigma(header) output; the RC chain hangs off it. */
[[nodiscard]] uint256 ComputeEpisodeDigest(const uint256& sigma, const EpisodeParams& params);

/** The per-round Merkle roots (round 0..R-1) the episode digest is committed over -- the witness
 *  material a prover needs to reconstruct the episode. `digest == SHA256d("BTX_RC_EPISODE_V1" ||
 *  roots...)`, so a caller can re-derive the digest from these as a self-check. Byte-exact to the
 *  digest path (same run_episode_impl). */
[[nodiscard]] std::vector<uint256> ComputeEpisodeRoundRoots(const uint256& sigma,
                                                            const EpisodeParams& params);

// ---- GPU episode backend (matador_core CUDA builds only) --------------------------------
// ComputeEpisodeDigestGPU is byte-exact to ComputeEpisodeDigest but runs the whole episode on
// the GPU (~200x at datacenter dims). Defined in cuda/rc_episode_accel.cu; only linked when
// MATADOR_ENABLE_CUDA. Callers must guard on MATADOR_ENABLE_CUDA (compile) and check
// RCEpisodeGpuAvailable() (runtime) before use, and fall back to the CPU oracle otherwise.
#ifdef MATADOR_ENABLE_CUDA
[[nodiscard]] bool RCEpisodeGpuAvailable();
[[nodiscard]] uint256 ComputeEpisodeDigestGPU(const uint256& sigma, const EpisodeParams& params);
/** Per-round Merkle roots via the GPU episode (witness material). Byte-exact to
 *  ComputeEpisodeRoundRoots; the CPU reference is HOURS at Profile-1 dims, so the witness
 *  path must use this whenever a device is present. */
[[nodiscard]] std::vector<uint256> ComputeEpisodeRoundRootsGPU(const uint256& sigma,
                                                               const EpisodeParams& params);
#endif

/** E8M0 block scale e in {0..3}: SHA256("BTX_MATEXPAND_MXSCALE_V44LT"||prf||LE32(i)||LE32(bj))[0]&3. */
[[nodiscard]] uint8_t DeriveMxScale(const uint256& prf_key, uint32_t i, uint32_t bj);

/** 32-wide MX Extract on int64 raws: tile ChaCha20 mantissas (M11 rejection) scaled by 1<<e.
 *  This is the hot stage -- one independent tile per (i,bj), which is why it parallelises. */
void ExtractMxTileInt64(const uint256& prf_key, uint32_t i, uint32_t bj,
                        const int64_t raw64[32], int8_t out[32]);

} // namespace matmul::v4::rc

#endif // MATADOR_MATMUL_V4_RC_H
