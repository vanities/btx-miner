// rc_episode_accel.cu -- GPU episode digest backend for matador_core.
//
// Thin wrapper: compiles the byte-exact research episode solver (rc_gpu_solver.cu, which the
// bench gates against goldens 5b1bff3c toy / 1cc5709d datacenter) as a LIBRARY unit and exposes
// a clean host entry the RC solve loop can call. Keeping the research file as the single source
// of truth (still builds standalone as the bench) means byte-exactness is preserved by
// construction -- we do not re-derive the ~1900-line pipeline, we link it.
//
// RC_GPU_SOLVER_AS_LIB suppresses the bench main(). All of the bench's file-scope symbols are
// `static` (internal linkage), so they do not collide with matmul_v4_rc.cpp's CPU path.

#define RC_GPU_SOLVER_AS_LIB 1
#include "rc_gpu_episode.cu"

#include <cstring>

#include <span.h>
#include <uint256.h>
#include <matmul/matmul_v4_rc.h>

namespace matmul::v4::rc {

// EpisodeParams (clean-stack) -> Params (bench). Field-for-field identical shapes.
static Params RcBenchParamsFromEpisode(const EpisodeParams& e)
{
    Params p;
    p.rounds = e.rounds;   p.d_head = e.d_head; p.n_q = e.n_q;   p.n_ctx = e.n_ctx;
    p.L_lyr = e.L_lyr;     p.d_model = e.d_model; p.d_ff = e.d_ff; p.b_seq = e.b_seq;
    p.T_leaf = e.T_leaf;   p.share_ep = e.share_ep; p.rowblock_x0 = e.rowblock_x0;
    return p;
}

bool RCEpisodeGpuAvailable()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Byte-exact to ComputeEpisodeDigest(sigma, params) (the CPU oracle) -- same digest, ~200x
// faster at datacenter dims. sigma<->H256 and H256<->uint256 mirror the CPU path's
// h256_from / to_uint256 exactly (direct 32-byte copies).
uint256 ComputeEpisodeDigestGPU(const uint256& sigma, const EpisodeParams& ep)
{
    H256 s{};
    std::memcpy(s.data(), sigma.data(), 32);
    const H256 d = run_episode_gpu(s, RcBenchParamsFromEpisode(ep));
    return uint256{Span<const unsigned char>{d.data(), 32}};
}

// Round roots on the GPU path: same episode run, roots exported instead of discarded.
// Byte-exact to ComputeEpisodeRoundRoots (the CPU reference) by construction -- the digest
// is SHA256d(tag || roots), so `digest == SHA256d(tag || returned roots)` is a free
// self-check callers can (and the witness path does) rely on.
std::vector<uint256> ComputeEpisodeRoundRootsGPU(const uint256& sigma, const EpisodeParams& ep)
{
    H256 s{};
    std::memcpy(s.data(), sigma.data(), 32);
    std::vector<H256> roots;
    (void)run_episode_gpu_roots(s, RcBenchParamsFromEpisode(ep), &roots);
    std::vector<uint256> out;
    out.reserve(roots.size());
    for (const auto& r : roots)
        out.emplace_back(Span<const unsigned char>{r.data(), 32});
    return out;
}

}  // namespace matmul::v4::rc
