// v4.6 ENC_RC COUPLED puzzle (V3 transcript family) -- byte-exact CPU reference.
//
// Reproduces btx PR#89's frozen medium-V3 golden (a4bb0cc4...) under the independent
// COUP_*_V3 domain tags. Lifted from the validated research oracle
// (research/matmul-v4/bench/rc_coupled_solver_v3.cpp). Used two ways:
//   1. rc_coupled_probe gates it in CI, so an edit to the bank / GEMM / mix / Extract path
//      cannot silently fork from consensus.
//   2. It is the oracle any GPU coupled solver is diffed against.
//
// INERT until a finite nMatMulRCCoupledHeight -- but v4.6 upstream made V3 production the
// DEFAULT coupled profile behind single-switch activation, so this is the shape that goes
// live. Production dims: barriers=8 lobes=8 W=8192 bank_pages=1536 M=128 P=24 (51 GiB packed
// bank) + exchange_rounds=4. The medium shape below is the CI golden.
#ifndef MATADOR_MATMUL_V4_RC_COUPLED_H
#define MATADOR_MATMUL_V4_RC_COUPLED_H

#include <uint256.h>

#include <cstdint>
#include <vector>

// Forward decl only: primitives/block.h drags in serialize.h, which nvcc cannot parse, and this
// header is consumed by the CUDA coupled backend (rc_coupled_accel.cu). RCBankTemplateHash only
// needs the reference; implementation files include primitives/block.h themselves.
class CBlockHeader;

namespace matmul::v4::rc {

/** Coupled puzzle shape + digest-affecting options (ENC_RC_V3 transcript). */
struct CoupParamsV3 {
    uint32_t barriers{4};
    uint32_t lobes{4};
    uint32_t lobe_width{64};
    uint32_t bank_pages{64};
    uint32_t rows_per_lobe{32};          // GEMM M; >=32 selects the uint64-wrap mix ring
    uint32_t pages_per_barrier_lobe{4};  // P (full-bank schedule)
    uint32_t exchange_rounds{0};         // V3 production = 4; medium golden = 0
};

/** Medium-V3 CI shape (MakeMediumV3RCCoupParams + MakeMediumV3RCCoupOptions). */
[[nodiscard]] CoupParamsV3 MediumV3CoupParams();
/** V3 production shape (MakeProductionV3RCCoupParams + MakeV3RCCoupOptions). */
[[nodiscard]] CoupParamsV3 ProductionV3CoupParams();

/**
 * Nonce-independent coupled bank: dequantized int8 pages + the bank commitment root. Depends
 * only on (bank_template_hash, height, params) -- build ONCE per block template and feed every
 * nonce through ComputeCoupledDigestV3WithBank. At production dims the dequantized retention is
 * ~96 GiB (1536 pages x 64 MiB); packed FP4 residency is the GPU backend's concern, not this
 * CPU reference's.
 */
struct CoupledBankV3 {
    std::vector<std::vector<int8_t>> pages;   // bank_pages entries of lobe_width*lobe_width int8
    uint256 bank_root;                        // double-SHA bank commitment (streamed)
};

/** Derive all bank pages + the bank commitment root for one template. Byte-exact to the bank
 *  leg of ComputeCoupledDigestV3 (which is now a build-then-compute wrapper over this). */
[[nodiscard]] CoupledBankV3 BuildCoupledBankV3(const uint256& bank_template_hash, uint32_t height,
                                               const CoupParamsV3& params);

/** Per-nonce coupled digest against a prebuilt bank. Returns null on a params/bank shape
 *  mismatch (never grinds blind). */
[[nodiscard]] uint256 ComputeCoupledDigestV3WithBank(const uint256& sigma,
                                                     const CoupledBankV3& bank,
                                                     const CoupParamsV3& params);

/**
 * Coupled V3 episode digest = SHA256d(COUP_EPISODE_V3 || bank_root || barrier_roots...).
 * `sigma` = DeriveSigma(header); `bank_template_hash` = RCBankTemplateHash(header) (the
 * nonce-nulled template projection, so the bank is shared across nonces of one template).
 * One-shot convenience: builds the bank then computes -- a solve loop should instead call
 * BuildCoupledBankV3 once per template + ComputeCoupledDigestV3WithBank per nonce (the bank
 * rebuild is the dominant per-nonce cost otherwise).
 */
[[nodiscard]] uint256 ComputeCoupledDigestV3(const uint256& sigma,
                                             const uint256& bank_template_hash,
                                             uint32_t height, const CoupParamsV3& params);

/** `bank_template_hash` for ComputeCoupledDigestV3, derived from a header: the nonce/seed/
 *  matmul_digest-nulled template projection hashed via ComputeMatMulHeaderHash (so the coupled
 *  bank is shared across every nonce of one template). Byte-exact to upstream RCBankTemplateHash. */
[[nodiscard]] uint256 RCBankTemplateHash(const CBlockHeader& header);

// GPU coupled backend (rc_coupled_accel.cu; only linked when MATADOR_ENABLE_CUDA). Byte-exact to
// the CPU oracle above (gated by rc_gpu_coupled_probe: GPU == CPU == golden a4bb0cc4). Pages are
// derived ON DEVICE -- template-cached in VRAM when the bank fits, re-derived per use otherwise
// -- so nothing is uploaded per nonce. Callers must check RCCoupledGpuAvailable() (runtime) and
// fall back to the CPU oracle otherwise.
#ifdef MATADOR_ENABLE_CUDA
[[nodiscard]] bool RCCoupledGpuAvailable();
/** Bank commitment root streamed from device-derived pages -- once per template; the GPU
 *  equivalent of BuildCoupledBankV3().bank_root without the ~96 GiB host retention. */
[[nodiscard]] uint256 ComputeCoupledBankRootGPU(const uint256& bank_template_hash, uint32_t height,
                                                const CoupParamsV3& params);
/** Per-nonce coupled digest on GPU. `bank_root` from ComputeCoupledBankRootGPU (or the CPU
 *  BuildCoupledBankV3), once per template. */
[[nodiscard]] uint256 ComputeCoupledDigestV3GPU(const uint256& sigma,
                                                const uint256& bank_template_hash, uint32_t height,
                                                const uint256& bank_root,
                                                const CoupParamsV3& params);
#endif

} // namespace matmul::v4::rc

#endif // MATADOR_MATMUL_V4_RC_COUPLED_H
