// MEASURE-ONLY bench for the ENC_RC COUPLED (V3) solver path -- the standing instrument for
// coupled-solver A/Bs (CPU restructures now, the GPU coupled backend later).
//
// Reports, per shape:
//   bank_build_ms   -- BuildCoupledBankV3 once (the per-TEMPLATE cost after the bank hoist)
//   nonce_ms        -- ComputeCoupledDigestV3WithBank per nonce (the real grind cost)
//   oneshot_ms      -- legacy ComputeCoupledDigestV3 per nonce (bank rebuilt per nonce; what the
//                      solve loop paid before the hoist). Skipped when --no-oneshot.
//
// Shapes: medium (CI golden dims), production (8/8/8192/1536/128/24/x4, ~96 GiB dequant bank --
// only on hosts with the RAM), or scaled (production structure at reduced W/pages for laptop
// runs). Digest correctness is NOT gated here (rc_coupled_probe does that); this only times.
//
// Usage: rc_coupled_bench [--shape medium|scaled|production] [--nonces N] [--no-oneshot]

#include <matmul/matmul_v4_rc_coupled.h>
#include <uint256.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static uint256 PU(std::string_view hex) { return uint256::FromHex(hex).value(); }

int main(int argc, char** argv)
{
    std::string shape = "scaled";
    int nonces = 3;
    bool run_oneshot = true;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--shape" && i + 1 < argc) shape = argv[++i];
        else if (a == "--nonces" && i + 1 < argc) nonces = std::atoi(argv[++i]);
        else if (a == "--no-oneshot") run_oneshot = false;
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    matmul::v4::rc::CoupParamsV3 p;
    if (shape == "medium") {
        p = matmul::v4::rc::MediumV3CoupParams();
    } else if (shape == "production") {
        p = matmul::v4::rc::ProductionV3CoupParams();
    } else if (shape == "scaled") {
        // production structure (8 barriers/8 lobes/exchange 4, full-bank schedule b*l*P == pages)
        // at W=2048, 384 pages -> 1.5 GiB dequant bank; laptop-runnable.
        p.barriers = 8; p.lobes = 8; p.lobe_width = 2048; p.bank_pages = 384;
        p.rows_per_lobe = 128; p.pages_per_barrier_lobe = 6; p.exchange_rounds = 4;
    } else {
        std::fprintf(stderr, "unknown shape: %s\n", shape.c_str());
        return 2;
    }

    const double bank_gib = double(p.bank_pages) * p.lobe_width * p.lobe_width / (1024.0 * 1024.0 * 1024.0);
    std::printf("[rc-coupled-bench] shape=%s barriers=%u lobes=%u W=%u pages=%u M=%u P=%u xr=%u "
                "(dequant bank %.2f GiB)\n",
                shape.c_str(), p.barriers, p.lobes, p.lobe_width, p.bank_pages, p.rows_per_lobe,
                p.pages_per_barrier_lobe, p.exchange_rounds, bank_gib);

    const uint256 tmpl =
        PU("221c4a4edd1bdf4ecae55f67c7aad7691420a36405e7453bf24def8292ef1c1c");

    auto t0 = Clock::now();
    const auto bank = matmul::v4::rc::BuildCoupledBankV3(tmpl, /*height=*/0, p);
    std::printf("[rc-coupled-bench] bank_build_ms=%.1f (once per template)\n", ms_since(t0));

#ifdef MATADOR_ENABLE_CUDA
    const bool gpu = matmul::v4::rc::RCCoupledGpuAvailable();
    uint256 gpu_root;
    if (gpu) {
        t0 = Clock::now();
        gpu_root = matmul::v4::rc::ComputeCoupledBankRootGPU(tmpl, 0, p);
        std::printf("[rc-coupled-bench] gpu_bank_root_ms=%.1f root==cpu=%d\n", ms_since(t0),
                    gpu_root == bank.bank_root ? 1 : 0);
    }
#endif

    // distinct sigmas per nonce so no cross-nonce memoization could hide in a future backend
    for (int i = 0; i < nonces; ++i) {
        uint256 sigma = PU("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2");
        *sigma.begin() = static_cast<unsigned char>(i + 1);

        t0 = Clock::now();
        const uint256 d = matmul::v4::rc::ComputeCoupledDigestV3WithBank(sigma, bank, p);
        const double with_bank_ms = ms_since(t0);

#ifdef MATADOR_ENABLE_CUDA
        if (gpu) {
            t0 = Clock::now();
            const uint256 dg = matmul::v4::rc::ComputeCoupledDigestV3GPU(sigma, tmpl, 0, gpu_root, p);
            const double gpu_ms = ms_since(t0);
            if (dg != d) {
                std::printf("!! GPU/CPU DIGEST MISMATCH at nonce %d -- BYTES FORKED\n", i);
                return 1;
            }
            std::printf("[rc-coupled-bench] nonce=%d gpu_nonce_ms=%.1f (cpu %.1f -> %.1fx)\n",
                        i, gpu_ms, with_bank_ms, with_bank_ms / gpu_ms);
        }
#endif

        double oneshot_ms = -1.0;
        if (run_oneshot) {
            t0 = Clock::now();
            const uint256 d2 = matmul::v4::rc::ComputeCoupledDigestV3(sigma, tmpl, 0, p);
            oneshot_ms = ms_since(t0);
            if (d != d2) {
                std::printf("!! split/one-shot DIGEST MISMATCH at nonce %d -- BYTES FORKED\n", i);
                return 1;
            }
        }
        std::printf("[rc-coupled-bench] nonce=%d nonce_ms=%.1f%s digest=%s\n", i, with_bank_ms,
                    run_oneshot ? (" oneshot_ms=" + std::to_string(oneshot_ms)).c_str() : "",
                    d.GetHex().substr(0, 16).c_str());
    }
    return 0;
}
