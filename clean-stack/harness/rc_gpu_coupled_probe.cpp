// Byte-exact gate for the GPU coupled-V3 backend (rc_coupled_accel.cu).
//
// 1. Medium-V3 golden: GPU digest == CPU oracle == frozen a4bb0cc4, and the GPU-streamed bank
//    root == the CPU BuildCoupledBankV3 root. Medium has exchange_rounds=0 and 4 barriers, so:
// 2. Production-structure shape (8 barriers -> both mix patterns x4, exchange_rounds=2, 8 lobes,
//    W=128): GPU == CPU. No frozen golden exists for this shape; the CPU oracle IS the reference
//    (itself gated by rc_coupled_probe on the medium golden).
//
// Skips (exit 0, loud) when no CUDA device is present -- the CI box may be CPU-only; the pc run
// is the authoritative gate.

#include <matmul/matmul_v4_rc_coupled.h>
#include <uint256.h>

#include <cstdio>
#include <string_view>

static uint256 PU(std::string_view hex) { return uint256::FromHex(hex).value(); }

int main()
{
#ifndef MATADOR_ENABLE_CUDA
    std::printf("rc_gpu_coupled_probe: built without CUDA -- nothing to gate\n");
    return 0;
#else
    if (!matmul::v4::rc::RCCoupledGpuAvailable()) {
        std::printf("rc_gpu_coupled_probe: no CUDA device -- SKIPPED (run on the rig)\n");
        return 0;
    }

    const uint256 sigma =
        PU("86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2");
    const uint256 tmpl =
        PU("221c4a4edd1bdf4ecae55f67c7aad7691420a36405e7453bf24def8292ef1c1c");
    static constexpr const char* kGolden =
        "a4bb0cc42e2b97631d126a0dcdae26ad83b2f287d885322392a564990a95bac4";

    int fails = 0;

    // 1) medium golden, three-way
    {
        const auto p = matmul::v4::rc::MediumV3CoupParams();
        const auto bank = matmul::v4::rc::BuildCoupledBankV3(tmpl, 0, p);
        const uint256 root_gpu = matmul::v4::rc::ComputeCoupledBankRootGPU(tmpl, 0, p);
        const uint256 cpu = matmul::v4::rc::ComputeCoupledDigestV3WithBank(sigma, bank, p);
        const uint256 gpu = matmul::v4::rc::ComputeCoupledDigestV3GPU(sigma, tmpl, 0, root_gpu, p);
        const bool root_ok = root_gpu == bank.bank_root;
        const bool cpu_ok  = cpu.GetHex() == kGolden;
        const bool gpu_ok  = gpu.GetHex() == kGolden;
        std::printf("  medium  bank_root gpu==cpu=%d  cpu==golden=%d  gpu==golden=%d (%s)\n",
                    root_ok, cpu_ok, gpu_ok, gpu.GetHex().substr(0, 16).c_str());
        if (!root_ok || !cpu_ok || !gpu_ok) ++fails;
    }

    // 2) production structure at probe scale: 8 barriers (asc+desc mixes x4), exchange rounds on
    {
        matmul::v4::rc::CoupParamsV3 p;
        p.barriers = 8; p.lobes = 8; p.lobe_width = 128; p.bank_pages = 32;
        p.rows_per_lobe = 32; p.pages_per_barrier_lobe = 2; p.exchange_rounds = 2;   // 8*8*2 hits all 32 pages 4x
        const auto bank = matmul::v4::rc::BuildCoupledBankV3(tmpl, 7, p);
        const uint256 root_gpu = matmul::v4::rc::ComputeCoupledBankRootGPU(tmpl, 7, p);
        const uint256 cpu = matmul::v4::rc::ComputeCoupledDigestV3WithBank(sigma, bank, p);
        const uint256 gpu = matmul::v4::rc::ComputeCoupledDigestV3GPU(sigma, tmpl, 7, root_gpu, p);
        const bool root_ok = root_gpu == bank.bank_root;
        const bool ok = root_ok && !cpu.IsNull() && cpu == gpu;
        std::printf("  prodstr bank_root gpu==cpu=%d  gpu==cpu=%d (%s)\n",
                    root_ok, cpu == gpu, gpu.GetHex().substr(0, 16).c_str());
        if (!ok) ++fails;
    }

    std::printf("=> GPU coupled V3 %s\n",
                fails == 0 ? "BYTE-EXACT (a4bb0cc4 + prod-structure cross-check)"
                           : "FORKED FROM THE CPU ORACLE");
    return fails == 0 ? 0 : 1;
#endif
}
