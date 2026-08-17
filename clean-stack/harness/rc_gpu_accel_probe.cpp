// Byte-exact gate for the GPU ENC_RC episode backend (cuda/rc_episode_accel.cu):
// ComputeEpisodeDigestGPU MUST equal the CPU oracle ComputeEpisodeDigest AND the frozen toy
// golden 5b1bff3c. This is what lets SolveMatMulRCEpisode swap the CPU oracle for the ~200x GPU
// path without touching consensus. Runs at toy dims (fast, golden-gated).

#include <matmul/matmul_v4_rc.h>
#include <uint256.h>

#include <cstdio>

int main()
{
    const char* kGolden = "5b1bff3c835b1c8e7816a2cccb181eb2fc30a99d97a971d73108c52a8238acd4";
    // Same sigma the rc_probe/bench golden uses (arbitrary fixed value -> deterministic digest).
    const uint256 sigma = uint256::FromHex(
        "86c171d7ee6152a3a2a592a5c400adb9a680a06f3247b55f1e0935e129282fe2").value();
    const matmul::v4::rc::EpisodeParams toy{};  // toy dims = the frozen golden shape

    const uint256 cpu = matmul::v4::rc::ComputeEpisodeDigest(sigma, toy);
    std::printf("  cpu oracle : %s\n", cpu.GetHex().c_str());

#ifdef MATADOR_ENABLE_CUDA
    if (!matmul::v4::rc::RCEpisodeGpuAvailable()) {
        std::printf("  gpu backend: UNAVAILABLE (no CUDA device) -- cannot gate; treat as skip\n");
        return 0;
    }
    const uint256 gpu = matmul::v4::rc::ComputeEpisodeDigestGPU(sigma, toy);
    std::printf("  gpu backend: %s\n", gpu.GetHex().c_str());
    const bool cpu_gold = cpu.GetHex() == kGolden;
    const bool gpu_gold = gpu.GetHex() == kGolden;
    const bool match    = cpu == gpu;
    std::printf("=> cpu==golden %d, gpu==golden %d, gpu==cpu %d\n", cpu_gold, gpu_gold, match);
    return (cpu_gold && gpu_gold && match) ? 0 : 1;
#else
    std::printf("  (built without MATADOR_ENABLE_CUDA -- GPU backend not present)\n");
    return cpu.GetHex() == kGolden ? 0 : 1;
#endif
}
