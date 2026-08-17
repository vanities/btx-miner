// Runtime "can this GPU do a native FP4 pass?" detector for the miner.
//
// The block-scaled FP4 mma (mma.sync.kind::mxf4.block_scale) only ASSEMBLES for sm_120a
// (consumer Blackwell: 5090 / RTX PRO 6000). A plain -arch build for any other card FAILS at
// ptxas (verified: sm_86 -> "Unsupported gpu architecture compute_86a"; sm_100a -> "mma with
// block scale not supported"). So a single miner binary must be a MULTI-ARCH FATBIN with the FP4
// path guarded by __CUDA_ARCH__ -- then it runs on any card and only issues FP4 where legal.
//
// The B200 (sm_100) has native block-scaled MXFP4 too, but through a DIFFERENT instruction
// family (tcgen05.mma), not this warp-level mma -- so this detector reports it as "needs the
// SM100 path", not capable-via-this-kernel. That is honest: we have not implemented tcgen05.
//
// build (fatbin, runs anywhere; FP4 real only on sm_120a):
//   nvcc -O3 \
//     -gencode arch=compute_80,code=sm_80 \
//     -gencode arch=compute_86,code=sm_86 \
//     -gencode arch=compute_89,code=sm_89 \
//     -gencode arch=compute_90,code=sm_90 \
//     -gencode arch=compute_100,code=sm_100 \
//     -gencode arch=compute_120a,code=sm_120a \
//     fp4_capability.cu -o fp4cap
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Real FP4 mma only on sm_120a; empty elsewhere so the sm_8x/9x/100 fatbin slices still build.
__global__ void fp4_probe(int32_t* ok)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    // 4-bit packed E2M1 (nibble 0x5 = +3), UE8M0 scale 2^0. Values are irrelevant; we only need
    // the instruction to ISSUE without faulting.
    uint32_t a0 = 0x55555555u, a1 = 0x55555555u, a2 = 0x55555555u, a3 = 0x55555555u;
    uint32_t b0 = 0x55555555u, b1 = 0x55555555u;
    uint32_t sA = 0x7F7F7F7Fu, sB = 0x7F7F7F7Fu;
    float d[4] = {0, 0, 0, 0};
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(sA), "n"(0), "n"(0), "r"(sB), "n"(0), "n"(0));
    if (threadIdx.x == 0) *ok = 1 + (d[0] < -1e30f ? 1 : 0);   // 1 = issued; the +? never fires
#else
    if (threadIdx.x == 0) *ok = 0;   // instruction not compiled for this arch
#endif
}

// 0 = no FP4 pass; 1 = native FP4 pass via mma.kind::mxf4 (sm_120a); 2 = has HW but needs the
// SM100 tcgen05 path we have not built.
static int fp4_pass_capability(cudaDeviceProp& p, const char** why)
{
    if (p.major == 12) {                       // consumer Blackwell (5090, RTX PRO 6000)
        int32_t* d = nullptr; int32_t h = -1;
        if (cudaMalloc(&d, 4) != cudaSuccess) { *why = "cudaMalloc failed"; return 0; }
        cudaMemset(d, 0, 4);
        fp4_probe<<<1, 32>>>(d);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { *why = cudaGetErrorString(e); cudaFree(d); return 0; }
        cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost); cudaFree(d);
        if (h >= 1) { *why = "mma.kind::mxf4 issued"; return 1; }
        *why = "sm_120 but FP4 kernel not in this fatbin"; return 0;
    }
    if (p.major == 10) { *why = "B200-class: block-scaled MXFP4 is via tcgen05, not built"; return 2; }
    *why = "pre-Blackwell: no native FP4 mma"; return 0;
}

int main()
{
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("no CUDA device\n"); return 2; }
    const char* why = "";
    const int cap = fp4_pass_capability(p, &why);
    const char* label = cap == 1 ? "YES (native mma.kind::mxf4)"
                      : cap == 2 ? "PARTIAL (HW yes, needs SM100 tcgen05 path)"
                                 : "NO (int8 fallback)";
    printf("device      : %s  sm_%d%d\n", p.name, p.major, p.minor);
    printf("fp4 pass    : %s\n", label);
    printf("reason      : %s\n", why);
    printf("__FP4CAP__ {\"name\":\"%s\",\"sm\":\"%d%d\",\"fp4_pass\":%d,\"reason\":\"%s\"}\n",
           p.name, p.major, p.minor, cap, why);
    return 0;
}
