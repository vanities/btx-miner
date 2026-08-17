// kind::mxf4 ACCUMULATOR-EXACTNESS probe for the BMX4-C operand GEMM (5090 sm_120a).
//
// The FP4 operand-GEMM lever (replace the INT8 Q=B*V with a hand kind::mxf4 GEMM,
// ~2x tensor) is only viable if it is BYTE-EXACT to the INT8 s32 path. The M11
// operands dequantize to <= |48| (mantissa {0,±1,±2,±3,±4,±6} << E8M0 {0..3}), so a
// K=4096 product-sum maxes at 4096 * 48*48 = 9,437,184 < 2^24 (16,777,216). f32 holds
// every integer <= 2^24 exactly, so IF the mxf4 tensor core accumulates in true f32
// (t=24), the FP4 result is bit-identical to INT8 at our magnitudes. Hopper-class FP4
// rounds at ~t=14 -> would NOT be exact. This probe settles it, layout-agnostic:
// uniform max operands so every output element == the same known integer.
//
//   A,B elems = E2M1 nibble 0x7 (= +6);  block scale = UE8M0 0x82 (= 2^3).
//   per k-element product = 6*6 * 2^(3+3) = 36*64 = 2304.
//   one m16n8k64 mma accumulates 64*2304 = 147456;  64 mmas -> 9,437,184 (our worst case).
//
// build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_mxf4_exact_probe.cu -o fp4x
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

// One warp, one 16x8 output tile, K = 64*NMMA accumulated in a single f32 accumulator.
__global__ void mxf4_exact(float* out, int nmma){
    uint32_t a0=0x77777777u,a1=0x77777777u,a2=0x77777777u,a3=0x77777777u; // packed E2M1 +6
    uint32_t b0=0x77777777u,b1=0x77777777u;
    uint32_t sA=0x82828282u, sB=0x82828282u;                              // UE8M0 2^3
    float d[4]={0.f,0.f,0.f,0.f};
    for(int n=0;n<nmma;n++){
        asm volatile(
          "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
          : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
            "r"(sA),"n"(0),"n"(0), "r"(sB),"n"(0),"n"(0));
    }
    if(threadIdx.x==0) out[0]=d[0];
    if(threadIdx.x==1) out[1]=d[0];
}

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d\n",p.name,p.major,p.minor);
    // probe the mma is even available on this arch
    float* d; CK(cudaMalloc(&d,4*sizeof(float)));
    mxf4_exact<<<1,32>>>(d,1);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf(" kind::mxf4 unavailable: %s\n",cudaGetErrorString(e)); return 3; }

    for(int K : {64, 4096, 8192}){
        int nmma=K/64;
        mxf4_exact<<<1,32>>>(d,nmma);
        CK(cudaDeviceSynchronize());
        float h[2]; CK(cudaMemcpy(h,d,2*sizeof(float),cudaMemcpyDeviceToHost));
        // exact integer reference: K * (6*6) * 2^(3+3) = K * 2304
        double ref = (double)K * 2304.0;
        long long got = (long long)h[0];
        printf(" K=%-5d  mxf4 acc = %.0f   exact int = %.0f   %s\n",
               K, (double)h[0], ref, ((double)h[0]==ref)?"EXACT":"** ROUNDED (t<24) **");
        if(K==4096) printf("   (K=4096 is the BMX4-C operand-GEMM depth; 9,437,184 is the byte-exactness worst case, < 2^24=16,777,216)\n");
    }
    printf("VERDICT: if K=4096 is EXACT, a hand kind::mxf4 operand GEMM CAN be byte-identical to INT8.\n");
    return 0;
}
