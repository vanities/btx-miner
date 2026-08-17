// Byte-exact native FP4 GEMM for the RC miner -- feasibility gate, step 1.
//
// GOAL: prove a native mma.kind::mxf4 reproduces the INT8 integer result exactly on M11 operands,
// so an FP4-computed share still hashes to 03977a81. The scheme (worked out, not yet a full GEMM):
//   * M11 mantissas {0,+-1,+-2,+-3,+-4,+-6} are all E2M1-exact -> one nibble each, no decomposition.
//   * A's MX scale is along K -> feed it as the A block_scale (E8M0). B's scale is along the free
//     axis -> it factors OUT of the k-sum, applied as a per-output-column post-multiply.
//   * FP4 mma accumulates in FP32 (exact only to 2^24); RC contractions overflow that, so drain
//     to int32 every ~900 k-steps. No extra mma passes -> the ~2x FP4 rate survives.
//
// This step validates the MECHANISM (encoding + block scale + FP32->int exactness) with UNIFORM
// operands, which is independent of the fragment layout. If a uniform tile is not bit-exact, the
// scheme is dead before we invest in the layout. Layout-correct non-uniform tiles are step 2.
//
// build (5090): nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_gemm_exact.cu -o fp4gx
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

// E2M1 nibble codes (sign<<3 | exp<<1 | man). Magnitudes: 0,0.5,1,1.5,2,3,4,6.
//   1 -> 0b010 (exp1,man0)   2 -> 0b100   3 -> 0b101   4 -> 0b110   6 -> 0b111   0 -> 0b000
// M11 uses {0,1,2,3,4,6}; a negative sets bit 3.
__device__ __host__ static inline uint8_t e2m1_code(int v){
    int mag = v<0?-v:v; uint8_t c;
    switch(mag){ case 0:c=0;break; case 1:c=0b010;break; case 2:c=0b100;break;
                 case 3:c=0b101;break; case 4:c=0b110;break; case 6:c=0b111;break;
                 default:c=0;break; }
    return v<0 ? (c|0b1000) : c;
}
// E8M0 code for 2^s (s in 0..3): biased exponent 127+s.
__device__ __host__ static inline uint8_t e8m0_code(int s){ return (uint8_t)(127+s); }

// One m16n8k64 tile, UNIFORM operands: every A elem = va (M11), every B elem = vb, A block-scale
// = 2^sa, B block-scale = unit. Expected C[i,j] = 64 * va * vb * 2^sa for all i,j (k=64).
__global__ void fp4_uniform_tile(int va, int vb, int sa, int32_t* out /*16*8*/){
    const uint8_t na = e2m1_code(va), nb = e2m1_code(vb);
    const uint32_t A = (uint32_t)(na)*0x11111111u;   // 8 identical nibbles per .b32
    const uint32_t B = (uint32_t)(nb)*0x11111111u;
    uint32_t a0=A,a1=A,a2=A,a3=A, b0=B,b1=B;
    // scale operands: A = 2^sa (E8M0), B = unit (2^0). scale_vec::2X wants a small vector; the
    // block-scale selector args {%13}/{%14,%15} pick lanes -- uniform so any selection is fine.
    uint32_t sA = (uint32_t)e8m0_code(sa)*0x01010101u;
    uint32_t sB = (uint32_t)e8m0_code(0) *0x01010101u;
    float d[4]={0,0,0,0};
    asm volatile(
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
        "r"(sA),"n"(0),"n"(0), "r"(sB),"n"(0),"n"(0));
    // Each thread holds 4 accumulator values (c0..c3) for its output positions. Write them all;
    // for uniform inputs every position should equal the same integer.
    const int lane = threadIdx.x;
    #pragma unroll
    for(int i=0;i<4;i++) out[lane*4+i] = (int32_t)llrintf(d[i]);
}

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d\n", p.name,p.major,p.minor);
    int32_t* d; CK(cudaMalloc(&d,128*sizeof(int32_t)));
    struct Case{int va,vb,sa;} cases[]={
        {1,1,0},{2,3,0},{6,6,0},{3,4,0},   // encoding: expect 64*va*vb
        {2,3,1},{6,6,2},{1,1,3},            // + block scale: expect 64*va*vb*2^sa
        {-2,3,0},{6,-6,1},{-3,-4,2},        // signs
    };
    int fails=0;
    for(auto c: cases){
        CK(cudaMemset(d,0,128*sizeof(int32_t)));
        fp4_uniform_tile<<<1,32>>>(c.va,c.vb,c.sa,d);
        cudaError_t e=cudaDeviceSynchronize();
        if(e!=cudaSuccess){ printf(" va=%d vb=%d sa=%d -> LAUNCH ERR %s\n",c.va,c.vb,c.sa,cudaGetErrorString(e)); fails++; continue; }
        int32_t h[128]; CK(cudaMemcpy(h,d,sizeof(h),cudaMemcpyDeviceToHost));
        long expect=(long)64*c.va*c.vb*(1<<c.sa);
        // check all 128 written slots agree with expect (uniform)
        int ok=1; for(int i=0;i<128;i++) if(h[i]!=expect){ ok=0; break; }
        printf(" va=%3d vb=%3d sa=%d  expect %6ld  got %6d  %s\n",
               c.va,c.vb,c.sa,expect,h[0], ok?"EXACT":"MISMATCH");
        if(!ok) fails++;
    }
    printf("\n%s\n", fails==0 ? "MECHANISM BYTE-EXACT -- encoding + block scale + FP32->int all exact"
                              : "MECHANISM BROKEN -- FP4 cannot reproduce int8 (or layout differs)");
    return fails==0?0:1;
}
