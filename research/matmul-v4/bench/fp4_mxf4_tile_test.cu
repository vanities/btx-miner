// Single-tile kind::mxf4 (m16n8k64, E2M1) correctness + byte-exactness test on sm_120a.
// Step 1 of a hand-written FP4 operand GEMM: verify the A(16x64)/B(64x8) fragment layout
// by matching a HOST integer reference on random M11 operands, with UNIFORM 2^0 block
// scales (isolates the A/B layout from the scale layout). M11 operands are the E2M1
// exact-int subset {0,±1,±2,±3,±4,±6}, so with 2^0 scale the mma computes the exact
// integer matmul C[16][8] = sum_k A[m][k]*B[k][n]. If the mma output == host int ref for
// varied operands (sums hit odd/low-bit values -> real t=24 stress), then BOTH the layout
// is correct AND the accumulator is byte-exact.
//
// Derived m16n8k64 .e2m1 layout (grouped with .s4/.u4 in PTX ISA), double-K of m16n8k32.s8:
//   groupID = lane>>2 (0..7), tig = lane&3 (0..3)
//   A[16][64]: a0={row gID,      col tig*8+[0..7]}   a1={row gID+8, col tig*8+[0..7]}
//              a2={row gID,      col tig*8+32+[0..7]} a3={row gID+8, col tig*8+32+[0..7]}
//   B[64][8]:  b0={row(K) tig*8+[0..7],    col(N) gID}  b1={row(K) tig*8+32+[0..7], col gID}
//   nibble j of a register sits at bits [4j,4j+3].
// E2M1 nibble encodings (sign|exp2|mant1): +1=0x2 +2=0x4 +3=0x5 +4=0x6 +6=0x7,
//   -1=0xA -2=0xC -3=0xD -4=0xE -6=0xF, 0=0x0.
//
// build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_mxf4_tile_test.cu -o fp4tile
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

// map an exact-int M11 value in {0,±1,±2,±3,±4,±6} to its E2M1 nibble
__host__ __device__ inline uint8_t m11_to_nib(int v){
    switch(v){ case 0:return 0x0; case 1:return 0x2; case 2:return 0x4; case 3:return 0x5;
               case 4:return 0x6; case 6:return 0x7; case -1:return 0xA; case -2:return 0xC;
               case -3:return 0xD; case -4:return 0xE; case -6:return 0xF; } return 0x0;
}
// the 8 legal magnitudes for random M11 draws
__host__ inline int rand_m11(){ static const int V[11]={0,1,2,3,4,6,-1,-2,-3,-4,-6}; return V[rand()%11]; }

// A[16][64], B[64][8] as int (host), packed to nibbles on device per the derived layout.
__global__ void mxf4_tile(const uint8_t* __restrict__ Anib, const uint8_t* __restrict__ Bnib,
                          float* out){
    int lane=threadIdx.x&31; int gID=lane>>2, tig=lane&3;
    // pack A: a0..a3
    uint32_t a[4]={0,0,0,0};
    for(int j=0;j<8;j++){
        int c0=tig*8+j, c32=tig*8+32+j;
        a[0]|=(uint32_t)Anib[(gID)  *64 + c0 ]<<(4*j);
        a[1]|=(uint32_t)Anib[(gID+8)*64 + c0 ]<<(4*j);
        a[2]|=(uint32_t)Anib[(gID)  *64 + c32]<<(4*j);
        a[3]|=(uint32_t)Anib[(gID+8)*64 + c32]<<(4*j);
    }
    // pack B: b0,b1  (B row-major [K][N] in Bnib)
    uint32_t b[2]={0,0};
    for(int j=0;j<8;j++){
        int k0=tig*8+j, k32=tig*8+32+j;
        b[0]|=(uint32_t)Bnib[k0 *8 + gID]<<(4*j);
        b[1]|=(uint32_t)Bnib[k32*8 + gID]<<(4*j);
    }
    uint32_t sA=0x7F7F7F7Fu, sB=0x7F7F7F7Fu;   // UE8M0 2^0, uniform
    float d[4]={0,0,0,0};
    asm volatile(
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]),
        "r"(sA),"n"(0),"n"(0), "r"(sB),"n"(0),"n"(0));
    // C fragment (m16n8, f32): thread holds c[0..3] at
    //   c0={row gID,   col tig*2+0}  c1={row gID,   col tig*2+1}
    //   c2={row gID+8, col tig*2+0}  c3={row gID+8, col tig*2+1}
    out[(gID)  *8 + tig*2+0]=d[0];
    out[(gID)  *8 + tig*2+1]=d[1];
    out[(gID+8)*8 + tig*2+0]=d[2];
    out[(gID+8)*8 + tig*2+1]=d[3];
}

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d\n",p.name,p.major,p.minor);
    srand(argc>1?atoi(argv[1]):1);
    const int M=16,N=8,K=64;
    int Ai[16*64], Bi[64*8];
    uint8_t Anib[16*64], Bnib[64*8];
    for(int i=0;i<M*K;i++){ Ai[i]=rand_m11(); Anib[i]=m11_to_nib(Ai[i]); }
    for(int i=0;i<K*N;i++){ Bi[i]=rand_m11(); Bnib[i]=m11_to_nib(Bi[i]); }
    // host reference: exact int matmul
    long long ref[16*8];
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long long s=0; for(int k=0;k<K;k++) s+=(long long)Ai[m*K+k]*Bi[k*N+n]; ref[m*N+n]=s; }

    uint8_t *dA,*dB; float* dO;
    CK(cudaMalloc(&dA,M*K)); CK(cudaMalloc(&dB,K*N)); CK(cudaMalloc(&dO,M*N*4));
    CK(cudaMemcpy(dA,Anib,M*K,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,Bnib,K*N,cudaMemcpyHostToDevice));
    mxf4_tile<<<1,32>>>(dA,dB,dO);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("mma launch failed: %s\n",cudaGetErrorString(e)); return 3; }
    float ho[16*8]; CK(cudaMemcpy(ho,dO,M*N*4,cudaMemcpyDeviceToHost));

    int mism=0,shown=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){
        long long got=(long long)ho[m*N+n];
        if(got!=ref[m*N+n]){ mism++; if(shown++<8) printf("  MISMATCH C[%2d][%d] mma=%lld ref=%lld\n",m,n,got,ref[m*N+n]); }
    }
    printf("%s  (%d/%d elements match; if all match, A/B layout is correct AND accumulator byte-exact)\n",
           mism==0?"=> ALL MATCH ✓":"=> LAYOUT/EXACTNESS MISMATCH", M*N-mism, M*N);
    return mism==0?0:1;
}
