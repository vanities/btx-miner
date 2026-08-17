// Step 2: verify the SFA block-scale layout for m16n8k64 kind::mxf4 scale_vec::2X on sm_120a.
// A operand (= BMX4-C's dB) carries a per-(row, 32-K-block) UE8M0 scale; B operand (= dV,
// raw mantissa) is uniform 2^0. Forum-derived SFA rule: the tig in {2,3} threads of each
// quad contribute (16 threads -> 16 rows); byte-id-a=0 takes the low 2 bytes of sA = the
// 2 scales for K-blocks {0,1}. Hypothesis for the row mapping: tig==2 -> row gID, tig==3
// -> row gID+8. Host ref applies A[m][k]*2^scaleA[m][k/32] * B[k][n]; if the mma matches,
// the full FP4 recipe (layout + scale + exactness) is validated.
// build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_mxf4_scale_test.cu -o fp4sc
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)
__host__ __device__ inline uint8_t m11_to_nib(int v){
    switch(v){ case 0:return 0x0; case 1:return 0x2; case 2:return 0x4; case 3:return 0x5;
               case 4:return 0x6; case 6:return 0x7; case -1:return 0xA; case -2:return 0xC;
               case -3:return 0xD; case -4:return 0xE; case -6:return 0xF; } return 0x0; }
__host__ inline int rand_m11(){ static const int V[11]={0,1,2,3,4,6,-1,-2,-3,-4,-6}; return V[rand()%11]; }

// scaleA[m][blk] in {0,1,2,3} for m in 0..15, blk in 0..1 (the two 32-K-blocks)
__global__ void mxf4_tile_sc(const uint8_t* Anib,const uint8_t* Bnib,const uint8_t* scaleA,float* out){
    int lane=threadIdx.x&31; int gID=lane>>2, tig=lane&3;
    uint32_t a[4]={0,0,0,0},b[2]={0,0};
    for(int j=0;j<8;j++){
        int c0=tig*8+j, c32=tig*8+32+j;
        a[0]|=(uint32_t)Anib[(gID)*64+c0 ]<<(4*j);  a[1]|=(uint32_t)Anib[(gID+8)*64+c0 ]<<(4*j);
        a[2]|=(uint32_t)Anib[(gID)*64+c32]<<(4*j);  a[3]|=(uint32_t)Anib[(gID+8)*64+c32]<<(4*j);
        int k0=tig*8+j, k32=tig*8+32+j;
        b[0]|=(uint32_t)Bnib[k0 *8+gID]<<(4*j);     b[1]|=(uint32_t)Bnib[k32*8+gID]<<(4*j);
    }
    // SFA: default every byte to valid 2^0 (0x7F) so no 0x00 (=2^-127) zeroes the tile.
    // Overlay real per-row scales. HYP (env-free A/B via -DCONTRIB): default tig in {2,3}
    // map row gID / gID+8. UE8M0(2^s)=0x7F+s.
    uint32_t sA=0x7F7F7F7Fu;
#ifndef CONTRIB01
    if(tig==2){ int r=gID;   sA=(uint32_t)(0x7F+scaleA[r*2+0]) | ((uint32_t)(0x7F+scaleA[r*2+1])<<8) | 0x7F7F0000u; }
    if(tig==3){ int r=gID+8; sA=(uint32_t)(0x7F+scaleA[r*2+0]) | ((uint32_t)(0x7F+scaleA[r*2+1])<<8) | 0x7F7F0000u; }
#else
    if(tig==0){ int r=gID;   sA=(uint32_t)(0x7F+scaleA[r*2+0]) | ((uint32_t)(0x7F+scaleA[r*2+1])<<8) | 0x7F7F0000u; }
    if(tig==1){ int r=gID+8; sA=(uint32_t)(0x7F+scaleA[r*2+0]) | ((uint32_t)(0x7F+scaleA[r*2+1])<<8) | 0x7F7F0000u; }
#endif
    uint32_t sB=0x7F7F7F7Fu;   // V uniform 2^0
    float d[4]={0,0,0,0};
    asm volatile(
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]),
        "r"(sA),"n"(0),"n"(0), "r"(sB),"n"(0),"n"(0));
    out[(gID)*8+tig*2+0]=d[0]; out[(gID)*8+tig*2+1]=d[1];
    out[(gID+8)*8+tig*2+0]=d[2]; out[(gID+8)*8+tig*2+1]=d[3];
}
int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0)); printf("device: %s sm_%d%d\n",p.name,p.major,p.minor);
    srand(argc>1?atoi(argv[1]):1);
    const int M=16,N=8,K=64;
    int Ai[16*64],Bi[64*8]; uint8_t An[16*64],Bn[64*8],scA[16*2];
    for(int i=0;i<M*K;i++){ Ai[i]=rand_m11(); An[i]=m11_to_nib(Ai[i]); }
    for(int i=0;i<K*N;i++){ Bi[i]=rand_m11(); Bn[i]=m11_to_nib(Bi[i]); }
    for(int i=0;i<M*2;i++) scA[i]=rand()%4;   // scale exponent 0..3 per (row, K-block)
    long long ref[16*8];
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long long s=0;
        for(int k=0;k<K;k++){ int blk=k/32; s+=(long long)Ai[m*K+k]*(1<<scA[m*2+blk])*(long long)Bi[k*N+n]; }
        ref[m*N+n]=s; }
    uint8_t *dA,*dB,*dS; float* dO;
    CK(cudaMalloc(&dA,M*K)); CK(cudaMalloc(&dB,K*N)); CK(cudaMalloc(&dS,M*2)); CK(cudaMalloc(&dO,M*N*4));
    CK(cudaMemcpy(dA,An,M*K,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB,Bn,K*N,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS,scA,M*2,cudaMemcpyHostToDevice));
    mxf4_tile_sc<<<1,32>>>(dA,dB,dS,dO);
    cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){ printf("mma failed: %s\n",cudaGetErrorString(e)); return 3; }
    float ho[16*8]; CK(cudaMemcpy(ho,dO,M*N*4,cudaMemcpyDeviceToHost));
    int mism=0,shown=0;
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long long got=(long long)ho[m*N+n];
        if(got!=ref[m*N+n]){ mism++; if(shown++<10) printf("  MISMATCH C[%2d][%d] mma=%lld ref=%lld  scA[row]={%d,%d}\n",m,n,got,ref[m*N+n],scA[m*2],scA[m*2+1]); } }
    printf("%s  (%d/%d match)\n", mism==0?"=> SCALE LAYOUT CORRECT ✓ (full FP4 recipe validated)":"=> SCALE MISMATCH (iterate row/byte mapping)", M*N-mism, M*N);
    return mism==0?0:1;
}
