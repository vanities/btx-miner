// END-TO-END test of the HYPOTHETICAL scale-axis FIX: does the swapped E8M0 indexing
// make BMX4-C's S2 (Q = B*V) computable byte-exactly by the 5090's kind::mxf4 tensor
// cores at the REAL deploy shape (M=n=4096, N=m=1024, K=n=4096)?
//
// Models the POST-FIX consensus (NOT current bytes -- see FINDINGS-fp4-scale-axis.md):
//   B[k][j] = mu_B[k][j] << sc[k*(n/32) + j/32]   <- SWAPPED: per-row k, per-32 along j=K
//   V raw M11 (unchanged).
// Ground truth = the validated row-major cuBLASLt INT8 path on the dequantized s8
// operands (exactly what the solver's INT8 path would produce for these bytes).
// FP4 path  = hand tiled kind::mxf4 GEMM: packed E2M1 nibbles + hw SFA block scales
// (the recipe validated in fp4_mxf4_tile_test / fp4_mxf4_scale_test), f32 accum.
// |B|<=48, |V|<=6 -> |Q| <= 288*n = 1,179,648 < 2^24, so every f32 partial sum is an
// exact integer: if the layouts are right the two paths are BIT-identical as integers.
//
// build: nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_mxf4_swapped_e2e.cu -lcublasLt -o fp4e2e
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));return 2;} }while(0)

static inline uint8_t m11_to_nib(int v){
    switch(v){ case 0:return 0x0; case 1:return 0x2; case 2:return 0x4; case 3:return 0x5;
               case 4:return 0x6; case 6:return 0x7; case -1:return 0xA; case -2:return 0xC;
               case -3:return 0xD; case -4:return 0xE; case -6:return 0xF; } return 0x0; }
static inline int rand_m11(){ static const int V[11]={0,1,2,3,4,6,-1,-2,-3,-4,-6}; return V[rand()%11]; }

// ---- validated row-major INT8 GEMM (verbatim from c13_bench, byte-exact in --gate) ----
static bool int8_gemm(cublasLtHandle_t lt,cudaStream_t s,void*ws,size_t wsz,
               const int8_t*dA,const int8_t*dB,int32_t*dC,uint32_t M,uint32_t N,uint32_t Kd){
    cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;
    cublasLtMatmulPreference_t pref=nullptr; bool ok=false;
    do{
        if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I))break;
        cublasOperation_t opn=CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opn,sizeof(opn));
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof(opn));
        cublasLtOrder_t row=CUBLASLT_ORDER_ROW;
        auto mk=[&](cublasLtMatrixLayout_t*L,cudaDataType t,uint64_t r,uint64_t c,int64_t ld){
            if(cublasLtMatrixLayoutCreate(L,t,r,c,ld))return false;
            return 0==cublasLtMatrixLayoutSetAttribute(*L,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row));};
        if(!mk(&la,CUDA_R_8I,M,Kd,Kd)||!mk(&lb,CUDA_R_8I,Kd,N,N)||!mk(&lc,CUDA_R_32I,M,N,N))break;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
        cublasLtMatmulHeuristicResult_t hr{}; int got=0;
        if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pref,1,&hr,&got)||got==0){
            printf("  [cublasLt] no INT8 algo for %ux%ux%u\n",M,N,Kd); break;}
        int32_t alpha=1,beta=0;
        cublasStatus_t st=cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dC,lc,dC,lc,&hr.algo,ws,wsz,s);
        if(st){printf("  [cublasLt] matmul fail status=%d\n",(int)st);break;}
        ok=true;
    }while(0);
    if(pref)cublasLtMatmulPreferenceDestroy(pref);
    if(lc)cublasLtMatrixLayoutDestroy(lc); if(lb)cublasLtMatrixLayoutDestroy(lb); if(la)cublasLtMatrixLayoutDestroy(la);
    if(op)cublasLtMatmulDescDestroy(op);
    return ok;
}

// ---- the hand mxf4 tiled GEMM (naive global-load version: correctness floor) ----
// Q[M x N] = B[M x K] * V[K x N].  Bnib: row-major packed nibbles (col j at nibble j).
// Vtnib: V TRANSPOSED, row-major packed (row c = output col, nibble j = K position),
// so both fragment loads are single aligned u32 loads. SFAb: 0x7F+exp, [row][j/32].
// One warp per 16x8 output tile, K-loop in 64-chunks, accum in registers.
__global__ void mxf4_gemm(const uint8_t* __restrict__ Bnib, const uint8_t* __restrict__ Vtnib,
                          const uint8_t* __restrict__ SFAb, float* __restrict__ Q,
                          uint32_t M, uint32_t N, uint32_t K){
    const uint32_t lane=threadIdx.x&31u, warp=threadIdx.x>>5;
    const uint32_t Nt=N/8, w=blockIdx.x*(blockDim.x>>5)+warp;
    if(w>=(M/16)*Nt) return;
    const uint32_t tm=w/Nt, tn=w%Nt, r0=tm*16, c0=tn*8;
    const uint32_t gID=lane>>2, tig=lane&3;
    const uint32_t nblk=K/32;
    const size_t rowB0=(size_t)(r0+gID)*K/2, rowB8=(size_t)(r0+gID+8)*K/2;   // byte offsets
    const size_t rowV =(size_t)(c0+gID)*K/2;
    float d0=0.f,d1=0.f,d2=0.f,d3=0.f;
    for(uint32_t kb=0;kb<K/64;kb++){
        const uint32_t j0=kb*64, bo=(j0+tig*8)>>1;                 // byte offset of this thread's 8 nibbles
        uint32_t a0=*(const uint32_t*)(Bnib+rowB0+bo);
        uint32_t a1=*(const uint32_t*)(Bnib+rowB8+bo);
        uint32_t a2=*(const uint32_t*)(Bnib+rowB0+bo+16);          // +32 cols = +16 bytes
        uint32_t a3=*(const uint32_t*)(Bnib+rowB8+bo+16);
        uint32_t b0=*(const uint32_t*)(Vtnib+rowV+bo);
        uint32_t b1=*(const uint32_t*)(Vtnib+rowV+bo+16);
        uint32_t sA=0x7F7F7F7Fu;
        if(tig==0){ const uint8_t* p=SFAb+(size_t)(r0+gID)*nblk+kb*2;   sA=(uint32_t)p[0]|((uint32_t)p[1]<<8)|0x7F7F0000u; }
        if(tig==1){ const uint8_t* p=SFAb+(size_t)(r0+gID+8)*nblk+kb*2; sA=(uint32_t)p[0]|((uint32_t)p[1]<<8)|0x7F7F0000u; }
        const uint32_t sB=0x7F7F7F7Fu;                              // V raw M11: uniform 2^0
        asm volatile(
          "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
          : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
            "r"(sA),"n"(0),"n"(0), "r"(sB),"n"(0),"n"(0));
    }
    Q[(size_t)(r0+gID)  *N + c0+tig*2+0]=d0;
    Q[(size_t)(r0+gID)  *N + c0+tig*2+1]=d1;
    Q[(size_t)(r0+gID+8)*N + c0+tig*2+0]=d2;
    Q[(size_t)(r0+gID+8)*N + c0+tig*2+1]=d3;
}

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    const uint32_t n=argc>1?atoi(argv[1]):4096, m=argc>2?atoi(argv[2]):1024;
    const uint32_t M=n,N=m,K=n,nblk=n/32;
    const int seed=argc>3?atoi(argv[3]):1; srand(seed);
    printf("device: %s sm_%d%d | SWAPPED-consensus S2: Q[%u x %u] = B[%u x %u]*V, K=%u, seed=%d\n",
           p.name,p.major,p.minor,M,N,M,K,K,seed);

    // ---- host operand construction (post-fix consensus semantics) ----
    std::vector<int8_t>  muB((size_t)M*K), muV((size_t)K*N);
    std::vector<uint8_t> sc((size_t)M*nblk);
    for(auto& v:muB) v=(int8_t)rand_m11();
    for(auto& v:muV) v=(int8_t)rand_m11();
    for(auto& v:sc)  v=(uint8_t)(rand()%4);
    // dequantized s8 (ground-truth input): B[k][j] = mu << sc[k][j/32]  (SWAPPED axis)
    std::vector<int8_t> Bs8((size_t)M*K);
    for(uint32_t k=0;k<M;k++) for(uint32_t j=0;j<K;j++)
        Bs8[(size_t)k*K+j]=(int8_t)((int32_t)muB[(size_t)k*K+j]<<sc[(size_t)k*nblk+j/32]);
    // packed nibbles for the FP4 path
    std::vector<uint8_t> Bnib((size_t)M*K/2,0), Vtnib((size_t)N*K/2,0), SFAb((size_t)M*nblk);
    for(uint32_t k=0;k<M;k++) for(uint32_t j=0;j<K;j++){
        uint8_t nib=m11_to_nib(muB[(size_t)k*K+j]); size_t pos=(size_t)k*K+j;
        Bnib[pos>>1]|= (pos&1)? (nib<<4):nib; }
    for(uint32_t c=0;c<N;c++) for(uint32_t j=0;j<K;j++){       // V^T[c][j] = V[j][c]
        uint8_t nib=m11_to_nib(muV[(size_t)j*N+c]); size_t pos=(size_t)c*K+j;
        Vtnib[pos>>1]|= (pos&1)? (nib<<4):nib; }
    for(size_t i=0;i<SFAb.size();i++) SFAb[i]=(uint8_t)(0x7F+sc[i]);

    // ---- device buffers ----
    int8_t *dBs8,*dVs8; int32_t* dQref; uint8_t *dBn,*dVn,*dSF; float* dQf;
    CK(cudaMalloc(&dBs8,(size_t)M*K)); CK(cudaMalloc(&dVs8,(size_t)K*N)); CK(cudaMalloc(&dQref,(size_t)M*N*4));
    CK(cudaMalloc(&dBn,Bnib.size())); CK(cudaMalloc(&dVn,Vtnib.size())); CK(cudaMalloc(&dSF,SFAb.size()));
    CK(cudaMalloc(&dQf,(size_t)M*N*4));
    CK(cudaMemcpy(dBs8,Bs8.data(),Bs8.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVs8,muV.data(),muV.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBn,Bnib.data(),Bnib.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dVn,Vtnib.data(),Vtnib.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dSF,SFAb.data(),SFAb.size(),cudaMemcpyHostToDevice));

    cublasLtHandle_t lt; if(cublasLtCreate(&lt)){printf("ltCreate fail\n");return 2;}
    cudaStream_t s; CK(cudaStreamCreate(&s));
    void* ws; CK(cudaMalloc(&ws,(size_t)64<<20));

    // ---- ground truth: INT8 (the validated consensus-equivalent path) ----
    if(!int8_gemm(lt,s,ws,(size_t)64<<20,dBs8,dVs8,dQref,M,N,K)) return 3;
    CK(cudaStreamSynchronize(s));

    // ---- FP4 path ----
    const uint32_t warps=(M/16)*(N/8), thr=256, blocks=(warps*32+thr-1)/thr;
    mxf4_gemm<<<blocks,thr,0,s>>>(dBn,dVn,dSF,dQf,M,N,K);
    cudaError_t e=cudaStreamSynchronize(s);
    if(e!=cudaSuccess){ printf("mxf4 kernel failed: %s\n",cudaGetErrorString(e)); return 3; }

    // ---- byte-exact compare (f32 values are exact ints < 2^24) ----
    std::vector<int32_t> Qr((size_t)M*N); std::vector<float> Qf((size_t)M*N);
    CK(cudaMemcpy(Qr.data(),dQref,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(Qf.data(),dQf,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    size_t mism=0,shown=0;
    for(size_t i=0;i<(size_t)M*N;i++){
        long long g=llrintf(Qf[i]);
        if(g!=(long long)Qr[i]){ mism++; if(shown++<6) printf("  MISM [%zu] fp4=%lld int8=%d\n",i,g,Qr[i]); }
    }
    printf("%s (%zu/%u elements match)\n", mism? "=> MISMATCH":"=> BYTE-EXACT MATCH ✓ (post-fix consensus is hw-FP4-computable)",
           (size_t)M*N-mism,(unsigned)(M*N));

    // ---- timing (floor: naive global-load kernel; cuBLASLt call incl. desc setup) ----
    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    const int reps=20;
    CK(cudaEventRecord(t0,s));
    for(int r=0;r<reps;r++) mxf4_gemm<<<blocks,thr,0,s>>>(dBn,dVn,dSF,dQf,M,N,K);
    CK(cudaEventRecord(t1,s)); CK(cudaEventSynchronize(t1));
    float ms=0; CK(cudaEventElapsedTime(&ms,t0,t1)); ms/=reps;
    double tops=2.0*M*N*K/(ms*1e-3)/1e12;
    printf(" mxf4 naive-tiled : %7.3f ms  %6.1f TOPS  (floor: no smem staging/double-buffer)\n",ms,tops);
    CK(cudaEventRecord(t0,s));
    int8_gemm(lt,s,ws,(size_t)64<<20,dBs8,dVs8,dQref,M,N,K);
    CK(cudaEventRecord(t1,s)); CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms,t0,t1));
    printf(" cuBLASLt INT8    : %7.3f ms  %6.1f TOPS  (incl. per-call desc setup; plan-cached S2 measures ~0.067ms)\n",
           ms,2.0*M*N*K/(ms*1e-3)/1e12);
    return mism?1:0;
}
