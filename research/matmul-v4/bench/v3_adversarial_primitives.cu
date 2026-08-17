// V4.5 V3 adversarial economics -- primitive measurement (Workstream F/J/K).
//
// Question (his Workstream K): can a static template-scoped (nonce-null) bank guarantee a
// B200-over-5090 economic result when a miner batches unlimited nonces? The bank is the SAME for
// every nonce in a template, so a batched miner regenerates each overflow page ONCE and reuses it
// across all Q nonces -> the streaming penalty per nonce amortises as regen_cost/Q.
//
// To model that we need two measured primitives at the V3 production shape (M=128, K=N=8192):
//   1. per-page GEMM: 128x8192 . 8192x8192 int8 (cuBLASLt) -- the real per-page compute, same on
//      both cards modulo tensor rate. 1536 of these per nonce = 12 TiMAC.
//   2. per-page regenerate: ExpandMxDequant of one 8192x8192 page -- what a 5090 pays for the
//      overflow it cannot hold resident, amortised over Q.
// The Python model then sweeps Q and reports the per-dollar crossover.
//
// build (5090): nvcc -O3 -arch=sm_120 v3_adversarial_primitives.cu -lcublasLt -o v3prim
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)
static double now_ms(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3+ts.tv_nsec/1e6; }

// minimal int8 keystream-ish page fill to stand in for ExpandMxDequant regen cost (SHA/ChaCha
// bound; we already measured the real expand at ~1.47ms/page, this confirms the compute floor).
__global__ void regen_page(int8_t* p, size_t n, uint32_t seed){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    uint32_t x=seed ^ (uint32_t)i; x^=x<<13; x^=x>>17; x^=x<<5;   // cheap mixer (LOWER bound on real expand)
    p[i]=(int8_t)(x&7);
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d  SMs=%d\n",p.name,p.major,p.minor,p.multiProcessorCount);
    const int M=128,K=8192,N=8192;   // V3 per-page GEMM
    const int reps=50;

    // --- per-page int8 GEMM via cuBLASLt (row-major C=A.B -> col-major C'=B'.A') ---
    int8_t *A,*B; int32_t* C;
    CK(cudaMalloc(&A,(size_t)M*K)); CK(cudaMalloc(&B,(size_t)K*N)); CK(cudaMalloc(&C,(size_t)M*N*4));
    CK(cudaMemset(A,1,(size_t)M*K)); CK(cudaMemset(B,1,(size_t)K*N));
    cublasLtHandle_t lt; cublasLtCreate(&lt); void* ws; size_t wss=256<<20; CK(cudaMalloc(&ws,wss));
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I);
    cublasOperation_t NN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&NN,sizeof(NN));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&NN,sizeof(NN));
    cublasLtMatrixLayout_t lb,la,lc; cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,N,K,N);
    cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K); cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,N,M,N);
    int32_t al=1,be=0; auto g=[&]{ return cublasLtMatmul(lt,op,&al,B,lb,A,la,&be,C,lc,C,lc,nullptr,ws,wss,0); };
    if(g()!=CUBLAS_STATUS_SUCCESS){ printf("cublasLt failed (M=%d K=%d N=%d)\n",M,K,N); return 2; }
    CK(cudaDeviceSynchronize());
    double t0=now_ms(); for(int r=0;r<reps;r++) g(); CK(cudaDeviceSynchronize());
    double gemm_ms=(now_ms()-t0)/reps;
    double gmac=1.0*M*K*N/1e9;
    printf("per-page GEMM  M=%d K=%d N=%d : %.4f ms  (%.0f GMAC/s, %.1f GMAC)\n",
           M,K,N,gemm_ms, gmac/(gemm_ms/1e3), gmac);

    // --- per-page regen (int8 8192x8192) LOWER bound; real ExpandMxDequant measured ~1.47ms ---
    int8_t* pg; size_t pn=(size_t)K*N; CK(cudaMalloc(&pg,pn));
    int thr=256, blk=(pn+thr-1)/thr;
    regen_page<<<blk,thr>>>(pg,pn,1); CK(cudaDeviceSynchronize());
    t0=now_ms(); for(int r=0;r<reps;r++) regen_page<<<blk,thr>>>(pg,pn,r); CK(cudaDeviceSynchronize());
    double regen_ms=(now_ms()-t0)/reps;
    printf("per-page regen (int8 %dx%d, cheap-mixer LOWER bound) : %.4f ms  (real expand ~1.47ms)\n",K,N,regen_ms);

    // --- per-nonce totals at V3 ---
    const int PAGES=1536;
    printf("\nV3 per-nonce: %d pages\n", PAGES);
    printf("  compute (1536 x per-page GEMM)         : %8.1f ms  (%.1f TiMAC)\n",
           PAGES*gemm_ms, PAGES*gmac/1024.0);
    printf("  regen ALL 1536 pages (real 1.47ms)     : %8.1f ms\n", PAGES*1.47);
    printf("\n__V3PRIM__ {\"card\":\"%s\",\"sms\":%d,\"gemm_ms\":%.5f,\"regen_ms_lb\":%.5f,\"pages\":%d}\n",
           p.name,p.multiProcessorCount,gemm_ms,regen_ms,PAGES);
    return 0;
}
