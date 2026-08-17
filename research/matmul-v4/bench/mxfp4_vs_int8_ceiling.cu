// mxfp4_vs_int8_ceiling.cu -- RAW throughput microbench: cuBLASLt MXFP4 vs INT8 IMMA at the
// exact RC Profile-1 phase-2 GEMM shapes on this GPU. NO exactness here -- this measures the
// CEILING that decides whether an exact-MX FP4 kernel could ever beat the INT8 path.
//
// Decision logic (exact-MX accumulation, FP32 24-bit limit, |product|<=2304):
//   passes(K) = ceil(K / floor(2^24 / 2304)) = ceil(K / 7281)
//   up-proj   K=d_model=4096  -> 1 FP4 pass
//   down-proj K=d_ff=16384    -> 3 FP4 panels
// FP4 wins a GEMM iff  passes(K) * t_fp4 < t_int8.
//
// build: nvcc -O3 -arch=sm_120 --std=c++17 mxfp4_vs_int8_ceiling.cu -lcublasLt -o mxfp4_ceiling
// run:   ./mxfp4_ceiling            (uses Profile-1 shapes; args override M N K)

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CK(x) do{ auto e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define LK(x) do{ auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLASLt err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)

static cublasLtHandle_t g_lt;
static void* g_ws; static size_t g_wsz = size_t(512)<<20;

// time one matmul config, best-heuristic algo, median of reps
static float time_matmul(cublasLtMatmulDesc_t op, const void* A, cublasLtMatrixLayout_t la,
                         const void* B, cublasLtMatrixLayout_t lb, void* C,
                         cublasLtMatrixLayout_t lc, const void* alpha, const void* beta, int reps)
{
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &g_wsz, sizeof(g_wsz));
    cublasLtMatmulHeuristicResult_t cand[8]; int n=0;
    cublasStatus_t hs=cublasLtMatmulAlgoGetHeuristic(g_lt, op, la, lb, lc, lc, pref, 8, cand, &n);
    cublasLtMatmulPreferenceDestroy(pref);
    if (hs!=CUBLAS_STATUS_SUCCESS){ printf("  (heuristic status %d -- config unsupported)\n",(int)hs); return -1.f; }
    if (n==0){ printf("  (no algo for this config)\n"); return -1.f; }
    // pick fastest candidate by a quick timed probe
    float best=1e30f; int bi=0;
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for(int i=0;i<n;++i){
        if(cublasLtMatmul(g_lt,op,alpha,A,la,B,lb,beta,C,lc,C,lc,&cand[i].algo,g_ws,g_wsz,0)
           !=CUBLAS_STATUS_SUCCESS) continue;
        CK(cudaDeviceSynchronize());
        cudaEventRecord(e0); for(int r=0;r<5;++r)
            cublasLtMatmul(g_lt,op,alpha,A,la,B,lb,beta,C,lc,C,lc,&cand[i].algo,g_ws,g_wsz,0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=5;
        if(ms<best){best=ms;bi=i;}
    }
    // final timed run of the best
    for(int r=0;r<3;++r) cublasLtMatmul(g_lt,op,alpha,A,la,B,lb,beta,C,lc,C,lc,&cand[bi].algo,g_ws,g_wsz,0);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int r=0;r<reps;++r)
        cublasLtMatmul(g_lt,op,alpha,A,la,B,lb,beta,C,lc,C,lc,&cand[bi].algo,g_ws,g_wsz,0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms/reps;
}

// INT8 IMMA: C[m,n](i32) = A[m,k]*B[k,n], row-major -> col-major C'=B'A'
static float bench_int8(int m,int n,int k,int reps){
    int8_t *dA,*dB; int32_t *dC;
    CK(cudaMalloc(&dA,(size_t)m*k)); CK(cudaMalloc(&dB,(size_t)k*n)); CK(cudaMalloc(&dC,(size_t)m*n*4));
    CK(cudaMemset(dA,1,(size_t)m*k)); CK(cudaMemset(dB,1,(size_t)k*n));
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I);
    cublasOperation_t N=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&N,sizeof(N));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N,sizeof(N));
    cublasLtMatrixLayout_t lB,lA,lC;
    cublasLtMatrixLayoutCreate(&lB,CUDA_R_8I,n,k,n);
    cublasLtMatrixLayoutCreate(&lA,CUDA_R_8I,k,m,k);
    cublasLtMatrixLayoutCreate(&lC,CUDA_R_32I,n,m,n);
    int32_t alpha=1,beta=0;
    float ms=time_matmul(op,dB,lB,dA,lA,dC,lC,&alpha,&beta,reps);
    cublasLtMatrixLayoutDestroy(lB);cublasLtMatrixLayoutDestroy(lA);cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatmulDescDestroy(op); cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return ms;
}

// MXFP4: A,B = CUDA_R_4F_E2M1 (2 vals/byte), per-32 UE8M0 block scales, C = fp32, compute 32F.
static float bench_mxfp4(int m,int n,int k,int reps){
    size_t aElems=(size_t)m*k, bElems=(size_t)k*n;
    void *dA,*dB; float* dC;
    uint8_t *dSA,*dSB;
    CK(cudaMalloc(&dA,aElems/2)); CK(cudaMalloc(&dB,bElems/2)); CK(cudaMalloc(&dC,(size_t)m*n*4));
    CK(cudaMalloc(&dSA,(size_t)m*(k/32))); CK(cudaMalloc(&dSB,(size_t)n*(k/32)));
    CK(cudaMemset(dA,0x22,aElems/2)); CK(cudaMemset(dB,0x22,bElems/2));  // benign E2M1 nibbles
    CK(cudaMemset(dSA,127,(size_t)m*(k/32))); CK(cudaMemset(dSB,127,(size_t)n*(k/32)));
    cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatrixLayout_t lB,lA,lC;
    cublasLtMatrixLayoutCreate(&lB,CUDA_R_4F_E2M1,n,k,n);
    cublasLtMatrixLayoutCreate(&lA,CUDA_R_4F_E2M1,k,m,k);
    cublasLtMatrixLayoutCreate(&lC,CUDA_R_32F,n,m,n);
    float alpha=1,beta=0;
    cublasOperation_t N=CUBLAS_OP_N;
    // Attempt A: MXFP4 with UE8M0 block scaling (our exact format).
    float ms=-1.f;
    {
        cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&N,sizeof(N));
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N,sizeof(N));
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm));
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm));
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dSB,sizeof(dSB));
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dSA,sizeof(dSA));
        ms=time_matmul(op,dB,lB,dA,lA,dC,lC,&alpha,&beta,reps);
        cublasLtMatmulDescDestroy(op);
    }
    // Attempt B (fallback): raw FP4 tensor throughput, no block scaling -- still the
    // hardware ceiling we care about (block scaling is applied in-flight, ~free).
    if (ms<0.f){
        printf("  [block-scale path unsupported; measuring raw FP4 throughput] ");
        cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&N,sizeof(N));
        cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N,sizeof(N));
        ms=time_matmul(op,dB,lB,dA,lA,dC,lC,&alpha,&beta,reps);
        cublasLtMatmulDescDestroy(op);
    }
    cublasLtMatrixLayoutDestroy(lB);cublasLtMatrixLayoutDestroy(lA);cublasLtMatrixLayoutDestroy(lC);
    cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dSA);cudaFree(dSB);
    return ms;
}

static void report(const char* name,int m,int n,int k,int fp4_passes,int reps){
    float ti=bench_int8(m,n,k,reps);
    float tf=bench_mxfp4(m,n,k,reps);
    printf("  %-10s M=%d N=%d K=%d  int8=%.3f ms  fp4(1pass)=%.3f ms  ratio=%.2fx\n",
           name,m,n,k,ti,tf, (tf>0)? ti/tf : 0.f);
    if(tf>0){
        float exact_fp4 = fp4_passes*tf;
        printf("             exact-MX needs %d FP4 pass(es) => %.3f ms vs int8 %.3f ms  => FP4 %s\n",
               fp4_passes, exact_fp4, ti, (exact_fp4<ti)?"WINS":"loses");
    }
}

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("mxfp4_vs_int8_ceiling on %s sm_%d%d\n\n",p.name,p.major,p.minor);
    LK(cublasLtCreate(&g_lt)); CK(cudaMalloc(&g_ws,g_wsz));
    const int reps=30;
    if(argc>=4){ int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]);
        report("custom",m,n,k,(k+7280)/7281,reps); return 0; }
    // Profile-1 phase-2 shapes (b_seq=16384, d_model=4096, d_ff=16384)
    report("ffn-up",   16384,16384,4096, 1, reps);   // K=4096 -> 1 exact FP4 pass
    report("ffn-down", 16384,4096, 16384,3, reps);   // K=16384 -> 3 exact FP4 panels
    // phase-1-ish small-K sanity
    report("qk-small", 512, 786432, 128, 1, reps);   // K=128 -> 1 pass (attention scores)
    return 0;
}
