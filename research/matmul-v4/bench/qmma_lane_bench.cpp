// qmma_lane_bench.cpp -- the missing FP4 number: time the branch's native MXFP4 QMMA lane
// (TryLaunchRcOzakiMxfp4GemmS8S8Int64Device, SM120_MMA, device-side panel packing) against
// cuBLASLt INT8 IMMA at the RC Profile-1 phase-2 GEMM shapes, on-device operands only.
//
// This is the measurement mxfp4_vs_int8_ceiling.cu could not deliver (cuBLASLt 13.3 rejects
// generic FP4 layouts on sm_120 GB202 with heuristic status 7): the hand-QMMA kernel is the
// only FP4 path that actually runs on this card, so we time IT.
//
// Decision rule (exact-MX accumulation, FP32 24-bit limit, |product|<=2304):
//   passes(K) = ceil(K/7281):  ffn-up K=4096 -> 1, ffn-down K=16384 -> 3.
//   The lane already runs its own panel split internally, so its wall time IS the exact
//   cost -- compare t_fp4_lane directly vs t_int8 per GEMM.
//
// Operands are MX-representable by construction (mu in {0,+-1,+-2,+-3,+-4,+-6} times 2^e,
// e per 32-block along K), i.e. exactly the Extract output distribution; arbitrary int8
// would make the device FactorBlockToMx pack fail by design.
//
// Build/run on the 5090 host: qmma_lane_build.sh (links against the btx-97 build tree).

#include "cuda/matmul_v4_rc_mx_ozaki_native.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// bitcoin-core link contract: every binary provides the translation hook
// (extern first: a bare const definition would get internal linkage)
extern const std::function<std::string(const char*)> G_TRANSLATION_FUN;
const std::function<std::string(const char*)> G_TRANSLATION_FUN = nullptr;

#define CK(x) do{ auto e=(x); if(e!=cudaSuccess){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
#define LK(x) do{ auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLASLt err %d @%d\n",(int)s,__LINE__);exit(1);} }while(0)

static cublasLtHandle_t g_lt;
static void* g_ws; static size_t g_wsz = size_t(512)<<20;

static uint64_t splitmix(uint64_t x){ x+=0x9e3779b97f4a7c15ULL; x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; return x^(x>>31); }

// MX-valid fill: value = mu * 2^e, mu in E2M1 set, e shared per 32-element block along K.
// k_stride==1  -> K contiguous within a row (left operand, rows x K row-major)
// k_stride==cols -> K is the row index    (right operand, K x cols row-major)
static void fill_mx(std::vector<int8_t>& v, uint32_t outer, uint32_t K, uint32_t cols,
                    bool k_contig, uint64_t seed)
{
    static const int8_t mus[11] = {0,1,-1,2,-2,3,-3,4,-4,6,-6};
    for (uint32_t o = 0; o < outer; ++o) {
        for (uint32_t k = 0; k < K; ++k) {
            const uint64_t blk = (uint64_t)o * (K/32 + 1) + (k >> 5);
            const int e = (int)(splitmix(seed ^ blk) & 3);                    // 2^0..2^3
            const int8_t mu = mus[splitmix(seed*0x51ed ^ ((uint64_t)o<<32) ^ k) % 11];
            const size_t idx = k_contig ? (size_t)o * K + k : (size_t)k * cols + o;
            v[idx] = (int8_t)(mu << e);                                       // |v| <= 48
        }
    }
}

// --- INT8 IMMA baseline (same recipe as mxfp4_vs_int8_ceiling.cu) ---
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
    if (hs!=CUBLAS_STATUS_SUCCESS || n==0){ printf("  (int8 heuristic status %d n=%d)\n",(int)hs,n); return -1.f; }
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
    for(int r=0;r<3;++r) cublasLtMatmul(g_lt,op,alpha,A,la,B,lb,beta,C,lc,C,lc,&cand[bi].algo,g_ws,g_wsz,0);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int r=0;r<reps;++r)
        cublasLtMatmul(g_lt,op,alpha,A,la,B,lb,beta,C,lc,C,lc,&cand[bi].algo,g_ws,g_wsz,0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms/reps;
}

static float bench_int8(uint32_t m,uint32_t n,uint32_t k,const int8_t* hA,const int8_t* hB,int reps){
    int8_t *dA,*dB; int32_t *dC;
    CK(cudaMalloc(&dA,(size_t)m*k)); CK(cudaMalloc(&dB,(size_t)k*n)); CK(cudaMalloc(&dC,(size_t)m*n*4));
    CK(cudaMemcpy(dA,hA,(size_t)m*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB,(size_t)k*n,cudaMemcpyHostToDevice));
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

// --- the QMMA lane, device-pointer entry, stream-ordered event timing ---
static float bench_qmma(uint32_t m,uint32_t n,uint32_t k,const int8_t* hA,const int8_t* hB,
                        int reps, const std::vector<int8_t>& vA, const std::vector<int8_t>& vB)
{
    int8_t *dA,*dB; int64_t *dC;
    CK(cudaMalloc(&dA,(size_t)m*k)); CK(cudaMalloc(&dB,(size_t)k*n));
    CK(cudaMalloc(&dC,(size_t)m*n*8));
    CK(cudaMemcpy(dA,hA,(size_t)m*k,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hB,(size_t)k*n,cudaMemcpyHostToDevice));
    std::string err;
    const uint64_t launches0 = matmul_v4::cuda::RcOzakiCudaMxfp4NativeTensorLaunchCount();
    // warm-up (arena growth + first-launch JIT)
    for (int r=0;r<2;++r) {
        if (!matmul_v4::cuda::TryLaunchRcOzakiMxfp4GemmS8S8Int64Device(dA,dB,dC,m,k,n,nullptr,&err)) {
            printf("  QMMA launch FAILED: %s\n", err.c_str());
            cudaFree(dA);cudaFree(dB);cudaFree(dC); return -1.f;
        }
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r=0;r<reps;++r)
        (void)matmul_v4::cuda::TryLaunchRcOzakiMxfp4GemmS8S8Int64Device(dA,dB,dC,m,k,n,nullptr,&err);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=reps;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    const uint64_t launches1 = matmul_v4::cuda::RcOzakiCudaMxfp4NativeTensorLaunchCount();
    printf("  native tensor launches during bench: %" PRIu64 " (scalar-tail %" PRIu64 " total)\n",
           launches1-launches0, matmul_v4::cuda::RcOzakiCudaMxfp4ScalarTailLaunchCount());

    // correctness spot-check vs CPU int64 dot on 4 sampled entries
    static const uint32_t rr[4]={0,1,7,31};
    bool ok=true;
    for (int s=0;s<4;++s){
        uint32_t r = rr[s]%m, c = (uint32_t)(splitmix(s+77)% n);
        int64_t got; CK(cudaMemcpy(&got,dC+(size_t)r*n+c,8,cudaMemcpyDeviceToHost));
        int64_t want=0;
        for (uint32_t kk=0;kk<k;++kk) want += (int64_t)vA[(size_t)r*k+kk]*(int64_t)vB[(size_t)kk*n+c];
        if (got!=want){ printf("  EXACTNESS FAIL [%u,%u] got=%" PRId64 " want=%" PRId64 "\n",r,c,got,want); ok=false; }
    }
    if (ok) printf("  exactness spot-check: 4/4 match int64 oracle\n");
    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return ms;
}

static void report(const char* name,uint32_t m,uint32_t n,uint32_t k,int reps){
    std::vector<int8_t> A((size_t)m*k), B((size_t)k*n);
    fill_mx(A, m, k, k, /*k_contig=*/true,  0xA11CEULL + m + k);
    fill_mx(B, n, k, n, /*k_contig=*/false, 0xB0BULL + n + k);
    printf("%-10s M=%u N=%u K=%u  (exact panels: %u)\n", name, m, n, k, (k+7280)/7281);
    float tf = bench_qmma(m,n,k,A.data(),B.data(),reps,A,B);
    float ti = bench_int8(m,n,k,A.data(),B.data(),reps);
    printf("  int8=%.3f ms  qmma_lane=%.3f ms  fp4:int8 = %.2fx  => FP4 %s this GEMM\n\n",
           ti, tf, (tf>0)? ti/tf : 0.f, (tf>0 && tf<ti) ? "WINS" : "loses");
}

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("qmma_lane_bench on %s sm_%d%d\n", p.name, p.major, p.minor);
    printf("sm120a linked=%d\n", (int)matmul_v4::cuda::RcOzakiMxfp4Sm120aKernelLinked());
    if (!matmul_v4::cuda::SelfQualifyRcOzakiCudaMxfp4Once()) {
        printf("self-qual FAILED: backend=%s deficit=%s\n",
               matmul_v4::cuda::RcOzakiCudaMxfp4Backend().c_str(),
               matmul_v4::cuda::RcOzakiCudaMxfp4Deficit().c_str());
        return 1;
    }
    printf("self-qualified backend=%s arch=%s\n\n",
           matmul_v4::cuda::RcOzakiCudaMxfp4Backend().c_str(),
           matmul_v4::cuda::RcOzakiCudaMxfp4ArchKey().c_str());
    LK(cublasLtCreate(&g_lt)); CK(cudaMalloc(&g_ws,g_wsz));
    const int reps = (argc>1)? atoi(argv[1]) : 20;
    // RC Profile-1 phase-2 shapes (b_seq=16384, d_model=4096, d_ff=16384)
    report("ffn-up",   16384,16384,4096,  reps);
    report("ffn-down", 16384,4096, 16384, reps);
    return 0;
}
