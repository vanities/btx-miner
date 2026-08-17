// cuBLASLt FP4-vs-INT8 GEMM THROUGHPUT ceiling. Unlike the hand-PTX
// fp4_int8_ceiling.cu (warp-level mma.sync, sm_120-ONLY), this goes through
// cuBLASLt, which wraps tcgen05 on datacenter Blackwell (sm_100) internally --
// so it RUNS ON THE B200, where the hand-PTX path ptxas-rejects. cuBLASLt serves
// NVFP4 (E2M1 + VEC16_UE4M3), NOT BMX4-C's MXFP4 (UE8M0); but both are 4-bit E2M1
// on the same tensor engine, so for a raw THROUGHPUT ratio NVFP4 is a valid proxy
// for "how fast is this card's FP4 vs its INT8". Real dense GEMM (not a rail).
//
// build: nvcc -O3 -arch=sm_100 cublaslt_fp4_ceiling.cu -lcublasLt -o cbfp4   (B200)
//        nvcc -O3 -arch=sm_120 cublaslt_fp4_ceiling.cu -lcublasLt -o cbfp4   (5090)
#include <cstdio>
#include <cstdint>
#include <string>
#include <cublasLt.h>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

// Timed INT8 (s8->s32) GEMM: D[MxN] = A[MxK]*B[KxN], TN. Returns TOPS or 0.
static double time_int8(cublasLtHandle_t lt, cudaStream_t s, void* ws, size_t wsz,
                        uint32_t M, uint32_t N, uint32_t K, int reps){
  int8_t *dA,*dB; int32_t* dD;
  if(cudaMalloc(&dA,(size_t)M*K)||cudaMalloc(&dB,(size_t)N*K)||cudaMalloc(&dD,(size_t)M*N*4)) return 0;
  cudaMemset(dA,1,(size_t)M*K); cudaMemset(dB,1,(size_t)N*K);
  cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t la=nullptr,lb=nullptr,ld=nullptr; cublasLtMatmulPreference_t pref=nullptr;
  double tops=0;
  do{
    if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I)) break;
    cublasOperation_t opt=CUBLAS_OP_T,opn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opt,sizeof(opt));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof(opn));
    if(cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K)||cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,K,N,K)||
       cublasLtMatrixLayoutCreate(&ld,CUDA_R_32I,M,N,M)) break;
    cublasLtMatmulPreferenceCreate(&pref); cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
    cublasLtMatmulHeuristicResult_t h{}; int got=0;
    if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,ld,ld,pref,1,&h,&got)||got==0) break;
    int32_t alpha=1,beta=0;
    cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dD,ld,dD,ld,&h.algo,ws,wsz,s); cudaStreamSynchronize(s);
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a,s);
    for(int r=0;r<reps;r++) cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dD,ld,dD,ld,&h.algo,ws,wsz,s);
    cudaEventRecord(b,s); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    tops = 2.0*M*N*K*reps / (ms*1e-3) / 1e12;
    cudaEventDestroy(a); cudaEventDestroy(b);
  }while(0);
  if(pref)cublasLtMatmulPreferenceDestroy(pref); if(ld)cublasLtMatrixLayoutDestroy(ld);
  if(lb)cublasLtMatrixLayoutDestroy(lb); if(la)cublasLtMatrixLayoutDestroy(la); if(op)cublasLtMatmulDescDestroy(op);
  cudaFree(dA);cudaFree(dB);cudaFree(dD);
  return tops;
}

// Timed NVFP4 (E2M1 + VEC16_UE4M3) GEMM. Returns TOPS or 0 (unavailable).
static double time_nvfp4(cublasLtHandle_t lt, cudaStream_t s, void* ws, size_t wsz,
                         uint32_t M, uint32_t N, uint32_t K, int reps){
  uint8_t *dA,*dB,*dSF; float* dD;
  if(cudaMalloc(&dA,(size_t)M*K/2)||cudaMalloc(&dB,(size_t)N*K/2)||cudaMalloc(&dD,(size_t)M*N*4)) return 0;
  size_t kb=(K+15)/16, outer=(M>N?M:N); size_t sb=(((outer+127)/128)*128)*(((kb+3)/4)*4);
  if(cudaMalloc(&dSF, sb?sb:256)) return 0;
  cudaMemset(dA,0x11,(size_t)M*K/2); cudaMemset(dB,0x11,(size_t)N*K/2); cudaMemset(dSF,0x7F,sb?sb:256);
  cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t la=nullptr,lb=nullptr,ld=nullptr; cublasLtMatmulPreference_t pref=nullptr;
  double tops=0;
  do{
    if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)) break;
    cublasOperation_t opt=CUBLAS_OP_T,opn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opt,sizeof(opt));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof(opn));
    cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    if(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dSF,sizeof(dSF))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dSF,sizeof(dSF))) break;
    if(cublasLtMatrixLayoutCreate(&la,CUDA_R_4F_E2M1,K,M,K)||cublasLtMatrixLayoutCreate(&lb,CUDA_R_4F_E2M1,K,N,K)||
       cublasLtMatrixLayoutCreate(&ld,CUDA_R_32F,M,N,M)) break;
    cublasLtMatmulPreferenceCreate(&pref); cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
    cublasLtMatmulHeuristicResult_t h{}; int got=0;
    if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,ld,ld,pref,1,&h,&got)||got==0) break;
    float alpha=1.f,beta=0.f;
    if(cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dD,ld,dD,ld,&h.algo,ws,wsz,s)) break; cudaStreamSynchronize(s);
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a,s);
    for(int r=0;r<reps;r++) cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dD,ld,dD,ld,&h.algo,ws,wsz,s);
    cudaEventRecord(b,s); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    tops = 2.0*M*N*K*reps / (ms*1e-3) / 1e12;
    cudaEventDestroy(a); cudaEventDestroy(b);
  }while(0);
  if(pref)cublasLtMatmulPreferenceDestroy(pref); if(ld)cublasLtMatrixLayoutDestroy(ld);
  if(lb)cublasLtMatrixLayoutDestroy(lb); if(la)cublasLtMatrixLayoutDestroy(la); if(op)cublasLtMatmulDescDestroy(op);
  cudaFree(dA);cudaFree(dB);cudaFree(dD);cudaFree(dSF);
  return tops;
}

// Timed MXFP4 (E2M1 + VEC32_UE8M0) GEMM -- the EPISODE's actual FP4 format (block 32, UE8M0
// scale). Returns TOPS or 0 (unavailable). Scale block size 32 along K.
static double time_mxfp4(cublasLtHandle_t lt, cudaStream_t s, void* ws, size_t wsz,
                         uint32_t M, uint32_t N, uint32_t K, int reps){
  uint8_t *dA,*dB,*dSFa,*dSFb; float* dD;
  if(cudaMalloc(&dA,(size_t)M*K/2)||cudaMalloc(&dB,(size_t)N*K/2)||cudaMalloc(&dD,(size_t)M*N*4)) return 0;
  size_t kb=(K+31)/32;                       // 32-elem blocks along K
  size_t sba=(((M+127)/128)*128)*(((kb+3)/4)*4), sbb=(((N+127)/128)*128)*(((kb+3)/4)*4);
  if(cudaMalloc(&dSFa, sba?sba:256)||cudaMalloc(&dSFb, sbb?sbb:256)) return 0;
  cudaMemset(dA,0x11,(size_t)M*K/2); cudaMemset(dB,0x11,(size_t)N*K/2);
  cudaMemset(dSFa,0x7F,sba?sba:256); cudaMemset(dSFb,0x7F,sbb?sbb:256);
  cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t la=nullptr,lb=nullptr,ld=nullptr; cublasLtMatmulPreference_t pref=nullptr;
  double tops=0;
  do{
    if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)) break;
    cublasOperation_t opt=CUBLAS_OP_T,opn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opt,sizeof(opt));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof(opn));
    cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dSFa,sizeof(dSFa))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dSFb,sizeof(dSFb))) break;
    if(cublasLtMatrixLayoutCreate(&la,CUDA_R_4F_E2M1,K,M,K)||cublasLtMatrixLayoutCreate(&lb,CUDA_R_4F_E2M1,K,N,K)||
       cublasLtMatrixLayoutCreate(&ld,CUDA_R_32F,M,N,M)) break;
    cublasLtMatmulPreferenceCreate(&pref); cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
    cublasLtMatmulHeuristicResult_t h{}; int got=0;
    if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,ld,ld,pref,1,&h,&got)||got==0) break;
    float alpha=1.f,beta=0.f;
    if(cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dD,ld,dD,ld,&h.algo,ws,wsz,s)) break; cudaStreamSynchronize(s);
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a,s);
    for(int r=0;r<reps;r++) cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dD,ld,dD,ld,&h.algo,ws,wsz,s);
    cudaEventRecord(b,s); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    tops = 2.0*M*N*K*reps / (ms*1e-3) / 1e12;
    cudaEventDestroy(a); cudaEventDestroy(b);
  }while(0);
  if(pref)cublasLtMatmulPreferenceDestroy(pref); if(ld)cublasLtMatrixLayoutDestroy(ld);
  if(lb)cublasLtMatrixLayoutDestroy(lb); if(la)cublasLtMatrixLayoutDestroy(la); if(op)cublasLtMatmulDescDestroy(op);
  cudaFree(dA);cudaFree(dB);cudaFree(dD);cudaFree(dSFa);cudaFree(dSFb);
  return tops;
}

static void run(cublasLtHandle_t lt,cudaStream_t s,void* ws,size_t wsz,
                const char* label,uint32_t M,uint32_t N,uint32_t K,int reps,
                const char* dev,int maj,int min){
  double int8  = time_int8 (lt,s,ws,wsz,M,N,K,reps);
  double nvfp4 = time_nvfp4(lt,s,ws,wsz,M,N,K,reps);
  double mxfp4 = time_mxfp4(lt,s,ws,wsz,M,N,K,reps);
  double rn=(int8>0&&nvfp4>0)?nvfp4/int8:0, rm=(int8>0&&mxfp4>0)?mxfp4/int8:0;
  printf("\n[%s]  M=%u N=%u K=%u\n", label,M,N,K);
  printf("  INT8  (CUDA_R_8I)                  : %8.1f TOPS\n", int8);
  printf("  NVFP4 (E2M1+VEC16_UE4M3)           : %8.1f TOPS   %.2fx INT8%s\n",
         nvfp4, rn, nvfp4>0?"":"  (no algo)");
  printf("  MXFP4 (E2M1+VEC32_UE8M0, EPISODE)  : %8.1f TOPS   %.2fx INT8%s\n",
         mxfp4, rm, mxfp4>0?"":"  (no algo)");
  printf("CSVCBFP4,%s,sm_%d%d,%s,%u,%u,%u,%.1f,%.1f,%.1f,%.3f,%.3f\n",
         dev,maj,min,label,M,N,K,int8,nvfp4,mxfp4,rn,rm);
}

int main(int argc,char**argv){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  const int reps = argc>1?atoi(argv[1]):50;
  printf("device: %s  sm_%d%d  cuBLASLt %zu  reps=%d\n",
         p.name,p.major,p.minor,(size_t)cublasLtGetVersion(),reps);

  cublasLtHandle_t lt; if(cublasLtCreate(&lt)){printf("cublasLtCreate failed\n");return 2;}
  cudaStream_t s; CK(cudaStreamCreate(&s));
  const size_t wsz=(size_t)512<<20; void* ws; CK(cudaMalloc(&ws,wsz));

  // Square saturation baseline + the episode's ACTUAL fused-FFN GEMM shapes (DC dims,
  // b_seq=87552 d_model=4096 d_ff=16384). Fat-M tests whether FP4 headroom survives the
  // real aspect ratio, not just a saturated square.
  run(lt,s,ws,wsz,"square-8192", 8192,8192,8192, reps, p.name,p.major,p.minor);
  run(lt,s,ws,wsz,"ffn-up",   87552,16384,4096,  reps, p.name,p.major,p.minor); // X.W_up : K=d_model
  run(lt,s,ws,wsz,"ffn-down", 87552,4096,16384,  reps, p.name,p.major,p.minor); // H.W_dn : K=d_ff

  cudaFree(ws); cudaStreamDestroy(s); cublasLtDestroy(lt);
  return 0;
}
