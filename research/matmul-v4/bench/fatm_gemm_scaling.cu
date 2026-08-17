// fat-M GEMM scaling: does batching nonces into a fatter M let the B200 use its tensor width?
// M sweep at K=N=8192 int8 (cuBLASLt). If both cards plateau by M~4096, the fat-M ratio is the
// TRUE peak B200/5090 edge -- the best case for numair's residency thesis.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 2;}}while(0)
static double now_ms(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("device: %s sm_%d%d SMs=%d\n",p.name,p.major,p.minor,p.multiProcessorCount);
  int K=8192,N=8192;
  cublasLtHandle_t lt; cublasLtCreate(&lt); void* ws; size_t wss=512<<20; CK(cudaMalloc(&ws,wss));
  int8_t *B; CK(cudaMalloc(&B,(size_t)K*N)); CK(cudaMemset(B,1,(size_t)K*N));
  printf("%8s %12s %10s\n","M","GMAC/s","ms");
  for(int M : {128,512,2048,4096,8192,16384}){
    int8_t *A; int32_t* C; CK(cudaMalloc(&A,(size_t)M*K)); CK(cudaMalloc(&C,(size_t)M*N*4)); CK(cudaMemset(A,1,(size_t)M*K));
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I);
    cublasOperation_t NN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&NN,sizeof(NN));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&NN,sizeof(NN));
    cublasLtMatrixLayout_t lb,la,lc; cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,N,K,N);
    cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K); cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,N,M,N);
    int32_t al=1,be=0; auto g=[&]{return cublasLtMatmul(lt,op,&al,B,lb,A,la,&be,C,lc,C,lc,nullptr,ws,wss,0);};
    if(g()!=CUBLAS_STATUS_SUCCESS){printf("%8d  FAILED\n",M);continue;}
    CK(cudaDeviceSynchronize()); int reps=30; double t=now_ms();
    for(int r=0;r<reps;r++) g(); CK(cudaDeviceSynchronize()); double ms=(now_ms()-t)/reps;
    printf("%8d %12.0f %10.3f\n",M,(1.0*M*K*N/1e9)/(ms/1e3),ms);
    cudaFree(A);cudaFree(C);
  }
  return 0;
}
