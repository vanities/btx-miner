// Byte-exact tiled native FP4 GEMM vs cuBLASLt INT8 -- the real benchmark.
//
// Gates 1+2 proved a single mma.kind::mxf4 tile is byte-exact to int8 on M11 operands. This tiles
// it into a full C(MxN)=A(MxK).B(KxN) with chunked FP32->int32 draining (partial sums held under
// 2^24), validates byte-exact against a host int8 reference at a small size, then times it at the
// RC phase-2 shape against cuBLASLt s8s8.
//
// HONEST EXPECTATION: cuBLASLt int8 is heavily tuned; this hand kernel has the 1.98x raw FP4
// tensor rate but not cuBLASLt's smem pipelining, so v1 may NOT beat it end-to-end. The number
// tells us how much tuning the gap needs -- and byte-exactness holds regardless.
//
// Simplification for this bench: operands are already the M11 MANTISSAS (E2M1-exact), block
// scales unit. That isolates the GEMM engine; the A-K-scale (block_scale) + B-free-scale
// (post-multiply) plumbing is proven in gate 1 and folds in at integration.
//
// build (5090): nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_gemm_tiled.cu -lcublasLt -o fp4t
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

__host__ __device__ static inline uint8_t e2m1_code(int v){
    int m=v<0?-v:v; uint8_t c;
    switch(m){case 0:c=0;break;case 1:c=0b010;break;case 2:c=0b100;break;
              case 3:c=0b101;break;case 4:c=0b110;break;case 6:c=0b111;break;default:c=0;break;}
    return v<0?(c|0b1000):c;
}
static const int8_t M11V[8]={1,2,3,4,6,-1,-2,-3};

// Pack an MxK int8-M11 matrix (row-major) into E2M1 nibbles (row-major, 2 vals/byte).
static void pack_e2m1(const std::vector<int8_t>& v, std::vector<uint8_t>& out, int M,int K){
    out.assign((size_t)M*K/2,0);
    for(size_t i=0;i<(size_t)M*K;i++){ uint8_t n=e2m1_code(v[i]); if(i&1) out[i/2]|=n<<4; else out[i/2]|=n; }
}

// One block computes a 16(m) x 8(n) C tile; grid covers MxN. K looped in 64-steps, drained to
// int32 every DRAIN steps so the FP32 accumulator never passes 2^24 (max |partial| per k =
// 6*6=36, so 2^24/36 ~ 466k k-elements; DRAIN=4096 k-steps = 262144 k >> safe... use 8192 k).
#define KDRAIN 8192
__global__ void fp4_gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                         int32_t* __restrict__ C, int M,int N,int K){
    const int mt = blockIdx.y*16, nt = blockIdx.x*8;
    if(mt>=M||nt>=N) return;
    const int t=threadIdx.x, gr=t/4, tc=t%4;
    int32_t acc[4]={0,0,0,0};
    float d[4]={0,0,0,0};
    const uint32_t sU=0x7F7F7F7Fu;
    int since=0;
    for(int k0=0;k0<K;k0+=64){
        // pack A(16x64) rows mt+gr, mt+gr+8 ; B(64x8) col nt+gr
        auto pkA=[&](int row,int kb)->uint32_t{ uint32_t r=0;
            const uint8_t* p=A+((size_t)row*K + kb)/2;
            #pragma unroll
            for(int i=0;i<8;i++){ int kk=kb+i; uint8_t byte=A[((size_t)row*K+kk)/2]; uint8_t n=(kk&1)?(byte>>4):(byte&0xF); r|=(uint32_t)n<<(i*4);} return r; };
        auto pkB=[&](int col,int kb)->uint32_t{ uint32_t r=0;
            #pragma unroll
            for(int i=0;i<8;i++){ int kk=kb+i; uint8_t byte=B[((size_t)kk*N+col)/2]; uint8_t n=(col&1)?(byte>>4):(byte&0xF); r|=(uint32_t)n<<(i*4);} return r; };
        uint32_t a0=pkA(mt+gr,k0+tc*8), a1=pkA(mt+gr+8,k0+tc*8), a2=pkA(mt+gr,k0+tc*8+32), a3=pkA(mt+gr+8,k0+tc*8+32);
        uint32_t b0=pkB(nt+gr,k0+tc*8), b1=pkB(nt+gr,k0+tc*8+32);
        asm volatile(
          "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
          : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
            "r"(sU),"n"(0),"n"(0), "r"(sU),"n"(0),"n"(0));
        since+=64;
        if(since>=KDRAIN){ // drain FP32 -> int32, reset
            #pragma unroll
            for(int i=0;i<4;i++){ acc[i]+=(int32_t)llrintf(d[i]); d[i]=0.0f; } since=0;
        }
    }
    #pragma unroll
    for(int i=0;i<4;i++) acc[i]+=(int32_t)llrintf(d[i]);
    C[(size_t)(mt+gr)*N   + nt+tc*2]   = acc[0];
    C[(size_t)(mt+gr)*N   + nt+tc*2+1] = acc[1];
    C[(size_t)(mt+gr+8)*N + nt+tc*2]   = acc[2];
    C[(size_t)(mt+gr+8)*N + nt+tc*2+1] = acc[3];
}

static double now_ms(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3+ts.tv_nsec/1e6; }

int main(int argc,char**argv){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d\n",p.name,p.major,p.minor);

    // --- correctness at a small shape ---
    { int M=64,N=32,K=256;
      std::vector<int8_t> A(M*K),B(K*N);
      for(size_t i=0;i<A.size();i++) A[i]=M11V[i&7];
      for(size_t i=0;i<B.size();i++) B[i]=M11V[(i*3+1)&7];
      std::vector<int32_t> ref(M*N,0);
      for(int i=0;i<M;i++)for(int j=0;j<N;j++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[i*K+k]*B[k*N+j]; ref[i*N+j]=(int32_t)s; }
      std::vector<uint8_t> Ap,Bp; pack_e2m1(A,Ap,M,K); pack_e2m1(B,Bp,K,N);
      uint8_t *dA,*dB; int32_t* dC; CK(cudaMalloc(&dA,Ap.size())); CK(cudaMalloc(&dB,Bp.size())); CK(cudaMalloc(&dC,(size_t)M*N*4));
      CK(cudaMemcpy(dA,Ap.data(),Ap.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB,Bp.data(),Bp.size(),cudaMemcpyHostToDevice));
      dim3 g(N/8,M/16); fp4_gemm<<<g,32>>>(dA,dB,dC,M,N,K);
      cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){printf("launch %s\n",cudaGetErrorString(e));return 2;}
      std::vector<int32_t> got(M*N); CK(cudaMemcpy(got.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
      int fails=0; for(int i=0;i<M*N;i++) if(got[i]!=ref[i]) fails++;
      printf("correctness %dx%dx%d : %s (%d/%d mismatched)\n",M,N,K, fails==0?"BYTE-EXACT":"WRONG", fails, M*N);
      cudaFree(dA);cudaFree(dB);cudaFree(dC);
      if(fails){ printf("\nGEMM NOT byte-exact -- stop.\n"); return 1; }
    }

    // --- throughput at RC phase-2 shape ---
    int M=16384,N=4096,K=4096;
    std::vector<int8_t> A(M*K),B(K*N);
    for(size_t i=0;i<A.size();i++) A[i]=M11V[i&7];
    for(size_t i=0;i<B.size();i++) B[i]=M11V[(i*3+1)&7];
    std::vector<uint8_t> Ap,Bp; pack_e2m1(A,Ap,M,K); pack_e2m1(B,Bp,K,N);
    uint8_t *dA,*dB; int32_t* dC; CK(cudaMalloc(&dA,Ap.size())); CK(cudaMalloc(&dB,Bp.size())); CK(cudaMalloc(&dC,(size_t)M*N*4));
    CK(cudaMemcpy(dA,Ap.data(),Ap.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB,Bp.data(),Bp.size(),cudaMemcpyHostToDevice));
    dim3 g(N/8,M/16);
    fp4_gemm<<<g,32>>>(dA,dB,dC,M,N,K); CK(cudaDeviceSynchronize());
    const int reps=20; double t0=now_ms();
    for(int r=0;r<reps;r++) fp4_gemm<<<g,32>>>(dA,dB,dC,M,N,K);
    CK(cudaDeviceSynchronize()); double fp4ms=(now_ms()-t0)/reps;

    // cuBLASLt int8 baseline (same shape). row-major C=A.B -> col-major C'=B'.A'
    int8_t *iA,*iB; int32_t* iC; CK(cudaMalloc(&iA,(size_t)M*K)); CK(cudaMalloc(&iB,(size_t)K*N)); CK(cudaMalloc(&iC,(size_t)M*N*4));
    CK(cudaMemcpy(iA,A.data(),(size_t)M*K,cudaMemcpyHostToDevice)); CK(cudaMemcpy(iB,B.data(),(size_t)K*N,cudaMemcpyHostToDevice));
    cublasLtHandle_t lt; cublasLtCreate(&lt); void* ws; size_t wss=256<<20; CK(cudaMalloc(&ws,wss));
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I);
    cublasOperation_t NN=CUBLAS_OP_N; cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&NN,sizeof(NN));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&NN,sizeof(NN));
    cublasLtMatrixLayout_t lb,la,lc; cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,N,K,N);
    cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K); cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,N,M,N);
    int32_t al=1,be=0;
    auto lt_once=[&]{ return cublasLtMatmul(lt,op,&al,iB,lb,iA,la,&be,iC,lc,iC,lc,nullptr,ws,wss,0); };
    if(lt_once()!=CUBLAS_STATUS_SUCCESS){ printf("cublasLt failed\n"); return 2; }
    CK(cudaDeviceSynchronize()); t0=now_ms();
    for(int r=0;r<reps;r++) lt_once(); CK(cudaDeviceSynchronize()); double i8ms=(now_ms()-t0)/reps;

    double gflop=2.0*M*N*K/1e9;
    printf("\nRC phase-2 shape %dx%dx%d (%.0f GMAC)\n",M,N,K,gflop/2);
    printf("  FP4 hand kernel : %8.2f ms  (%6.0f GMAC/s)\n", fp4ms, (gflop/2)/(fp4ms/1e3));
    printf("  cuBLASLt INT8   : %8.2f ms  (%6.0f GMAC/s)\n", i8ms, (gflop/2)/(i8ms/1e3));
    printf("  FP4 : INT8 wall = %.2fx  %s\n", i8ms/fp4ms,
           i8ms/fp4ms>1.05?"(FP4 wins)": i8ms/fp4ms>0.95?"(tie)":"(cuBLASLt int8 still faster -- FP4 kernel needs tuning)");
    return 0;
}
