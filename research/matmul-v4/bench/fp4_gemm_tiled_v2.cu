// Byte-exact tiled FP4 GEMM v2 -- smem-staged, register-blocked. vs v1 (23.5ms, no smem,
// byte-by-byte global nibble reads, one 16x8 tile/block) this:
//   * stages A(BMxBK) and B(BKxBN) tiles into shared memory with coalesced vectorized loads
//   * each block computes a BM x BN output tile with 4 warps, each warp holding 8 mma tiles in
//     registers -> the smem load is amortised across 8 mmas x (BK/64) k-steps
//   * drains FP32->int32 once per mainloop iter (max partial BK*36 << 2^24)
// Still hand-rolled (no TMA / no double-buffer), so it will not match CUTLASS -- but it should
// close most of the 27x gap and show whether FP4 can approach cuBLASLt int8.
//
// build (5090): nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_gemm_tiled_v2.cu -lcublasLt -o fp4t2
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
static void pack_e2m1(const std::vector<int8_t>& v, std::vector<uint8_t>& out,int M,int K){
    out.assign((size_t)M*K/2,0);
    for(size_t i=0;i<(size_t)M*K;i++){ uint8_t n=e2m1_code(v[i]); if(i&1) out[i/2]|=n<<4; else out[i/2]|=n; }
}

// tile params
#define BM 64
#define BN 64
#define BK 128        // per mainloop iter; 2 k-steps of 64
#define THREADS 128   // 4 warps
// smem holds packed nibbles: As[BM][BK/2 bytes], Bs[BK][BN/2 bytes]
__global__ __launch_bounds__(THREADS) void fp4_gemm2(const uint8_t* __restrict__ A,
                        const uint8_t* __restrict__ B, int32_t* __restrict__ C, int M,int N,int K){
    __shared__ uint8_t As[BM*(BK/2)];
    __shared__ uint8_t Bs[BK*(BN/2)];
    const int t=threadIdx.x;
    const int wr=(t/32)/2, wc=(t/32)%2;     // 2x2 warp grid
    const int gr=(t%32)/4, tc=(t%32)%4;
    const int blockM=blockIdx.y*BM, blockN=blockIdx.x*BN;

    int32_t acc[8][4]; float d[8][4];        // 8 mma tiles/warp (2 rows x 4 cols of 16x8)
    #pragma unroll
    for(int i=0;i<8;i++)
        #pragma unroll
        for(int j=0;j<4;j++){ acc[i][j]=0; d[i][j]=0.0f; }
    const uint32_t sU=0x7F7F7F7Fu;

    for(int k0=0;k0<K;k0+=BK){
        // --- coalesced load A[BM][BK] -> As (packed), B[BK][BN] -> Bs (packed) ---
        // As: BM*BK/2 = 64*64 = 4096 bytes; load as uint4 (16B) chunks.
        for(int off=t*16; off<BM*(BK/2); off+=THREADS*16){
            int row=off/(BK/2), byte=off%(BK/2);
            *(uint4*)&As[off] = *(const uint4*)&A[((size_t)(blockM+row)*K + k0)/2 + byte];
        }
        for(int off=t*16; off<BK*(BN/2); off+=THREADS*16){
            int kk=off/(BN/2), byte=off%(BN/2);
            *(uint4*)&Bs[off] = *(const uint4*)&B[((size_t)(k0+kk)*N + blockN)/2 + byte];
        }
        __syncthreads();

        // --- 2 k-steps of 64 within BK ---
        #pragma unroll
        for(int ks=0; ks<BK; ks+=64){
            // pack fragment from smem: 8 nibbles at (row, kbase) in As / (kbase,col) in Bs
            auto Anib=[&](int row,int kb)->uint32_t{ uint32_t r=0;
                #pragma unroll
                for(int i=0;i<8;i++){ int kk=kb+i; uint8_t by=As[row*(BK/2)+kk/2]; uint8_t n=(kk&1)?(by>>4):(by&0xF); r|=(uint32_t)n<<(i*4);} return r; };
            auto Bnib=[&](int col,int kb)->uint32_t{ uint32_t r=0;
                #pragma unroll
                for(int i=0;i<8;i++){ int kk=kb+i; uint8_t by=Bs[kk*(BN/2)+col/2]; uint8_t n=(col&1)?(by>>4):(by&0xF); r|=(uint32_t)n<<(i*4);} return r; };
            #pragma unroll
            for(int rt=0;rt<2;rt++){          // 2 row-tiles (16 each)
                int rowbase=wr*32+rt*16;
                uint32_t a0=Anib(rowbase+gr,ks+tc*8), a1=Anib(rowbase+gr+8,ks+tc*8),
                         a2=Anib(rowbase+gr,ks+tc*8+32), a3=Anib(rowbase+gr+8,ks+tc*8+32);
                #pragma unroll
                for(int ct=0;ct<4;ct++){      // 4 col-tiles (8 each)
                    int col=wc*32+ct*8+gr;
                    uint32_t b0=Bnib(col,ks+tc*8), b1=Bnib(col,ks+tc*8+32);
                    int m=rt*4+ct;
                    asm volatile(
                      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                      : "+f"(d[m][0]),"+f"(d[m][1]),"+f"(d[m][2]),"+f"(d[m][3])
                      : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
                        "r"(sU),"n"(0),"n"(0), "r"(sU),"n"(0),"n"(0));
                }
            }
        }
        __syncthreads();
        // drain (BK=128, max partial 128*36=4608 << 2^24; safe every iter)
        #pragma unroll
        for(int i=0;i<8;i++)
            #pragma unroll
            for(int j=0;j<4;j++){ acc[i][j]+=(int32_t)llrintf(d[i][j]); d[i][j]=0.0f; }
    }

    // write C
    #pragma unroll
    for(int rt=0;rt<2;rt++){
        int rowbase=blockM+wr*32+rt*16;
        #pragma unroll
        for(int ct=0;ct<4;ct++){
            int colbase=blockN+wc*32+ct*8, m=rt*4+ct;
            C[(size_t)(rowbase+gr)*N   + colbase+tc*2]   = acc[m][0];
            C[(size_t)(rowbase+gr)*N   + colbase+tc*2+1] = acc[m][1];
            C[(size_t)(rowbase+gr+8)*N + colbase+tc*2]   = acc[m][2];
            C[(size_t)(rowbase+gr+8)*N + colbase+tc*2+1] = acc[m][3];
        }
    }
}
static double now_ms(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3+ts.tv_nsec/1e6; }

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d  (BM%d BN%d BK%d, %d thr)\n",p.name,p.major,p.minor,BM,BN,BK,THREADS);
    // correctness
    { int M=128,N=128,K=512;
      std::vector<int8_t> A(M*K),B(K*N);
      for(size_t i=0;i<A.size();i++) A[i]=M11V[i&7];
      for(size_t i=0;i<B.size();i++) B[i]=M11V[(i*3+1)&7];
      std::vector<int32_t> ref(M*N,0);
      for(int i=0;i<M;i++)for(int j=0;j<N;j++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[i*K+k]*B[k*N+j]; ref[i*N+j]=(int32_t)s; }
      std::vector<uint8_t> Ap,Bp; pack_e2m1(A,Ap,M,K); pack_e2m1(B,Bp,K,N);
      uint8_t *dA,*dB; int32_t* dC; CK(cudaMalloc(&dA,Ap.size())); CK(cudaMalloc(&dB,Bp.size())); CK(cudaMalloc(&dC,(size_t)M*N*4));
      CK(cudaMemcpy(dA,Ap.data(),Ap.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB,Bp.data(),Bp.size(),cudaMemcpyHostToDevice));
      dim3 g(N/BN,M/BM); fp4_gemm2<<<g,THREADS>>>(dA,dB,dC,M,N,K);
      cudaError_t e=cudaDeviceSynchronize(); if(e!=cudaSuccess){printf("launch %s\n",cudaGetErrorString(e));return 2;}
      std::vector<int32_t> got(M*N); CK(cudaMemcpy(got.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
      int fails=0; for(int i=0;i<M*N;i++) if(got[i]!=ref[i]) fails++;
      printf("correctness %dx%dx%d : %s (%d/%d)\n",M,N,K, fails==0?"BYTE-EXACT":"WRONG",fails,M*N);
      cudaFree(dA);cudaFree(dB);cudaFree(dC);
      if(fails) return 1;
    }
    // throughput
    int M=16384,N=4096,K=4096;
    std::vector<int8_t> A(M*K),B(K*N);
    for(size_t i=0;i<A.size();i++) A[i]=M11V[i&7];
    for(size_t i=0;i<B.size();i++) B[i]=M11V[(i*3+1)&7];
    std::vector<uint8_t> Ap,Bp; pack_e2m1(A,Ap,M,K); pack_e2m1(B,Bp,K,N);
    uint8_t *dA,*dB; int32_t* dC; CK(cudaMalloc(&dA,Ap.size())); CK(cudaMalloc(&dB,Bp.size())); CK(cudaMalloc(&dC,(size_t)M*N*4));
    CK(cudaMemcpy(dA,Ap.data(),Ap.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB,Bp.data(),Bp.size(),cudaMemcpyHostToDevice));
    dim3 g(N/BN,M/BM);
    fp4_gemm2<<<g,THREADS>>>(dA,dB,dC,M,N,K); CK(cudaDeviceSynchronize());
    const int reps=30; double t0=now_ms();
    for(int r=0;r<reps;r++) fp4_gemm2<<<g,THREADS>>>(dA,dB,dC,M,N,K);
    CK(cudaDeviceSynchronize()); double fp4ms=(now_ms()-t0)/reps;

    int8_t *iA,*iB; int32_t* iC; CK(cudaMalloc(&iA,(size_t)M*K)); CK(cudaMalloc(&iB,(size_t)K*N)); CK(cudaMalloc(&iC,(size_t)M*N*4));
    CK(cudaMemcpy(iA,A.data(),(size_t)M*K,cudaMemcpyHostToDevice)); CK(cudaMemcpy(iB,B.data(),(size_t)K*N,cudaMemcpyHostToDevice));
    cublasLtHandle_t lt; cublasLtCreate(&lt); void* ws; size_t wss=256<<20; CK(cudaMalloc(&ws,wss));
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I);
    cublasOperation_t NN=CUBLAS_OP_N; cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&NN,sizeof(NN));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&NN,sizeof(NN));
    cublasLtMatrixLayout_t lb,la,lc; cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,N,K,N);
    cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K); cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,N,M,N);
    int32_t al=1,be=0; auto lt1=[&]{ return cublasLtMatmul(lt,op,&al,iB,lb,iA,la,&be,iC,lc,iC,lc,nullptr,ws,wss,0); };
    lt1(); CK(cudaDeviceSynchronize()); t0=now_ms();
    for(int r=0;r<reps;r++) lt1(); CK(cudaDeviceSynchronize()); double i8ms=(now_ms()-t0)/reps;

    double gmac=1.0*M*N*K/1e9;
    printf("\nRC phase-2 %dx%dx%d (%.0f GMAC)\n",M,N,K,gmac);
    printf("  FP4 v2 hand   : %8.2f ms (%6.0f GMAC/s)\n", fp4ms, gmac/(fp4ms/1e3));
    printf("  cuBLASLt INT8 : %8.2f ms (%6.0f GMAC/s)\n", i8ms, gmac/(i8ms/1e3));
    printf("  FP4v2 : INT8  = %.2fx  (v1 was 0.04x)  %s\n", i8ms/fp4ms,
           i8ms/fp4ms>1.05?"FP4 WINS": i8ms/fp4ms>0.95?"tie":"int8 still ahead");
    return 0;
}
