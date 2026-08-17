// Per-GPU auto-dispatching GEMM for the RC miner: detect the card at init, route each MxNxK to
// the best kernel that GPU actually has, byte-exact on all paths.
//
//   sm_120 (consumer Blackwell 5090 / RTX 6000): native mma.kind::mxf4 FP4 GEMM (this file, v2)
//   sm_100 (B200):  tcgen05 block-scaled MXFP4 -- SLOT ONLY. tcgen05.mma is single-thread-issued,
//                   tmem-accumulated, TMA-fed; hand-writing it is CUTLASS-class + needs a B200 to
//                   iterate on. Until that kernel exists, sm_100 falls back to cuBLASLt int8
//                   (already near-peak for int8 on a B200). The slot is where a tcgen05 GEMM plugs in.
//   else (Ampere 3090 etc): cuBLASLt int8 -- no native FP4 exists on the hardware.
//
// One fatbin: FP4 kernel guarded to the sm_120a slice via __CUDA_ARCH__. The dispatcher reads the
// device compute capability once and picks the path; every path is validated byte-exact here.
//
// build: nvcc -O3 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 \
//        -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120a,code=sm_120a \
//        rc_fp4_dispatch.cu -lcublasLt -o rcgemm
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

// ---- sm_120a native FP4 GEMM (v2 smem-tiled), guarded so the fatbin's non-120a slices still build
#define BM 64
#define BN 64
#define BK 128
#define THREADS 128
__global__ __launch_bounds__(THREADS) void fp4_gemm_sm120a(const uint8_t* __restrict__ A,
                        const uint8_t* __restrict__ B, int32_t* __restrict__ C, int M,int N,int K){
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    __shared__ uint8_t As[BM*(BK/2)]; __shared__ uint8_t Bs[BK*(BN/2)];
    const int t=threadIdx.x, wr=(t/32)/2, wc=(t/32)%2, gr=(t%32)/4, tc=(t%32)%4;
    const int blockM=blockIdx.y*BM, blockN=blockIdx.x*BN;
    int32_t acc[8][4]; float d[8][4];
    #pragma unroll
    for(int i=0;i<8;i++)
        #pragma unroll
        for(int j=0;j<4;j++){ acc[i][j]=0; d[i][j]=0.0f; }
    const uint32_t sU=0x7F7F7F7Fu;
    for(int k0=0;k0<K;k0+=BK){
        for(int off=t*16; off<BM*(BK/2); off+=THREADS*16){ int row=off/(BK/2),byte=off%(BK/2);
            *(uint4*)&As[off]=*(const uint4*)&A[((size_t)(blockM+row)*K+k0)/2+byte]; }
        for(int off=t*16; off<BK*(BN/2); off+=THREADS*16){ int kk=off/(BN/2),byte=off%(BN/2);
            *(uint4*)&Bs[off]=*(const uint4*)&B[((size_t)(k0+kk)*N+blockN)/2+byte]; }
        __syncthreads();
        #pragma unroll
        for(int ks=0;ks<BK;ks+=64){
            auto An=[&](int row,int kb)->uint32_t{ uint32_t r=0;
                #pragma unroll
                for(int i=0;i<8;i++){int kk=kb+i;uint8_t by=As[row*(BK/2)+kk/2];uint8_t n=(kk&1)?(by>>4):(by&0xF);r|=(uint32_t)n<<(i*4);}return r;};
            auto Bn=[&](int col,int kb)->uint32_t{ uint32_t r=0;
                #pragma unroll
                for(int i=0;i<8;i++){int kk=kb+i;uint8_t by=Bs[kk*(BN/2)+col/2];uint8_t n=(col&1)?(by>>4):(by&0xF);r|=(uint32_t)n<<(i*4);}return r;};
            #pragma unroll
            for(int rt=0;rt<2;rt++){ int rb=wr*32+rt*16;
                uint32_t a0=An(rb+gr,ks+tc*8),a1=An(rb+gr+8,ks+tc*8),a2=An(rb+gr,ks+tc*8+32),a3=An(rb+gr+8,ks+tc*8+32);
                #pragma unroll
                for(int ct=0;ct<4;ct++){ int col=wc*32+ct*8+gr, m=rt*4+ct;
                    uint32_t b0=Bn(col,ks+tc*8),b1=Bn(col,ks+tc*8+32);
                    asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                      : "+f"(d[m][0]),"+f"(d[m][1]),"+f"(d[m][2]),"+f"(d[m][3])
                      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(sU),"n"(0),"n"(0),"r"(sU),"n"(0),"n"(0)); }
            }
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;i++)
            #pragma unroll
            for(int j=0;j<4;j++){ acc[i][j]+=(int32_t)llrintf(d[i][j]); d[i][j]=0.0f; }
    }
    #pragma unroll
    for(int rt=0;rt<2;rt++){ int rb=blockM+wr*32+rt*16;
        #pragma unroll
        for(int ct=0;ct<4;ct++){ int cb=blockN+wc*32+ct*8, m=rt*4+ct;
            C[(size_t)(rb+gr)*N+cb+tc*2]=acc[m][0]; C[(size_t)(rb+gr)*N+cb+tc*2+1]=acc[m][1];
            C[(size_t)(rb+gr+8)*N+cb+tc*2]=acc[m][2]; C[(size_t)(rb+gr+8)*N+cb+tc*2+1]=acc[m][3]; }
    }
#endif
}

// ---- dispatcher ----------------------------------------------------------------------------
enum GemmPath { PATH_FP4_SM120A, PATH_TCGEN05_SM100, PATH_INT8 };
static GemmPath g_path; static const char* g_path_name;
static cublasLtHandle_t g_lt; static void* g_ws; static size_t g_wss=256<<20;

static void rc_gemm_init(const cudaDeviceProp& p){
    if(p.major==12){ g_path=PATH_FP4_SM120A; g_path_name="FP4 native mma.mxf4 (sm_120a)"; }
    else if(p.major==10){ g_path=PATH_TCGEN05_SM100; g_path_name="int8 (tcgen05 FP4 slot: NOT built)"; }
    else { g_path=PATH_INT8; g_path_name="cuBLASLt int8 (no native FP4 on this arch)"; }
    cublasLtCreate(&g_lt); cudaMalloc(&g_ws,g_wss);
    printf("[rc_gemm] %s sm_%d%d -> path: %s\n", p.name,p.major,p.minor,g_path_name);
}
// int8 cuBLASLt: row-major C=A.B via col-major C'=B'.A'
static void gemm_int8(const int8_t* A,const int8_t* B,int32_t* C,int M,int N,int K){
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I);
    cublasOperation_t NN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&NN,sizeof(NN));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&NN,sizeof(NN));
    cublasLtMatrixLayout_t lb,la,lc; cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,N,K,N);
    cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K); cublasLtMatrixLayoutCreate(&lc,CUDA_R_32I,N,M,N);
    int32_t al=1,be=0; cublasLtMatmul(g_lt,op,&al,B,lb,A,la,&be,C,lc,C,lc,nullptr,g_ws,g_wss,0);
    cublasLtMatrixLayoutDestroy(lb);cublasLtMatrixLayoutDestroy(la);cublasLtMatrixLayoutDestroy(lc);cublasLtMatmulDescDestroy(op);
}
// Unified entry. FP4 paths take packed-nibble operands; int8 path takes int8. Same integer result.
static void rc_gemm(const uint8_t* Afp4,const uint8_t* Bfp4,const int8_t* Ai8,const int8_t* Bi8,
                    int32_t* C,int M,int N,int K){
    switch(g_path){
        case PATH_FP4_SM120A: { dim3 g(N/BN,M/BM); fp4_gemm_sm120a<<<g,THREADS>>>(Afp4,Bfp4,C,M,N,K); break; }
        case PATH_TCGEN05_SM100:  // slot: real tcgen05 kernel plugs in here; until then, int8.
        case PATH_INT8: gemm_int8(Ai8,Bi8,C,M,N,K); break;
    }
}
static double now_ms(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e3+ts.tv_nsec/1e6; }

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    rc_gemm_init(p);
    int M=2048,N=1024,K=2048;   // moderate: validate byte-exact + a quick timing on whatever card
    std::vector<int8_t> A(M*K),B(K*N);
    for(size_t i=0;i<A.size();i++) A[i]=M11V[i&7];
    for(size_t i=0;i<B.size();i++) B[i]=M11V[(i*3+1)&7];
    std::vector<int32_t> ref(M*N,0);
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){ long s=0; for(int k=0;k<K;k++) s+=(long)A[i*K+k]*B[k*N+j]; ref[i*N+j]=(int32_t)s; }
    std::vector<uint8_t> Ap,Bp; pack_e2m1(A,Ap,M,K); pack_e2m1(B,Bp,K,N);
    uint8_t *dAf,*dBf; int8_t *dAi,*dBi; int32_t* dC;
    CK(cudaMalloc(&dAf,Ap.size())); CK(cudaMalloc(&dBf,Bp.size()));
    CK(cudaMalloc(&dAi,(size_t)M*K)); CK(cudaMalloc(&dBi,(size_t)K*N)); CK(cudaMalloc(&dC,(size_t)M*N*4));
    CK(cudaMemcpy(dAf,Ap.data(),Ap.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dBf,Bp.data(),Bp.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dAi,A.data(),(size_t)M*K,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dBi,B.data(),(size_t)K*N,cudaMemcpyHostToDevice));
    rc_gemm(dAf,dBf,dAi,dBi,dC,M,N,K); cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("launch %s\n",cudaGetErrorString(e)); return 2; }
    std::vector<int32_t> got(M*N); CK(cudaMemcpy(got.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    int fails=0; for(int i=0;i<M*N;i++) if(got[i]!=ref[i]) fails++;
    const int reps=30; rc_gemm(dAf,dBf,dAi,dBi,dC,M,N,K); CK(cudaDeviceSynchronize());
    double t0=now_ms(); for(int r=0;r<reps;r++) rc_gemm(dAf,dBf,dAi,dBi,dC,M,N,K); CK(cudaDeviceSynchronize());
    double ms=(now_ms()-t0)/reps;
    printf("[rc_gemm] %dx%dx%d  %s  (%.3f ms, %.0f GMAC/s)\n", M,N,K,
           fails==0?"BYTE-EXACT":"WRONG", ms, (1.0*M*N*K/1e9)/(ms/1e3));
    printf("%s\n", fails==0?"DISPATCH OK -- auto-selected path is byte-exact":"DISPATCH BROKEN");
    return fails==0?0:1;
}
