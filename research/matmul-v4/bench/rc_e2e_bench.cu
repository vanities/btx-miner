// rc_e2e_bench.cu -- v4.5 ENC_RC (Resident Curriculum) END-TO-END perf bench at CONSENSUS dims.
//
// WHY THIS EXISTS: numair's own matmul-v4-rc-harness can only run toy (~10 KB) and medium (~1 MB)
// shapes -- `caps: 512MiB/2GiB/8GiB = skip` on every run -- which is exactly why his G2/G3/G4
// silicon gates are all still open. Nobody, including him, has measured RC at the real sizes.
// This is a PERFORMANCE MODEL (not byte-exact), so it survives his golden re-pins: shapes and
// operation mix change slowly, digests change hourly.
//
// Consensus dims (src/matmul/matmul_v4_rc.h @ btx PR#89):
//   rounds=4  d_head=128  n_q=512  n_ctx=786432  L_lyr=16  d_model=4096  b_seq=16384
//   Phase-1 KV  = 2 * n_ctx * d_head          = 192 MiB
//   Phase-2 act = b_seq * 2 * d_model * L_lyr = 2.00 GiB   (X activations + G grads)
//
// Per-layer Phase-2 work, from his matmul_v4_rc.cpp:
//   forward  ForwardLayer(W,X)      : (b_seq x d_model) @ (d_model x d_model)
//   dgrad    G . W                  : ExactGemmS8S8(G, W, b_seq, d_model, d_model)
//   wgrad    G^T . X                : d_model x d_model, contraction over b_seq
//   => 3 * b_seq * d_model^2 MAC per layer.
//
// THE NUMBER THAT MATTERS is k = wall(StoreOnlyX0) / wall(StoreAll) -- the time penalty a
// memory-poor card pays to skip holding the 2 GiB. His G3 gate needs k >= 1.3 for the
// memory-hardness to bite. Toy dims give ~1.1-1.27; naive arithmetic at consensus dims says
// ~3.5 (backward layer l must replay l forward layers => L(L-1)/2 = 120 extra passes vs 16).
// Measuring it is the whole point.
//
// build: nvcc -O3 -arch=sm_XX rc_e2e_bench.cu -lcublasLt -lnvidia-ml -o rc_e2e
// usage: rc_e2e [layers] [b_seq] [mode]     mode: 0=all 1=StoreAll 2=Every4 3=OnlyX0

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvml.h>

using i8 = int8_t; using i32 = int32_t; using i64 = int64_t;
using u32 = uint32_t; using u64 = uint64_t; using u8 = uint8_t;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- consensus constants (matmul_v4_rc.h) ----
static constexpr u32 kRounds   = 4;
static constexpr u32 kDHead    = 128;
static constexpr u32 kNQ       = 512;
static constexpr u32 kNCtx     = 786432;   // 0.75 Mi -> KV = 2*nctx*dhead = 192 MiB
static constexpr u32 kDModel   = 4096;
static constexpr u32 kLeafByte = 1024;     // 32x32 int8 Merkle leaf

// ---------------------------------------------------------------- cuBLASLt INT8 GEMM
static cublasLtHandle_t g_lt; static void* g_ws; static size_t g_wsz = size_t(256) << 20;

// C(m x n) = A(m x k) * B(k x n), row-major int8 -> int32, via Lt col-major swap trick.
static void gemm8(const i8* A, const i8* B, i32* C, int m, int n, int k)
{
    cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof(N));
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, k, m, k);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, k, n, k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, m, n, m);
    i32 alpha = 1, beta = 0;
    cublasStatus_t st = cublasLtMatmul(g_lt, op, &alpha, A, la, B, lb, &beta, C, lc, C, lc,
                                       nullptr, g_ws, g_wsz, 0);
    if (st != CUBLAS_STATUS_SUCCESS) {
        static int once = 0;
        if (!once++) printf("!! cublasLtMatmul FAILED status=%d  (m=%d n=%d k=%d) "
                            "-- all timings below are meaningless\n", (int)st, m, n, k);
    }
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

// ---- requantise int32 -> int8 (stands in for his MX Extract; same traffic + ALU shape) ----
__global__ void k_requant(const i32* __restrict__ in, i8* __restrict__ out, size_t n, u32 shift)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    i32 v = in[i] >> shift;
    out[i] = (i8)(v > 127 ? 127 : (v < -128 ? -128 : v));
}

// ---- Phase-1: softmax-free attention, streamed over the KV bank (FlashMX-shaped) ----
// scores(nq x tile) = Q(nq x dhead) @ Ktile^T ; acc(nq x dhead) += scores @ Vtile
// acc(nq x dhead) += tmp(nq x dhead). scores@V itself is a GEMM (tensor cores) -- hand-rolling
// it as one-thread-per-output was ~40x slower, and a "parallel" reduce was worse still (128-byte
// stride between lanes = zero coalescing). Phase 1 is two GEMMs, exactly like phase 2.
__global__ void k_acc_add(const i32* __restrict__ tmp, i32* __restrict__ acc, size_t n)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += tmp[i];
}

// ---- Phase-3: SHA256d over the round transcript (Merkle leaves) ----
__constant__ u32 K256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__device__ __forceinline__ u32 ror(u32 x,int n){return (x>>n)|(x<<(32-n));}
__device__ void sha_block(const u8* p, u32* h)
{
    u32 w[64];
#pragma unroll
    for (int i=0;i<16;i++) w[i]=(p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];
    for (int i=16;i<64;i++){u32 s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3),
                                 s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
                            w[i]=w[i-16]+s0+w[i-7]+s1;}
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i=0;i<64;i++){
        u32 S1=ror(e,6)^ror(e,11)^ror(e,25), ch=(e&f)^((~e)&g), t1=hh+S1+ch+K256[i]+w[i];
        u32 S0=ror(a,2)^ror(a,13)^ror(a,22), mj=(a&b)^(a&c)^(b&c), t2=S0+mj;
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}
// one thread per 1 KiB leaf -> SHA256d
__global__ void k_merkle_leaves(const u8* __restrict__ data, u32 nleaf, u8* __restrict__ out)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= nleaf) return;
    const u8* p = data + size_t(i) * kLeafByte;
    u32 h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (u32 b=0;b<kLeafByte/64;b++) sha_block(p+b*64,h);
    u8 d1[32];
#pragma unroll
    for(int j=0;j<8;j++){d1[j*4]=h[j]>>24;d1[j*4+1]=h[j]>>16;d1[j*4+2]=h[j]>>8;d1[j*4+3]=h[j];}
    u32 h2[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[64]={0}; memcpy(blk,d1,32); blk[32]=0x80; blk[62]=1; blk[63]=0;
    sha_block(blk,h2);
#pragma unroll
    for(int j=0;j<8;j++) out[size_t(i)*32+j*4]=h2[j]>>24;
}

// ---------------------------------------------------------------- Phase 2
enum Ckpt { StoreAll=0, StoreEvery4=1, StoreOnlyX0=2 };
static const char* ckpt_name(Ckpt c){ return c==StoreAll?"StoreAll":(c==StoreEvery4?"StoreEvery4":"StoreOnlyX0"); }
static bool kept(Ckpt c, u32 l){ return c==StoreAll ? true : (c==StoreEvery4 ? (l%4==0) : (l==0)); }

int main(int argc, char** argv)
{
    const u32 L      = argc>1 ? (u32)atoi(argv[1]) : 16;
    const u32 b_seq  = argc>2 ? (u32)atoi(argv[2]) : 16384;
    const int mode   = argc>3 ? atoi(argv[3]) : 0;
    const u32 d      = kDModel;

    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
    nvmlInit(); nvmlDevice_t nv; nvmlDeviceGetHandleByIndex(0,&nv);
    cublasLtCreate(&g_lt); CK(cudaMalloc(&g_ws,g_wsz));

    const size_t act = size_t(b_seq)*d;                      // one activation plane (int8)
    const size_t wsz = size_t(d)*d;                          // one weight matrix (int8)
    printf("device: %s sm_%d%d | ENC_RC CONSENSUS dims: L=%u b_seq=%u d_model=%u n_ctx=%u d_head=%u\n",
           pr.name, pr.major, pr.minor, L, b_seq, d, kNCtx, kDHead);
    printf("  phase1 KV   = %.0f MiB   phase2 X+G = %.2f GiB (StoreAll)   W = %.0f MiB\n",
           2.0*kNCtx*kDHead/1048576.0, 2.0*(L+1)*act/1073741824.0, double(L)*wsz/1048576.0);
    const double mac_layer = 3.0*double(b_seq)*d*d;          // fwd + dgrad + wgrad
    printf("  per-episode MAC (phase2) = %.2fe12  (3 x b_seq x d_model^2 x L)\n", mac_layer*L/1e12);

    // ---- allocations ----
    i8 *dK,*dV,*dQ; i32 *dScore,*dAcc;
    CK(cudaMalloc(&dK,size_t(kNCtx)*kDHead)); CK(cudaMalloc(&dV,size_t(kNCtx)*kDHead));
    CK(cudaMalloc(&dQ,size_t(kNQ)*kDHead));
    const u32 p1tile = 32768;
    CK(cudaMalloc(&dScore,size_t(kNQ)*p1tile*4)); CK(cudaMalloc(&dAcc,size_t(kNQ)*kDHead*4));
    i8* dScore8; CK(cudaMalloc(&dScore8,size_t(kNQ)*p1tile));
    i32* dAccTmp; CK(cudaMalloc(&dAccTmp,size_t(kNQ)*kDHead*4));
    std::vector<i8*> W(L); for(u32 l=0;l<L;l++) CK(cudaMalloc(&W[l],wsz));
    std::vector<i8*> X(L+1,nullptr); for(u32 l=0;l<=L;l++) CK(cudaMalloc(&X[l],act));
    i8 *dG,*dGprev; CK(cudaMalloc(&dG,act)); CK(cudaMalloc(&dGprev,act));
    i32* dTmp; CK(cudaMalloc(&dTmp,act*4));
    i32* dWg;  CK(cudaMalloc(&dWg,wsz*4));
    u8 *dLeaf,*dHash; const u32 nleaf = (u32)(act/kLeafByte);
    CK(cudaMalloc(&dLeaf,size_t(nleaf)*kLeafByte)); CK(cudaMalloc(&dHash,size_t(nleaf)*32));
    CK(cudaMemset(dK,3,size_t(kNCtx)*kDHead)); CK(cudaMemset(dV,5,size_t(kNCtx)*kDHead));
    CK(cudaMemset(dQ,7,size_t(kNQ)*kDHead));
    for(u32 l=0;l<L;l++) CK(cudaMemset(W[l],1,wsz));
    CK(cudaMemset(X[0],2,act));

    auto fwd_layer = [&](u32 l, i8* xin, i8* xout){
        gemm8(xin, W[l], dTmp, b_seq, d, d);
        k_requant<<<(act+255)/256,256>>>(dTmp, xout, act, 12);
    };

    // ---- Phase 1 (shared across modes; measured once) ----
    auto phase1 = [&](){
        CK(cudaMemset(dAcc,0,size_t(kNQ)*kDHead*4));
        const size_t sc = size_t(kNQ)*p1tile;
        for (u32 c0=0; c0<kNCtx; c0+=p1tile) {
            gemm8(dQ, dK + size_t(c0)*kDHead, dScore, kNQ, p1tile, kDHead);      // Q . K^T
            k_requant<<<(sc+255)/256,256>>>(dScore, dScore8, sc, 8);             // -> int8
            gemm8(dScore8, dV + size_t(c0)*kDHead, dAccTmp, kNQ, kDHead, p1tile); // scores . V
            k_acc_add<<<(size_t(kNQ)*kDHead+255)/256,256>>>(dAccTmp, dAcc, size_t(kNQ)*kDHead);
        }
    };

    // ---- Phase 2 under a checkpoint policy ----
    auto phase2 = [&](Ckpt c, long* recomputed){
        long rc = 0;
        for (u32 l=0; l<L; ++l) fwd_layer(l, X[l], X[l+1]);           // forward
        CK(cudaMemset(dG,1,act));
        for (int l=(int)L-1; l>=0; --l) {                              // backward
            if (!kept(c,(u32)l)) {                                     // replay to reach X[l]
                u32 src = 0; for (int m=l; m>=0; --m) if (kept(c,(u32)m)) { src=(u32)m; break; }
                for (u32 m=src; m<(u32)l; ++m) { fwd_layer(m, X[m], X[m+1]); ++rc; }
            }
            gemm8(dG, W[l], dTmp, b_seq, d, d);                        // dgrad  G . W
            k_requant<<<(act+255)/256,256>>>(dTmp, dGprev, act, 12);
            gemm8(dG, X[l], dWg, d, d, b_seq);                         // wgrad  G^T . X
            i8* t=dG; dG=dGprev; dGprev=t;
        }
        if (recomputed) *recomputed = rc;
    };

    auto phase3 = [&](){ k_merkle_leaves<<<(nleaf+255)/256,256>>>(dLeaf,nleaf,dHash); };

    // ---- warmup ----
    phase1(); phase2(StoreAll,nullptr); phase3(); CK(cudaDeviceSynchronize());

    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    // NOTE: power MUST be sampled while the kernels are in flight -- reading NVML after
    // cudaEventSynchronize() catches an already-idled GPU (that bug reported 88 W on a 600 W card).
    unsigned g_mw = 0;
    auto time_ms = [&](auto fn){
        CK(cudaDeviceSynchronize()); cudaEventRecord(a); fn(); cudaEventRecord(b);
        unsigned peak = 0;
        while (cudaEventQuery(b) == cudaErrorNotReady) {      // poll power during the work
            unsigned mw = 0;
            if (nvmlDeviceGetPowerUsage(nv, &mw) == NVML_SUCCESS && mw > peak) peak = mw;
        }
        CK(cudaEventSynchronize(b));
        if (peak) g_mw = peak;
        float ms; cudaEventElapsedTime(&ms,a,b); return ms; };

    const float t_p1 = time_ms(phase1);
    const float t_p3 = time_ms(phase3);
    printf("\n  phase1 (recall, 192 MiB KV) : %8.2f ms\n", t_p1);
    printf("  phase3 (merkle, %u leaves)  : %8.2f ms\n", nleaf, t_p3);

    printf("\n  %-12s %10s %10s %12s %8s   %s\n","checkpoint","p2 ms","total ms","episodes/s","watts","k vs StoreAll");
    double base = 0;
    for (int c=StoreAll; c<=StoreOnlyX0; ++c) {
        if (mode && (c != mode-1)) continue;
        long rc=0;
        const float t_p2 = time_ms([&]{ phase2((Ckpt)c,&rc); });
        const double total = (t_p1 + t_p2 + t_p3) * kRounds;   // rounds=4 per episode
        if (c==StoreAll) base = total;
        printf("  %-12s %10.1f %10.1f %12.2f %8.0f   %5.3f  (+%ld replay layers)\n",
               ckpt_name((Ckpt)c), t_p2, total, 1000.0/total, g_mw/1000.0,
               base>0?total/base:1.0, rc);
    }
    printf("\n  NOTE: perf model at consensus dims -- shapes/op-mix faithful, digests not byte-exact.\n");
    printf("        k = wall(StoreOnlyX0)/wall(StoreAll); his G3 gate needs k >= 1.3.\n");
    nvmlShutdown(); return 0;
}
