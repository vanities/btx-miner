// ############################################################################
// # PRIVATE - matador secret sauce. Do NOT contribute / push to any fork/PR.  #
// # This is v4_proto.cu (the public harness) PLUS the C-13 limb-decomposition #
// # combine (--c13): the mod-q combine as 25 s8->s32 tensor GEMMs. The public #
// # PR #90 harness deliberately does NOT contain this.                        #
// ############################################################################
// MatMul v4 per-stage bench + C-13 limb combine  (NVIDIA / cuBLASLt)  [PRIVATE]
// ---------------------------------------------------------------------------
// A standing check for the §K.2a-WT wall-time invariant: it runs the whole v4
// per-nonce hot path ON THE GPU and reports the measured per-stage split, so
// "the tensor GEMM must dominate measured wall-time" can be checked on real
// silicon rather than inferred from MAC/byte counts.
//
//   stage 1  operand-gen: A,B (n x n) + U (m x n) + V (n x m). With --wide, a
//            bit-exact wide counter-mode XOF matching ExpandBalancedS8Stream
//            (count -> prefix sum -> scatter); default is the retired
//            per-element XOF (kept for before/after comparison).
//   stage 2  cuBLASLt INT8->INT32: P = U*A (m x n), Q = B*V (n x m)   (§E.3)
//   stage 3  mod-q combine: Chat[a][c] = (sum_k P[a][k]*Q[k][c]) mod q, tiled.
//
// Per-stage CUDA-event timing over a batch of nonces => stage %split + nonce/s
// + a machine-parseable CSV row (+ board power via NVML).
//   --emit    reproduces the reference digest H(sigma||Chat) for fixed seeds,
//             BIT-EXACT to the CPU reference (cross-arch determinism check).
//   --verify  diffs cuBLASLt vs a scalar INT32 GEMM (self-consistency).
//   --wide    (default off) use the wide-stream XOF; omit for the legacy path.
//   --pipe    deploy-path pipeline: hoisted template P-split + expand||tensor
//             double-buffer overlap + device S4 digest chains (--kd <n> ring).
//
// build:  nvcc -O3 -arch=native matmul_v4_stage_bench.cu -lcublasLt -lnvidia-ml -o v4bench
// run:    ./v4bench 4096 32 --wide        # n=4096, 32 nonces, wide XOF
//         ./v4bench 4096 1  --emit --wide  # print the reference-matching digest

#include <cublasLt.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); exit(1);} } while(0)

// ===================== device SHA-256 (single 64B block) ====================
__constant__ uint32_t K256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ uint32_t ror(uint32_t x,int n){return (x>>n)|(x<<(32-n));}

// SHA-256 of msg of length len (len<=55) -> 32-byte digest out. one block.
__device__ void sha256_1blk(const uint8_t* msg, int len, uint8_t out[32]) {
    uint8_t blk[64];
    #pragma unroll
    for (int i=0;i<64;i++) blk[i]=0;
    for (int i=0;i<len;i++) blk[i]=msg[i];
    blk[len]=0x80;
    uint64_t bits=(uint64_t)len*8;
    for (int i=0;i<8;i++) blk[63-i]=(uint8_t)(bits>>(8*i));
    uint32_t w[64];
    #pragma unroll
    for (int i=0;i<16;i++)
        w[i]=(blk[i*4]<<24)|(blk[i*4+1]<<16)|(blk[i*4+2]<<8)|blk[i*4+3];
    #pragma unroll
    for (int i=16;i<64;i++){
        uint32_t s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a;
    uint32_t e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma unroll
    for (int i=0;i<64;i++){
        uint32_t S1=ror(e,6)^ror(e,11)^ror(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+K256[i]+w[i];
        uint32_t S0=ror(a,2)^ror(a,13)^ror(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    uint32_t H[8]={0x6a09e667+a,0xbb67ae85+b,0x3c6ef372+c,0xa54ff53a+d,
                   0x510e527f+e,0x9b05688c+f,0x1f83d9ab+g,0x5be0cd19+h};
    #pragma unroll
    for (int i=0;i<8;i++){out[i*4]=H[i]>>24;out[i*4+1]=H[i]>>16;out[i*4+2]=H[i]>>8;out[i*4+3]=H[i];}
}

// ============== chained SHA-256d sketch digest ON DEVICE (S4 stage) ==========
// One thread = one nonce's digest chain: SHA256d(tag||sigma||payload) where the
// payload is the raw device Chat buffer (SerializeSketch writes LE64, which IS
// the little-endian device memory layout -> zero-copy). The chain is serial per
// nonce; throughput comes from many chains in flight. The reference computes
// this stage on the HOST CPU (matmul_v4_bmx4_batch.cpp) -- device S4 is ours.
__device__ void sha256_compress(uint32_t* H, const uint8_t* blk){
    uint32_t w[64];
    #pragma unroll
    for(int i=0;i<16;i++) w[i]=(blk[i*4]<<24)|(blk[i*4+1]<<16)|(blk[i*4+2]<<8)|blk[i*4+3];
    #pragma unroll
    for(int i=16;i<64;i++){
        uint32_t s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    #pragma unroll
    for(int i=0;i<64;i++){
        uint32_t S1=ror(e,6)^ror(e,11)^ror(e,25), ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+K256[i]+w[i];
        uint32_t S0=ror(a,2)^ror(a,13)^ror(a,22), mj=(a&b)^(a&c)^(b&c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+S0+mj;
    }
    H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
}
// compress from 16 already-big-endian schedule words (no byte staging array,
// which would spill to local memory and dominate the serial chain's latency)
__device__ __forceinline__ void sha256_compress_w(uint32_t* H, const uint32_t* w0){
    uint32_t w[64];
    #pragma unroll
    for(int i=0;i<16;i++) w[i]=w0[i];
    #pragma unroll
    for(int i=16;i<64;i++){
        uint32_t s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    #pragma unroll
    for(int i=0;i<64;i++){
        uint32_t S1=ror(e,6)^ror(e,11)^ror(e,25), ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+K256[i]+w[i];
        uint32_t S0=ror(a,2)^ror(a,13)^ror(a,22), mj=(a&b)^(a&c)^(b&c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+S0+mj;
    }
    H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
}
// chain tid hashes hdr[hlen] || pays[(tid%nbuf)*stride .. +plen], SHA256d, out[tid*32].
// hlen is 45 (tag||sigma); the fast path exploits base%64==0, hlen=45 => the
// payload window [base-48, base+20) is 4-byte aligned and 17 uint32 loads +
// __byte_perm(wa[i],wa[i+1],0x3456) assemble the 16 BE schedule words directly.
__global__ void digest_chain(const uint8_t* __restrict__ hdr, int hlen,
                             const uint8_t* __restrict__ pays, size_t stride, int nbuf,
                             size_t plen, int nchains, uint8_t* __restrict__ out){
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=nchains) return;
    const uint8_t* pay = pays + (size_t)(tid % nbuf)*stride;
    uint32_t H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const uint64_t total=(uint64_t)hlen+plen;
    const uint64_t nb=(total+1+8+63)/64;            // padded block count
    const bool aligned45 = (hlen==45) && (((uintptr_t)pay & 3u)==0);
    for(uint64_t bi=0;bi<nb;++bi){
        const uint64_t base=bi*64;
        uint32_t w[16];
        if(aligned45 && base>=64 && base+64<=total){ // fast path: pure-payload block
            const uint32_t* pw=(const uint32_t*)(pay+(base-48)); // 48 = hlen rounded up to 4
            uint32_t wa[17];
            #pragma unroll
            for(int k2=0;k2<17;k2++) wa[k2]=__ldg(pw+k2);
            #pragma unroll
            for(int i=0;i<16;i++) w[i]=__byte_perm(wa[i],wa[i+1],0x3456);
        } else {                                    // header / padding boundary blocks
            #pragma unroll
            for(int i=0;i<16;i++){
                uint32_t v32=0;
                for(int k2=0;k2<4;k2++){
                    uint64_t pos=base+i*4+k2; uint8_t v;
                    if(pos<(uint64_t)hlen) v=hdr[pos];
                    else if(pos<total) v=__ldg(&pay[pos-hlen]);
                    else if(pos==total) v=0x80;
                    else if(pos>=nb*64-8) v=(uint8_t)((total*8)>>(8*(nb*64-1-pos)));
                    else v=0;
                    v32=(v32<<8)|v;
                }
                w[i]=v32;
            }
        }
        sha256_compress_w(H,w);
    }
    uint8_t d1[32];
    #pragma unroll
    for(int i=0;i<8;i++){d1[i*4]=H[i]>>24;d1[i*4+1]=H[i]>>16;d1[i*4+2]=H[i]>>8;d1[i*4+3]=H[i];}
    uint8_t d2[32]; sha256_1blk(d1,32,d2);          // second SHA of SHA256d
    for(int i=0;i<32;i++) out[(size_t)tid*32+i]=d2[i];
}

// consensus-exact: SHA(seed[32] || index_le[4] [|| retry_le[4]]), byte[0]<251 -> s8
__global__ void gen_operand(int8_t* out, const uint8_t* seed32, uint32_t count) {
    uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx>=count) return;
    uint8_t msg[40];
    // consensus SeedBytesLE: the reference reverses the 32-byte uint256 before
    // hashing (int8_field.cpp SeedBytesLE), so msg[i] = seed[31-i].
    #pragma unroll
    for (int i=0;i<32;i++) msg[i]=seed32[31-i];
    msg[32]=idx; msg[33]=idx>>8; msg[34]=idx>>16; msg[35]=idx>>24;
    uint8_t h[32];
    for (uint32_t retry=0; retry<256; ++retry){
        int len=36;
        if (retry>0){ msg[36]=retry;msg[37]=retry>>8;msg[38]=retry>>16;msg[39]=retry>>24; len=40; }
        sha256_1blk(msg,len,h);
        if (h[0]<251){ out[idx]=(int8_t)((int)h[0]-125); return; }
    }
    out[idx]=0;
}

// WIDE XOF (proposed fix): one SHA-256 per `per_hash` elements, consuming all 32
// output bytes in a counter/squeeze mode with rejection sampling. Models a fixed
// operand XOF that stops discarding 31/32 SHA bytes. per_hash<=30 keeps the ~2%
// rejection shortfall negligible (32*0.98~=31.4). This is a COST model of the fix
// (it is NOT byte-compatible with the per-element reference; --emit stays legacy).
__global__ void gen_operand_wide(int8_t* out, const uint8_t* seed32, uint32_t count, uint32_t per_hash){
    uint32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t base = t*per_hash;
    if (base>=count) return;
    uint8_t msg[36];
    #pragma unroll
    for (int i=0;i<32;i++) msg[i]=seed32[31-i];
    msg[32]=t; msg[33]=t>>8; msg[34]=t>>16; msg[35]=t>>24;
    uint8_t h[32]; sha256_1blk(msg,36,h);
    uint32_t w=0;
    for (int i=0;i<32 && w<per_hash && base+w<count; ++i)
        if (h[i]<251){ out[base+w]=(int8_t)((int)h[i]-125); ++w; }
    for (; w<per_hash && base+w<count; ++w) out[base+w]=0; // rare rejection shortfall (cost-model pad)
}

// BIT-EXACT wide counter-mode XOF, matching f50f0f8 ExpandBalancedS8Stream:
// block b -> SHA256(seed_le[32] || domain || LE64(b)) [41B, 1 SHA block], all 32
// bytes rejection-sampled (<251 -> byte-125) in stream order. Because rejection
// makes each element's output position depend on all prior blocks, this is a
// two-pass stream compaction (count -> exclusive prefix-sum -> scatter), exactly
// as the reference comment prescribes. domain 0x73='s' (operands), 0x71='q' (Fq).
// grid-stride: per-block work is independent, so output is byte-identical for ANY
// launch grid — a small grid turns these into co-resident "background" kernels
// that fill SM slots a concurrent tensor GEMM leaves idle (the --overlap lever).
__global__ void stream_count(const uint8_t* seed32, uint8_t domain, int8_t* scratch, uint32_t* cnt, uint32_t nblk){
    for(uint32_t b = blockIdx.x*blockDim.x+threadIdx.x; b<nblk; b+=gridDim.x*blockDim.x){
    uint8_t msg[41];
    #pragma unroll
    for(int i=0;i<32;i++) msg[i]=seed32[31-i];
    msg[32]=domain;
    uint64_t blk=b;
    #pragma unroll
    for(int i=0;i<8;i++) msg[33+i]=(uint8_t)(blk>>(8*i));
    uint8_t h[32]; sha256_1blk(msg,41,h);
    uint32_t c=0;
    for(int i=0;i<32;i++) if(h[i]<251) scratch[(size_t)b*32 + c++] = (int8_t)((int)h[i]-125);
    cnt[b]=c;
    }
}
__global__ void stream_scatter(const int8_t* scratch, const uint32_t* off, const uint32_t* cnt, int8_t* out, uint32_t nblk, uint32_t count){
    for(uint32_t b = blockIdx.x*blockDim.x+threadIdx.x; b<nblk; b+=gridDim.x*blockDim.x){
    uint32_t o=off[b], c=cnt[b];
    for(uint32_t r=0;r<c;r++){ uint32_t pos=o+r; if(pos<count) out[pos]=scratch[(size_t)b*32+r]; }
    }
}

// ===================== q = 2^61-1 combine ====================
__device__ __forceinline__ uint64_t fqreduce(unsigned __int128 x){
    const uint64_t Q=((uint64_t)1<<61)-1;
    uint64_t lo=(uint64_t)(x&Q), hi=(uint64_t)(x>>61);
    uint64_t s=lo+hi; s=(s&Q)+(s>>61); if(s>=Q)s-=Q; return s;
}
__device__ __forceinline__ uint64_t fqadd(uint64_t a,uint64_t b){const uint64_t Q=((uint64_t)1<<61)-1;uint64_t s=a+b;if(s>=Q)s-=Q;return s;}
__device__ __forceinline__ uint64_t fqmul(uint64_t a,uint64_t b){return fqreduce((unsigned __int128)a*b);}
__device__ __forceinline__ uint64_t fqfroms32(int32_t x){
    const uint64_t Q=((uint64_t)1<<61)-1;
    if(x>=0) return fqreduce((unsigned __int128)(uint64_t)x);
    uint64_t r=fqreduce((unsigned __int128)(uint64_t)(-(int64_t)x)); return r==0?0:Q-r;
}
__global__ void combine_modq(const int32_t* P,const int32_t* Q,uint64_t* Chat,uint32_t m,uint32_t n){
    size_t gid=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(gid>=(size_t)m*m) return;
    uint32_t a=gid/m, c=gid%m;
    const int32_t* prow=P+(size_t)a*n;
    uint64_t acc=0;
    for(uint32_t k=0;k<n;k++) acc=fqadd(acc,fqmul(fqfroms32(prow[k]),fqfroms32(Q[(size_t)k*m+c])));
    Chat[gid]=acc;
}

// Tiled mod-q GEMM: Chat[a][c] = sum_k P[a][k]*Q[k][c] over F_q. Coalesced loads
// of P (row-major, contiguous in k) and Q (row-major, contiguous in c) into
// shared memory, reused across the CT-tile. Byte-identical to combine_modq
// (F_q add is associative/commutative, so tile order does not change the sum).
#define CT 16
__global__ void combine_modq_tiled(const int32_t* __restrict__ P, const int32_t* __restrict__ Q,
                                   uint64_t* __restrict__ Chat, uint32_t m, uint32_t n){
    __shared__ uint64_t Ps[CT][CT];
    __shared__ uint64_t Qs[CT][CT];
    uint32_t row = blockIdx.y*CT + threadIdx.y; // a
    uint32_t col = blockIdx.x*CT + threadIdx.x; // c
    uint64_t acc = 0;
    for(uint32_t k0=0; k0<n; k0+=CT){
        uint32_t pk = k0 + threadIdx.x;         // warp varies tx -> contiguous in k (coalesced)
        Ps[threadIdx.y][threadIdx.x] = (row<m && pk<n) ? fqfroms32(P[(size_t)row*n + pk]) : 0;
        uint32_t qk = k0 + threadIdx.y;         // warp varies tx -> contiguous in c (coalesced)
        Qs[threadIdx.y][threadIdx.x] = (qk<n && col<m) ? fqfroms32(Q[(size_t)qk*m + col]) : 0;
        __syncthreads();
        #pragma unroll
        for(int kk=0; kk<CT; kk++) acc = fqadd(acc, fqmul(Ps[threadIdx.y][kk], Qs[kk][threadIdx.x]));
        __syncthreads();
    }
    if(row<m && col<m) Chat[(size_t)row*m + col] = acc;
}

// ===================== C-13 limb-decomposition combine (PRIVATE — matador edge) =====================
// Chat = P*Q mod q via 25 exact s8->s32 tensor-core GEMMs on base-128 limbs of the
// biased operands (P'=P+2^30), byte-exact to combine_modq_tiled (proven in c13_proof.cpp).
// Moves the length-n reduction onto tensor cores; only O(m^2) shift/correct on the ALU.
__device__ __forceinline__ uint64_t fqsub(uint64_t a,uint64_t b){const uint64_t Qp=((uint64_t)1<<61)-1;return a>=b?a-b:a+Qp-b;}
#define C13L 5
__global__ void c13_split(const int32_t* X, int8_t* limbs, size_t count, int32_t bias){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=count) return;
    uint32_t v=(uint32_t)(X[idx]+bias);
    #pragma unroll
    for(int i=0;i<C13L;i++) limbs[(size_t)i*count+idx]=(int8_t)((v>>(7*i))&0x7f);
}
__global__ void c13_rowsum(const int32_t* P, uint64_t* rowP, uint32_t m, uint32_t n, int32_t bias){
    uint32_t a=blockIdx.x*blockDim.x+threadIdx.x; if(a>=m) return;
    const int32_t* r=P+(size_t)a*n; uint64_t acc=0;
    for(uint32_t k=0;k<n;k++) acc=fqadd(acc,fqfroms32((int32_t)(r[k]+bias)));
    rowP[a]=acc;
}
__global__ void c13_colsum(const int32_t* Qm, uint64_t* colQ, uint32_t n, uint32_t m, int32_t bias){
    uint32_t c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=m) return;
    uint64_t acc=0; for(uint32_t k=0;k<n;k++) acc=fqadd(acc,fqfroms32((int32_t)(Qm[(size_t)k*m+c]+bias)));
    colQ[c]=acc;
}
__global__ void c13_reconstruct(const int32_t* G, const uint64_t* rowP, const uint64_t* colQ,
                                uint64_t* Chat, uint32_t m, uint64_t Bq, uint64_t nB2){
    uint32_t a=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(a>=m||c>=m) return;
    uint64_t pw[2*C13L-1]; pw[0]=1;
    #pragma unroll
    for(int t=1;t<2*C13L-1;t++) pw[t]=fqmul(pw[t-1],128);
    uint64_t acc=0;
    #pragma unroll
    for(int i=0;i<C13L;i++) for(int j=0;j<C13L;j++){
        int32_t g=G[((size_t)(i*C13L+j)*m + a)*m + c];
        acc=fqadd(acc, fqmul(pw[i+j], fqfroms32(g)));
    }
    acc=fqsub(acc, fqmul(Bq,rowP[a]));
    acc=fqsub(acc, fqmul(Bq,colQ[c]));
    acc=fqadd(acc, nB2);
    Chat[(size_t)a*m+c]=acc;
}
// --- FUSED C-13: replace the 25 (m x m) GEMMs with ONE (5m x n)*(n x 5m). dPlimb
// is already (5m x n) (P-limbs stacked vertically); write Q-limbs HORIZONTALLY so
// dQlh is (n x 5m); one GEMM yields the full 5x5 grid, block (i,j) = Pl_i*Ql_j at
// output rows [i*m..], cols [j*m..]. Byte-identical products, just fused layout. ---
__global__ void c13_split_qh(const int32_t* X, int8_t* qlh, uint32_t n, uint32_t mcol, int32_t bias){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t count=(size_t)n*mcol; if(idx>=count) return;
    uint32_t r=(uint32_t)(idx/mcol), c=(uint32_t)(idx%mcol);
    uint32_t v=(uint32_t)(X[idx]+bias);
    #pragma unroll
    for(int j=0;j<C13L;j++) qlh[(size_t)r*(C13L*mcol) + (size_t)j*mcol + c]=(int8_t)((v>>(7*j))&0x7f);
}
__global__ void c13_reconstruct_f(const int32_t* Gbig, const uint64_t* rowP, const uint64_t* colQ,
                                  uint64_t* Chat, uint32_t m, uint64_t Bq, uint64_t nB2){
    uint32_t a=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(a>=m||c>=m) return;
    const size_t M5=(size_t)C13L*m;
    uint64_t pw[2*C13L-1]; pw[0]=1;
    #pragma unroll
    for(int t=1;t<2*C13L-1;t++) pw[t]=fqmul(pw[t-1],128);
    uint64_t acc=0;
    #pragma unroll
    for(int i=0;i<C13L;i++) for(int j=0;j<C13L;j++){
        int32_t g=Gbig[(size_t)(i*m+a)*M5 + (size_t)(j*m+c)];
        acc=fqadd(acc, fqmul(pw[i+j], fqfroms32(g)));
    }
    acc=fqsub(acc, fqmul(Bq,rowP[a]));
    acc=fqsub(acc, fqmul(Bq,colQ[c]));
    acc=fqadd(acc, nB2);
    Chat[(size_t)a*m+c]=acc;
}
// --- BMX4-C NATIVE combine: 4 BALANCED base-64 digits (design 5.2, remainder-top
// rule), each plane a signed s8 in [-32,32]. Byte-identical Chat to the 5-limb
// base-128 path (both == ComputeCombineModQ) but: 16 limb-pairs not 25 (fused GEMM
// 4m x 4m not 5m x 5m = 36% fewer MACs), and BALANCED digits carry no +2^30 bias
// => NO rowsum/colsum, and reconstruct is a bare weighted sum with power-of-2
// weights w_ij = 64^(i+j) = 2^(6(i+j)) (all exps 6*(i+j) <= 36 < 61). Ports
// ComputeCombineLimbTensorBMX4C verbatim; fqfroms32 already maps signed->Fq. ---
#define BMXL 4
// ===================== BMX4-C M11 / E8M0 operand sampler ON DEVICE ===========
// Byte-exact port of the host bmxMant/bmxScale/bmxA/bmxB/bmxProj (which reproduce
// the reference matmul_v4_bmx4.cpp golden 4e192d8b). The mantissa stream uses
// REJECTION sampling (11/16 nibbles accepted) -> count/scan/scatter compaction,
// same shape as the s8 wide stream. The scale stream has NO rejection (every
// 2-bit value accepted) -> direct index map, no scan.
//   msg = seed_le(32) || domain(1) || LE64(block).  domain 'm'=0x6D / 'e'=0x65.
__device__ __forceinline__ int8_t bmx_nib(uint8_t nib, bool& acc){
    uint8_t s=(nib>>3)&1, e=(nib>>1)&3, m=nib&1; int mag=0; bool integer=true;
    switch(e){case 0:mag=0;integer=(m==0);break;case 1:mag=1;integer=(m==0);break;
              case 2:mag=(m==0)?2:3;break;case 3:mag=(m==0)?4:6;break;}
    if(!integer||(s&&mag==0)){acc=false;return 0;} acc=true; return (int8_t)(s?-mag:mag);
}
// one thread = one 64-nibble hash block: decode, store accepted M11 mantissas to
// scratch (max 64/block), publish the per-block count. Then scan+scatter compact.
__global__ void bmx_mant_count(const uint8_t* seed32, int8_t* scratch, uint32_t* cnt, uint32_t nblk){
    uint32_t b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nblk) return;
    uint8_t msg[41];
    #pragma unroll
    for(int i=0;i<32;i++) msg[i]=seed32[31-i];             // SeedBytesLE
    msg[32]=0x6D; uint64_t blk=b;
    #pragma unroll
    for(int i=0;i<8;i++) msg[33+i]=(uint8_t)(blk>>(8*i));
    uint8_t h[32]; sha256_1blk(msg,41,h);
    uint32_t c=0;
    for(int i=0;i<32;i++){
        uint8_t nb0=h[i]&0xF, nb1=(h[i]>>4)&0xF;           // low nibble first
        bool a; int8_t v;
        v=bmx_nib(nb0,a); if(a) scratch[(size_t)b*64+c++]=v;
        v=bmx_nib(nb1,a); if(a) scratch[(size_t)b*64+c++]=v;
    }
    cnt[b]=c;
}
__global__ void bmx_mant_scatter(const int8_t* scratch, const uint32_t* off, const uint32_t* cnt,
                                 int8_t* out, uint32_t nblk, size_t count){
    uint32_t b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nblk) return;
    uint32_t o=off[b], c=cnt[b];
    for(uint32_t r=0;r<c;r++){ size_t pos=(size_t)o+r; if(pos<count) out[pos]=scratch[(size_t)b*64+r]; }
}
// scale stream: element i -> hash block i/128, byte (i%128)/4, shift ((i%128)%4)*2.
// 2 bits, no rejection. One thread per output scale (uint8 in {0,1,2,3}).
__global__ void bmx_scale(const uint8_t* seed32, uint8_t* out, size_t count){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=count) return;
    uint64_t blk=i/128; uint32_t j=(uint32_t)(i%128), byte=j/4, sh=(j%4)*2;
    uint8_t msg[41];
    #pragma unroll
    for(int k=0;k<32;k++) msg[k]=seed32[31-k];
    msg[32]=0x65;
    #pragma unroll
    for(int k=0;k<8;k++) msg[33+k]=(uint8_t)(blk>>(8*k));
    uint8_t h[32]; sha256_1blk(msg,41,h);
    out[i]=(uint8_t)((h[byte]>>sh)&3);
}
// dequant A: scale per-32 along COLUMNS. o[i*n+k] = mu[i*n+k] << sc[i*nblk + k/32].
__global__ void bmx_dq_A(const int8_t* mu, const uint8_t* sc, int8_t* o, uint32_t n){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)n*n) return;
    uint32_t i=(uint32_t)(idx/n), k=(uint32_t)(idx%n), nblk=n/32;
    o[idx]=(int8_t)((int32_t)mu[idx]*(1<<sc[(size_t)i*nblk + k/32]));
}
// dequant B: scale per-32 along ROWS. o[k*n+j] = mu[k*n+j] << sc[(k/32)*n + j].
__global__ void bmx_dq_B(const int8_t* mu, const uint8_t* sc, int8_t* o, uint32_t n){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)n*n) return;
    uint32_t k=(uint32_t)(idx/n), j=(uint32_t)(idx%n);
    o[idx]=(int8_t)((int32_t)mu[idx]*(1<<sc[(size_t)(k/32)*n + j]));
}

__global__ void bmx4_split(const int32_t* X, int8_t* planes, size_t count){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=count) return;
    int32_t x=X[idx];                       // low 3 balanced digits d=((x+32)&63)-32 in [-32,31]
    #pragma unroll
    for(int l=0;l<BMXL-1;l++){ int32_t d=((x+32)&63)-32; planes[(size_t)l*count+idx]=(int8_t)d; x=(x-d)/64; }
    planes[(size_t)(BMXL-1)*count+idx]=(int8_t)x;   // remainder-top digit in [-32,32]
}
__global__ void bmx4_split_qh(const int32_t* X, int8_t* qlh, uint32_t n, uint32_t mcol){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t count=(size_t)n*mcol; if(idx>=count) return;
    uint32_t r=(uint32_t)(idx/mcol), c=(uint32_t)(idx%mcol); int32_t x=X[idx];   // Q-planes stacked HORIZONTALLY -> (n x 4m)
    #pragma unroll
    for(int l=0;l<BMXL-1;l++){ int32_t d=((x+32)&63)-32; qlh[(size_t)r*(BMXL*mcol)+(size_t)l*mcol+c]=(int8_t)d; x=(x-d)/64; }
    qlh[(size_t)r*(BMXL*mcol)+(size_t)(BMXL-1)*mcol+c]=(int8_t)x;
}
__global__ void bmx4_reconstruct(const int32_t* Gbig, uint64_t* Chat, uint32_t m){
    uint32_t a=blockIdx.y*blockDim.y+threadIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x;
    if(a>=m||c>=m) return;
    const size_t M4=(size_t)BMXL*m;
    uint64_t acc=0;                          // Chat = sum_ij 2^(6(i+j)) * FqFromSigned(S_ij), no bias term
    #pragma unroll
    for(int i=0;i<BMXL;i++) for(int j=0;j<BMXL;j++){
        int32_t g=Gbig[(size_t)(i*m+a)*M4 + (size_t)(j*m+c)];
        acc=fqadd(acc, fqmul((uint64_t)1<<(6*(i+j)), fqfroms32(g)));
    }
    Chat[(size_t)a*m+c]=acc;
}
// position-mixed XOR fold of a u64 buffer -> one u64 (order-independent equality
// check that still catches permutations, for serial-vs-pipelined byte-exactness)
__global__ void xor_fold64(const unsigned long long* x, size_t count, unsigned long long* acc){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long v=0;
    for(size_t k=i;k<count;k+=(size_t)gridDim.x*blockDim.x) v ^= x[k]*(2ull*k+1ull);
    if(v) atomicXor(acc,v);
}

// scalar INT8 GEMM (verify only): C[MxN]=A[MxK]*B[KxN] row-major
__global__ void gemm_scalar(const int8_t*A,const int8_t*B,int32_t*C,uint32_t M,uint32_t N,uint32_t Kd){
    size_t gid=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(gid>=(size_t)M*N)return;
    uint32_t r=gid/N,c=gid%N; const int8_t*ar=A+(size_t)r*Kd; int32_t acc=0;
    for(uint32_t k=0;k<Kd;k++) acc+=(int32_t)ar[k]*(int32_t)B[(size_t)k*N+c];
    C[gid]=acc;
}

// ===================== cuBLASLt INT8 GEMM (row-major, exact s8->s32) =========
bool int8_gemm(cublasLtHandle_t lt,cudaStream_t s,void*ws,size_t wsz,
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
        if(st){printf("  [cublasLt] matmul fail status=%d (M=%u N=%u K=%u)\n",(int)st,M,N,Kd);break;}
        ok=true;
    }while(0);
    if(pref)cublasLtMatmulPreferenceDestroy(pref);
    if(lc)cublasLtMatrixLayoutDestroy(lc); if(lb)cublasLtMatrixLayoutDestroy(lb); if(la)cublasLtMatrixLayoutDestroy(la);
    if(op)cublasLtMatmulDescDestroy(op);
    return ok;
}

// ---- host SHA-256 (arbitrary length) for the digest byte-exact cross-check ----
static void hsha256(const uint8_t* d, size_t len, uint8_t out[32]){
    static const uint32_t Kh[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    auto R=[](uint32_t x,int n){return (x>>n)|(x<<(32-n));};
    uint32_t s[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<uint8_t> m(d,d+len);
    uint64_t bits=(uint64_t)len*8; m.push_back(0x80);
    while(m.size()%64!=56) m.push_back(0);
    for(int i=7;i>=0;i--) m.push_back((uint8_t)(bits>>(8*i)));
    for(size_t off=0;off<m.size();off+=64){
        uint32_t w[64];
        for(int i=0;i<16;i++) w[i]=(m[off+i*4]<<24)|(m[off+i*4+1]<<16)|(m[off+i*4+2]<<8)|m[off+i*4+3];
        for(int i=16;i<64;i++){uint32_t a=R(w[i-15],7)^R(w[i-15],18)^(w[i-15]>>3),b=R(w[i-2],17)^R(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+a+w[i-7]+b;}
        uint32_t a=s[0],b=s[1],c=s[2],dd=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        for(int i=0;i<64;i++){uint32_t S1=R(e,6)^R(e,11)^R(e,25),ch=(e&f)^((~e)&g),t1=h+S1+ch+Kh[i]+w[i],S0=R(a,2)^R(a,13)^R(a,22),mj=(a&b)^(a&c)^(b&c),t2=S0+mj;h=g;g=f;f=e;e=dd+t1;dd=c;c=b;b=a;a=t1+t2;}
        s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=dd;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
    }
    for(int i=0;i<8;i++){out[i*4]=s[i]>>24;out[i*4+1]=s[i]>>16;out[i*4+2]=s[i]>>8;out[i*4+3]=s[i];}
}
// digest = SHA256d("BTX_MATMUL_V4" || sigma[32] || payload)  (matmul_v4.cpp ComputeSketchDigest)
static void sketch_digest(const uint8_t sigma[32], const uint8_t* payload, size_t plen, uint8_t out[32]){
    static const char tag[]="BTX_MATMUL_V4"; // 13 bytes, no null
    std::vector<uint8_t> buf; buf.insert(buf.end(),tag,tag+13);
    buf.insert(buf.end(),sigma,sigma+32); buf.insert(buf.end(),payload,payload+plen);
    uint8_t d1[32]; hsha256(buf.data(),buf.size(),d1); hsha256(d1,32,out);
}

// ===================== BMX4-C (v4.2) operand generation (host, byte-exact port of
//   src/matmul/matmul_v4_bmx4.cpp: M11 sampler + E8M0 scales). Reused GEMM/combine/
//   digest below reproduce the reference golden 4e192d8b (n=256 nonce=1). ==========
static const uint8_t kBmxMantDom=0x6D, kBmxScaleDom=0x65;   // 'm','e' XOF domains
static const uint32_t kBmxBlk=32;                            // E8M0 block length
static std::vector<uint8_t> hex2b(const std::string& h){ std::vector<uint8_t> b; for(size_t i=0;i+1<h.size();i+=2) b.push_back((uint8_t)std::stoi(h.substr(i,2),nullptr,16)); return b; }
static void wle64(uint8_t* o,uint64_t v){ for(int i=0;i<8;i++) o[i]=(uint8_t)(v>>(8*i)); }
// E2M1 nibble -> (accepted, M11 mantissa). Matches MantissaTable in the reference.
static int8_t bmxNib(uint8_t nib,bool& acc){ uint8_t s=(nib>>3)&1,e=(nib>>1)&3,m=nib&1; int mag=0; bool integer=true;
  switch(e){case 0:mag=0;integer=(m==0);break;case 1:mag=1;integer=(m==0);break;case 2:mag=(m==0)?2:3;break;case 3:mag=(m==0)?4:6;break;}
  if(!integer||(s&&mag==0)){acc=false;return 0;} acc=true; return (int8_t)(s?-mag:mag); }
// Wide counter-mode SHA XOF: seed(32) || domain(1) || LE64(block). Low nibble then high.
static void bmxMant(const std::vector<uint8_t>& seed,size_t count,int8_t* out){
  size_t filled=0; uint64_t blk=0; uint8_t msg[41]; memcpy(msg,seed.data(),32); msg[32]=kBmxMantDom;
  while(filled<count){ wle64(msg+33,blk); uint8_t h[32]; hsha256(msg,41,h);
    for(int i=0;i<32&&filled<count;i++){ uint8_t nb[2]={(uint8_t)(h[i]&0xF),(uint8_t)((h[i]>>4)&0xF)};
      for(uint8_t x:nb){ bool a=false; int8_t mu=bmxNib(x,a); if(a){ out[filled++]=mu; if(filled==count) break; } } }
    ++blk; } }
static void bmxScale(const std::vector<uint8_t>& seed,size_t count,uint8_t* out){
  size_t filled=0; uint64_t blk=0; uint8_t msg[41]; memcpy(msg,seed.data(),32); msg[32]=kBmxScaleDom;
  while(filled<count){ wle64(msg+33,blk); uint8_t h[32]; hsha256(msg,41,h);
    for(int i=0;i<32&&filled<count;i++){ for(int sh=0;sh<8&&filled<count;sh+=2) out[filled++]=(uint8_t)((h[i]>>sh)&3); }
    ++blk; } }
static inline int8_t bmxDq(int8_t mu,uint8_t e){ return (int8_t)((int32_t)mu*(1<<e)); }
static std::vector<int8_t> bmxA(const std::vector<uint8_t>& sd,uint32_t n){   // A: scale per-32 along cols
  size_t cnt=(size_t)n*n; std::vector<int8_t> mu(cnt); bmxMant(sd,cnt,mu.data());
  uint32_t nblk=n/kBmxBlk; std::vector<uint8_t> sc((size_t)n*nblk); bmxScale(sd,sc.size(),sc.data());
  std::vector<int8_t> o(cnt); for(uint32_t i=0;i<n;i++){size_t r=(size_t)i*n,sr=(size_t)i*nblk; for(uint32_t k=0;k<n;k++) o[r+k]=bmxDq(mu[r+k],sc[sr+(k/kBmxBlk)]);} return o; }
static std::vector<int8_t> bmxB(const std::vector<uint8_t>& sd,uint32_t n){   // B: scale per-32 along rows
  size_t cnt=(size_t)n*n; std::vector<int8_t> mu(cnt); bmxMant(sd,cnt,mu.data());
  uint32_t nblk=n/kBmxBlk; std::vector<uint8_t> sc((size_t)nblk*n); bmxScale(sd,sc.size(),sc.data());
  std::vector<int8_t> o(cnt); for(uint32_t k=0;k<n;k++){size_t r=(size_t)k*n,sr=(size_t)(k/kBmxBlk)*n; for(uint32_t j=0;j<n;j++) o[r+j]=bmxDq(mu[r+j],sc[sr+j]);} return o; }
static std::vector<int8_t> bmxProj(const std::vector<uint8_t>& sd,uint32_t rows,uint32_t cols){ // U/V scale-free M11
  size_t cnt=(size_t)rows*cols; std::vector<int8_t> o(cnt); bmxMant(sd,cnt,o.data()); return o; }

int main(int argc,char**argv){
    uint32_t n = argc>1?atoi(argv[1]):4096;
    int nonces = argc>2?atoi(argv[2]):32;
    bool verify = false, emit = false, wide = false, c13 = false, gate = false;
    int bovr=0; bool bmx4c=false, cprof=false, overlap=false;  // --b <2|4>; --bmx4c: golden emit; --cprof: combine sub-stage timing; --overlap: S1b/tensor pipeline proto
    bool pipe=false; int kdovr=0;               // --pipe: deploy pipeline (overlap + S4); --kd <chains>: digest ring size
    bool solve=false;                           // --solve: header->digest, GPU M11/E8M0 sampler, all 3 goldens
    for(int i=1;i<argc;i++){ std::string a=argv[i]; if(a=="--verify")verify=true; if(a=="--emit")emit=true; if(a=="--wide"||a=="--xof=wide")wide=true; if(a=="--c13")c13=true; if(a=="--gate")gate=true; if(a=="--bmx4c")bmx4c=true; if(a=="--cprof")cprof=true; if(a=="--overlap")overlap=true; if(a=="--b"&&i+1<argc)bovr=atoi(argv[++i]); if(a=="--pipe")pipe=true; if(a=="--kd"&&i+1<argc)kdovr=atoi(argv[++i]); if(a=="--solve")solve=true; }
    if(solve){ bovr=4; n=256; c13=true; }       // ENC-BMX4C C profile (m=n/4); size globals for the largest golden; c13 => allocate combine buffers
    if(bmx4c){ bovr=4; }  // ENC-BMX4C C profile: b=kTileB=4 -> m=n/4
    if(gate||overlap){ c13=true; wide=true; }  // gate/overlap always need the tensor-combine path + the real XOF
    if(pipe){ c13=true; wide=true; }  // pipe uses the BMX4 combine buffers + the real XOF
    const uint32_t b = bovr ? (uint32_t)bovr : ((gate||pipe||overlap) ? 4 : 8);  // gate/pipe/overlap default kTileB=4 (ENC-BMX4C, m=n/4); --b 2 = ENC-BMX4C-D (m=n/2)
    const uint32_t m=n/b;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    bool nvml_ok=false; nvmlDevice_t nvdev;
    if(nvmlInit_v2()==NVML_SUCCESS && nvmlDeviceGetHandleByIndex_v2(0,&nvdev)==NVML_SUCCESS) nvml_ok=true;
    printf("=== v4 bench | %s | n=%u m=%u nonces=%d xof=%s ===\n", prop.name, n,m,nonces, wide?"wide":"legacy");

    size_t nn=(size_t)n*n, mn=(size_t)m*n, nm=(size_t)n*m, mm=(size_t)m*m;
    int8_t *dA,*dB,*dU,*dV; int32_t *dP,*dQ,*dPs,*dQs; uint64_t *dChat,*dChatS;
    uint8_t *dSeedA,*dSeedB,*dSeedU,*dSeedV; void* ws;
    CK(cudaMalloc(&dA,nn));CK(cudaMalloc(&dB,nn));CK(cudaMalloc(&dU,mn));CK(cudaMalloc(&dV,nm));
    CK(cudaMalloc(&dP,mn*4));CK(cudaMalloc(&dQ,nm*4));CK(cudaMalloc(&dChat,mm*8));
    CK(cudaMalloc(&dSeedA,32));CK(cudaMalloc(&dSeedB,32));CK(cudaMalloc(&dSeedU,32));CK(cudaMalloc(&dSeedV,32));
    CK(cudaMalloc(&ws,(size_t)32<<20));
    // C-13 limb-decomposition buffers (5 base-128 limbs, 25 s8 GEMMs)
    int8_t *dPlimb=nullptr,*dQlimb=nullptr; int32_t* dG=nullptr; uint64_t *dRowP=nullptr,*dColQ=nullptr;
    int8_t* dQlh=nullptr;  // fused path: Q-limbs stacked HORIZONTALLY, (n x 5m)
    if(c13||bmx4c){ CK(cudaMalloc(&dPlimb,(size_t)C13L*mn)); CK(cudaMalloc(&dQlimb,(size_t)C13L*nm));
             CK(cudaMalloc(&dG,(size_t)C13L*C13L*mm*4)); CK(cudaMalloc(&dRowP,m*8)); CK(cudaMalloc(&dColQ,m*8));
             CK(cudaMalloc(&dQlh,(size_t)C13L*nm)); }
    // bit-exact wide-stream compaction buffers (sized for the largest operand, nn)
    uint32_t maxblk=(uint32_t)(nn/31 + 64);
    int8_t* dScratch; uint32_t *dCnt,*dOff;
    CK(cudaMalloc(&dScratch,(size_t)maxblk*32)); CK(cudaMalloc(&dCnt,maxblk*4)); CK(cudaMalloc(&dOff,maxblk*4));
    if(verify){CK(cudaMalloc(&dPs,mn*4));CK(cudaMalloc(&dQs,nm*4));CK(cudaMalloc(&dChatS,mm*8));}
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s; CK(cudaStreamCreate(&s));

    auto seed=[&](uint8_t*d,int base){uint8_t h[32];for(int i=0;i<32;i++)h[i]=base+i;CK(cudaMemcpy(d,h,32,cudaMemcpyHostToDevice));};
    int T=256;
    // bit-exact wide counter-mode XOF, matches f50f0f8 ExpandBalancedS8Stream (count -> scan -> scatter)
    auto gen_stream=[&](int8_t* out, uint8_t* dSeed, uint32_t count, uint8_t domain){
        uint32_t nblk=count/31 + 64;
        stream_count<<<(nblk+T-1)/T,T,0,s>>>(dSeed,domain,dScratch,dCnt,nblk);
        thrust::exclusive_scan(thrust::cuda::par.on(s), dCnt, dCnt+nblk, dOff);
        stream_scatter<<<(nblk+T-1)/T,T,0,s>>>(dScratch,dOff,dCnt,out,nblk,count);
    };
    auto genop=[&](int8_t* d, uint8_t* sd, size_t cnt){
        if(wide) gen_stream(d,sd,(uint32_t)cnt,0x73);                               // bit-exact wide stream (f50f0f8)
        else     gen_operand<<<((uint32_t)cnt+T-1)/T,T,0,s>>>(d,sd,(uint32_t)cnt);  // legacy per-element (pre-fix)
    };
    auto launch_combine=[&](int32_t* P,int32_t* Q,uint64_t* C){ dim3 bl(CT,CT), gr((m+CT-1)/CT,(m+CT-1)/CT); combine_modq_tiled<<<gr,bl,0,s>>>(P,Q,C,m,n); };
    // C-13: split -> 25 s8 GEMMs (G[i][j]=Pl_i*Ql_j) -> reconstruct (byte-exact to launch_combine)
    const uint64_t QP=((uint64_t)1<<61)-1; const uint64_t Bq=(uint64_t)1<<30;
    const uint64_t nB2=(uint64_t)((unsigned __int128)n*((unsigned __int128)Bq*Bq%QP)%QP);
    auto c13_combine=[&](int32_t* P,int32_t* Q,uint64_t* C){
        const int32_t BIAS=1<<30;
        c13_split<<<((uint32_t)mn+T-1)/T,T,0,s>>>(P,dPlimb,mn,BIAS);
        c13_split<<<((uint32_t)nm+T-1)/T,T,0,s>>>(Q,dQlimb,nm,BIAS);
        c13_rowsum<<<(m+T-1)/T,T,0,s>>>(P,dRowP,m,n,BIAS);
        c13_colsum<<<(m+T-1)/T,T,0,s>>>(Q,dColQ,n,m,BIAS);
        for(int i=0;i<C13L;i++) for(int j=0;j<C13L;j++)
            int8_gemm(lt,s,ws,(size_t)32<<20, dPlimb+(size_t)i*mn, dQlimb+(size_t)j*nm, dG+(size_t)(i*C13L+j)*mm, m,m,n);
        dim3 bl(16,16), gr((m+15)/16,(m+15)/16);
        c13_reconstruct<<<gr,bl,0,s>>>(dG,dRowP,dColQ,C,m,Bq,nB2);
    };
    // FUSED C-13: identical products/reconstruct math, but ONE (5m x n)*(n x 5m)
    // GEMM instead of 25 (m x m) GEMMs. dG is reused as the (5m x 5m) output
    // (same 25*mm*4 bytes). Byte-exact to c13_combine (verified in --gate).
    // combine sub-stage profiler (only when --cprof): isolates the elementwise
    // split/reconstruct passes from the fused GEMM inside S3, so we know which to cut.
    static float cp_split=0,cp_gemm=0,cp_recon=0; static int cp_calls=0;
    cudaEvent_t cpa,cpb,cpc,cpd; if(cprof){CK(cudaEventCreate(&cpa));CK(cudaEventCreate(&cpb));CK(cudaEventCreate(&cpc));CK(cudaEventCreate(&cpd));}
    auto c13_fused=[&](int32_t* P,int32_t* Q,uint64_t* C){
        const int32_t BIAS=1<<30;
        if(cprof) CK(cudaEventRecord(cpa,s));
        c13_split<<<((uint32_t)mn+T-1)/T,T,0,s>>>(P,dPlimb,mn,BIAS);          // (5m x n) vertical stack
        c13_split_qh<<<((uint32_t)nm+T-1)/T,T,0,s>>>(Q,dQlh,n,m,BIAS);        // (n x 5m) horizontal stack
        c13_rowsum<<<(m+T-1)/T,T,0,s>>>(P,dRowP,m,n,BIAS);
        c13_colsum<<<(m+T-1)/T,T,0,s>>>(Q,dColQ,n,m,BIAS);
        if(cprof) CK(cudaEventRecord(cpb,s));
        int8_gemm(lt,s,ws,(size_t)32<<20, dPlimb, dQlh, dG, C13L*m, C13L*m, n); // ONE GEMM
        if(cprof) CK(cudaEventRecord(cpc,s));
        dim3 bl(16,16), gr((m+15)/16,(m+15)/16);
        c13_reconstruct_f<<<gr,bl,0,s>>>(dG,dRowP,dColQ,C,m,Bq,nB2);
        if(cprof){ CK(cudaEventRecord(cpd,s)); CK(cudaStreamSynchronize(s));
            float a,b2,c2; CK(cudaEventElapsedTime(&a,cpa,cpb));CK(cudaEventElapsedTime(&b2,cpb,cpc));CK(cudaEventElapsedTime(&c2,cpc,cpd));
            if(cp_calls>0){cp_split+=a;cp_gemm+=b2;cp_recon+=c2;} cp_calls++; }
    };
    // BMX4-C NATIVE fused combine: 4 balanced base-64 planes, ONE (4m x n)*(n x 4m)
    // GEMM (16 blocks), bare weighted reconstruct. Reuses dPlimb/dQlh/dG (all sized
    // for 5 limbs, so 4 fits). Byte-exact to launch_combine (checked in --gate).
    auto bmx4_combine=[&](int32_t* P,int32_t* Q,uint64_t* C){
        bmx4_split<<<((uint32_t)mn+T-1)/T,T,0,s>>>(P,dPlimb,mn);            // (4m x n) vertical stack
        bmx4_split_qh<<<((uint32_t)nm+T-1)/T,T,0,s>>>(Q,dQlh,n,m);         // (n x 4m) horizontal stack
        int8_gemm(lt,s,ws,(size_t)32<<20, dPlimb, dQlh, dG, BMXL*m, BMXL*m, n); // ONE GEMM, 16 blocks
        dim3 bl(16,16), gr((m+15)/16,(m+15)/16);
        bmx4_reconstruct<<<gr,bl,0,s>>>(dG,C,m);
    };
    auto combine=[&](int32_t* P,int32_t* Q,uint64_t* C){ if(c13) c13_combine(P,Q,C); else launch_combine(P,Q,C); };
    float t_gen=0,t_gemm=0,t_comb=0; double pw_sum=0; long pw_n=0;
    cudaEvent_t e0,e1,e2,e3; CK(cudaEventCreate(&e0));CK(cudaEventCreate(&e1));CK(cudaEventCreate(&e2));CK(cudaEventCreate(&e3));

    // ---- --gate: v4.1 batched-sketch per-nonce MARGINAL stage split on the GPU
    //      (reference bench/matmul_v4_stage_bench.cpp §K.2a-WT methodology).
    //      S0 (A,U,V + P=U*A) is TEMPLATE-scoped -> amortized out. Per nonce we
    //      time S1b (expand B, integer/SHA), S2 (Q=B*V, tensor), and BOTH combine
    //      choices: S3' ALU-direct (integer) vs S3b C-13 limb-tensor. A miner takes
    //      min(S3',S3b); which one a 5090 picks decides whether FP4 (which only
    //      speeds the TENSOR stages) even touches the combine. That is the gate. ----
    if(gate){
        const int Q = nonces>1?nonces:8;                 // window size
        // S0 (once): A,U,V expand + P=U*A. Amortized over the window.
        seed(dSeedA,7); seed(dSeedU,50); seed(dSeedV,200);
        CK(cudaEventRecord(e0,s));
        genop(dA,dSeedA,nn); genop(dU,dSeedU,mn); genop(dV,dSeedV,nm);
        int8_gemm(lt,s,ws,(size_t)32<<20,dU,dA,dP,m,n,n);              // P = U*A  (m x n, K=n)
        CK(cudaEventRecord(e1,s)); CK(cudaStreamSynchronize(s));
        float s0=0; CK(cudaEventElapsedTime(&s0,e0,e1));

        // Byte-exact gate first: all three combine paths must agree on Chat, or the
        // times are void (a stage only counts if it computes the consensus bytes).
        {
            seed(dSeedB,100);
            genop(dB,dSeedB,nn);
            int8_gemm(lt,s,ws,(size_t)32<<20,dB,dV,dQ,n,m,n);
            std::vector<uint64_t> h_alu(mm),h_c13(mm),h_fus(mm),h_bmx(mm);
            launch_combine(dP,dQ,dChat); CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(h_alu.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
            c13_combine(dP,dQ,dChat);    CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(h_c13.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
            c13_fused(dP,dQ,dChat);      CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(h_fus.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
            bmx4_combine(dP,dQ,dChat);   CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(h_bmx.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
            size_t m1=0,m2=0,m3=0; for(size_t i=0;i<mm;i++){ if(h_c13[i]!=h_alu[i])m1++; if(h_fus[i]!=h_alu[i])m2++; if(h_bmx[i]!=h_alu[i])m3++; }
            printf(" byte-exact: C-13 vs ALU = %zu mism, FUSED vs ALU = %zu mism, BMX4(4b64) vs ALU = %zu mism %s\n",
                   m1,m2,m3,(m1||m2||m3)?"** FAIL - TIMES VOID **":"(all four agree)");
            if(m1||m2||m3) return 1;
        }
        float t_s1b=0,t_s2=0,t_alu=0,t_c13=0,t_fus=0,t_bmx=0; double pw=0; long pwn=0;
        cudaEvent_t ea,eb,ec,ed,ee,ef,eg; CK(cudaEventCreate(&ea));CK(cudaEventCreate(&eb));CK(cudaEventCreate(&ec));CK(cudaEventCreate(&ed));CK(cudaEventCreate(&ee));CK(cudaEventCreate(&ef));CK(cudaEventCreate(&eg));
        for(int it=0; it<Q+1; ++it){                                   // +1 warmup, dropped
            seed(dSeedB,100+it);
            CK(cudaEventRecord(ea,s));
            genop(dB,dSeedB,nn);                                       // S1b: expand B (n x n)
            CK(cudaEventRecord(eb,s));
            int8_gemm(lt,s,ws,(size_t)32<<20,dB,dV,dQ,n,m,n);          // S2: Q = B*V (n x m, K=n) TENSOR
            CK(cudaEventRecord(ec,s));
            launch_combine(dP,dQ,dChat);                              // S3': ALU-direct mod-q (integer)
            CK(cudaEventRecord(ed,s));
            c13_combine(dP,dQ,dChat);                                 // S3b: C-13 limb-tensor, 25 GEMMs (TENSOR)
            CK(cudaEventRecord(ee,s));
            c13_fused(dP,dQ,dChat);                                   // S3f: FUSED single-GEMM C-13 (TENSOR)
            CK(cudaEventRecord(ef,s));
            bmx4_combine(dP,dQ,dChat);                               // S3g: BMX4-C NATIVE 4-base-64 fused (TENSOR)
            CK(cudaEventRecord(eg,s));
            CK(cudaStreamSynchronize(s));
            float a,b2,c2,d2,f2,g2; CK(cudaEventElapsedTime(&a,ea,eb));CK(cudaEventElapsedTime(&b2,eb,ec));
            CK(cudaEventElapsedTime(&c2,ec,ed));CK(cudaEventElapsedTime(&d2,ed,ee));CK(cudaEventElapsedTime(&f2,ee,ef));CK(cudaEventElapsedTime(&g2,ef,eg));
            if(it>0){ t_s1b+=a; t_s2+=b2; t_alu+=c2; t_c13+=d2; t_fus+=f2; t_bmx+=g2;
                if(nvml_ok){ unsigned int mw; if(nvmlDeviceGetPowerUsage(nvdev,&mw)==NVML_SUCCESS){ pw+=mw; pwn++; } } }
        }
        t_s1b/=Q; t_s2/=Q; t_alu/=Q; t_c13/=Q; t_fus/=Q; t_bmx/=Q;
        const float s0_amort = s0/Q;
        float s3=t_alu; const char* s3name="ALU-direct"; bool s3_is_tensor=false;
        if(t_c13<s3){ s3=t_c13; s3name="C-13 25GEMM"; s3_is_tensor=true; }
        if(t_fus<s3){ s3=t_fus; s3name="C-13 FUSED"; s3_is_tensor=true; }
        if(t_bmx<s3){ s3=t_bmx; s3name="BMX4 4b64"; s3_is_tensor=true; }
        const float marg = t_s1b + t_s2 + s3;                          // per-nonce marginal (S4 excl., see note)
        const float tensor_ms = t_s2 + (s3_is_tensor? s3 : 0.0f);
        // achieved INT8 throughput (2 ops per MAC)
        auto tops=[&](double macs,float ms){ return ms>0? 2.0*macs/(ms*1e-3)/1e12 : 0.0; };
        const double s2_macs=(double)nn*m, c13_macs=25.0*(double)mm*n, bmx4_macs=16.0*(double)mm*n; // S2=n*n*m ; C-13/fused=25 ; BMX4=16 * (m*m*n)
        const double avg_w=(nvml_ok&&pwn>0)?(pw/pwn/1000.0):0.0;
        printf("\n=== v4.1 BATCHED-SKETCH GATE | %s | n=%u m=%u window Q=%d ===\n",prop.name,n,m,Q);
        printf(" S0 template A,U,V+P=U*A (amortized/nonce) : %8.3f ms   (paid once/%d)\n", s0_amort, Q);
        printf(" ---- per-nonce MARGINAL ----\n");
        printf(" S1b expand B          (integer/SHA)       : %8.3f ms  %5.1f%%\n", t_s1b, 100*t_s1b/marg);
        printf(" S2  Q=B*V             (TENSOR)            : %8.3f ms  %5.1f%%   %.0f INT8 TOPS\n", t_s2, 100*t_s2/marg, tops(s2_macs,t_s2));
        printf(" S3' combine ALU-direct(integer)          : %8.3f ms\n", t_alu);
        printf(" S3b combine C-13 25-GEMM (TENSOR)        : %8.3f ms   %.0f INT8 TOPS\n", t_c13, tops(c13_macs,t_c13));
        printf(" S3f combine C-13 FUSED 1-GEMM (TENSOR)   : %8.3f ms   %.0f INT8 TOPS   [%.2fx vs 25-GEMM]\n", t_fus, tops(c13_macs,t_fus), t_c13/t_fus);
        printf(" S3g combine BMX4-C 4-base-64 (TENSOR)    : %8.3f ms   %.0f INT8 TOPS   [%.2fx vs FUSED, %.2fx vs 25-GEMM]  <- DEPLOY\n", t_bmx, tops(bmx4_macs,t_bmx), t_fus/t_bmx, t_c13/t_bmx);
        printf(" S3  chosen = %-11s (%s)         : %8.3f ms  %5.1f%%\n", s3name, s3_is_tensor?"tensor":"integer", s3, 100*s3/marg);
        printf(" ------------------------------------------------\n");
        printf(" per-nonce marginal (S4 digest excl.)     : %8.3f ms   %.0f nonce/s   %.0f W\n", marg, 1000.0/marg, avg_w);
        printf(" >> TENSOR share (S2 + chosen-if-tensor)   = %5.1f%%   [gate: majority => FP4 matters]\n", 100*tensor_ms/marg);
        printf(" >> if FP4 gives 2x on tensor stages, marginal -> %.3f ms  (%+.1f%% nonce/s)\n",
               marg - tensor_ms*0.5f, 100.0*((marg)/(marg-tensor_ms*0.5f)-1.0));
        printf("    CAVEAT (v4.1): the C-13 limbs are base-128 digits (0..127) which do NOT fit\n");
        printf("    E2M1 {0,1,2,3,4,6}; base-5 re-decomposition = 196 GEMMs = 7.8x MACs > 2x rate\n");
        printf("    => FP4 can NEVER ride the v4.1 combine. Real v4.1 FP4 surface = S2 only (+%.1f%%).\n",
               100.0*((marg)/(marg-t_s2*0.5f)-1.0));
        printf("    v4.2/BMX4-C exists to fix this: M11 = E2M1's exact-int subset, operands FP4-native.\n");
        printf(" note: S4 (device digest of the %ux%u sketch) not timed here (host SHA unrepresentative);\n", m,m);
        printf("       it is integer work, so the real tensor share is <= the number above.\n");
        printf("CSVGATE,%s,%u,%u,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.1f,%.1f,%.4f\n",
               prop.name,n,m,Q,s0_amort,t_s1b,t_s2,t_alu,t_c13,t_fus,s3_is_tensor?1:0,100*tensor_ms/marg,avg_w,t_bmx);
        if(cprof && cp_calls>1){ int cn=cp_calls-1; float sp=cp_split/cn,gm=cp_gemm/cn,rc=cp_recon/cn,tt=sp+gm+rc;
            printf("\n [--cprof] FUSED combine internals (per call, avg over %d):\n",cn);
            printf("   split+rowsum+colsum (elementwise): %7.3f ms  %5.1f%%\n", sp,100*sp/tt);
            printf("   fused GEMM (tensor)              : %7.3f ms  %5.1f%%\n", gm,100*gm/tt);
            printf("   reconstruct+mod-q (elementwise)  : %7.3f ms  %5.1f%%\n", rc,100*rc/tt);
            printf("   => elementwise overhead = %.1f%% of the combine (the non-GEMM lever)\n", 100*(sp+rc)/tt); }
        if(nvml_ok) nvmlShutdown();
        return 0;
    }

    // ---- --overlap: S1b/tensor PIPELINE prototype. S1b (expand B, SHA/ALU) of
    //      nonce k+1 runs on a second stream behind S2+S3g (tensor) of nonce k,
    //      double-buffered B, event-fenced. Measures the REAL overlap ceiling
    //      before the solver is touched. Byte-exactness: per-nonce Chat checksums,
    //      pipelined arm MUST equal the serial arm on identical seeds, or the
    //      times are void. NOTE: thrust::exclusive_scan hides a cudaMalloc/Free
    //      per call (device-wide sync = would serialize the streams), so both
    //      arms use CUB DeviceScan with a preallocated temp — identical integer
    //      scan, and the B bytes are cross-checked vs the thrust path below. ----
    if(overlap){
        const int Q = nonces>1?nonces:16, W=2;      // measured window + pipeline-fill warmup
        void* dScan=nullptr; size_t scanBytes=0;
        cub::DeviceScan::ExclusiveSum(nullptr,scanBytes,dCnt,dOff,(int)maxblk);
        CK(cudaMalloc(&dScan,scanBytes));
        int genB=0;  // gen grid cap: 0 = full grid; small = co-resident background kernel (byte-identical either way)
        auto gen_cub=[&](int8_t* out, const uint8_t* dSeed, uint32_t count, cudaStream_t st){
            uint32_t nblk=count/31+64; size_t sb=scanBytes;
            uint32_t g = genB? (uint32_t)genB : (nblk+T-1)/T;
            stream_count<<<g,T,0,st>>>(dSeed,0x73,dScratch,dCnt,nblk);
            cub::DeviceScan::ExclusiveSum(dScan,sb,dCnt,dOff,(int)nblk,st);
            stream_scatter<<<g,T,0,st>>>(dScratch,dOff,dCnt,out,nblk,count);
        };
        // psplit=false models the SOLVER: P is template-scoped, so its limb planes
        // (dPlimb) are computed ONCE in S0 and reused per nonce — byte-identical
        // (P never changes inside the window; validated vs the psplit=true checksums).
        auto bmx4_st=[&](int32_t* P,int32_t* Qd,uint64_t* C,cudaStream_t st,bool psplit){
            if(psplit) bmx4_split<<<((uint32_t)mn+T-1)/T,T,0,st>>>(P,dPlimb,mn);
            bmx4_split_qh<<<((uint32_t)nm+T-1)/T,T,0,st>>>(Qd,dQlh,n,m);
            int8_gemm(lt,st,ws,(size_t)32<<20, dPlimb, dQlh, dG, BMXL*m, BMXL*m, n);
            dim3 bl(16,16), gr((m+15)/16,(m+15)/16);
            bmx4_reconstruct<<<gr,bl,0,st>>>(dG,C,m);
        };
        unsigned long long *dAcc,*dSumS,*dSumP;
        CK(cudaMalloc(&dAcc,8)); CK(cudaMalloc(&dSumS,(size_t)Q*8)); CK(cudaMalloc(&dSumP,(size_t)Q*8));
        auto fold=[&](const void* buf,size_t bytes,unsigned long long* acc,cudaStream_t st){
            CK(cudaMemsetAsync(acc,0,8,st));
            xor_fold64<<<256,256,0,st>>>((const unsigned long long*)buf,bytes/8,acc);
        };
        // per-nonce PINNED host seed slots (async H2D source must stay stable while
        // the copy is in flight; host can enqueue many nonces ahead)
        uint8_t* hSeeds; CK(cudaHostAlloc(&hSeeds,(size_t)(W+Q+4)*32,cudaHostAllocDefault));
        for(int it=0;it<W+Q;++it) for(int i=0;i<32;i++) hSeeds[(size_t)it*32+i]=(uint8_t)(100+it+i);
        uint8_t* dSeedB2; CK(cudaMalloc(&dSeedB2,2*32));
        int8_t* dB2[2]; CK(cudaMalloc(&dB2[0],nn)); CK(cudaMalloc(&dB2[1],nn));
        cudaStream_t sMain,sGen,sMainH,sGenL;
        CK(cudaStreamCreateWithFlags(&sMain,cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&sGen,cudaStreamNonBlocking));
        int prLeast=0,prGreatest=0; CK(cudaDeviceGetStreamPriorityRange(&prLeast,&prGreatest));
        CK(cudaStreamCreateWithPriority(&sMainH,cudaStreamNonBlocking,prGreatest));  // tensor stream: high prio
        CK(cudaStreamCreateWithPriority(&sGenL,cudaStreamNonBlocking,prLeast));      // SHA gen: low prio, fills gaps
        cudaEvent_t evGen[2],evFree[2],t0,t1,ea,eb,ec,ed;
        for(int i=0;i<2;i++){ CK(cudaEventCreate(&evGen[i])); CK(cudaEventCreate(&evFree[i])); }
        CK(cudaEventCreate(&t0));CK(cudaEventCreate(&t1));CK(cudaEventCreate(&ea));CK(cudaEventCreate(&eb));CK(cudaEventCreate(&ec));CK(cudaEventCreate(&ed));

        // S0 template (once): A,U,V + P=U*A — same as --gate
        seed(dSeedA,7); seed(dSeedU,50); seed(dSeedV,200);
        genop(dA,dSeedA,nn); genop(dU,dSeedU,mn); genop(dV,dSeedV,nm);
        int8_gemm(lt,s,ws,(size_t)32<<20,dU,dA,dP,m,n,n);
        CK(cudaStreamSynchronize(s));

        // CUB-vs-thrust B-byte cross-check (the scan swap must not change a byte)
        unsigned long long hs1=0,hs2=0;
        {
            seed(dSeedB,100);
            gen_stream(dB,dSeedB,nn,0x73);  fold(dB,nn,dAcc,s);  CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(&hs1,dAcc,8,cudaMemcpyDeviceToHost));
            gen_cub(dB2[0],dSeedB,nn,s);    fold(dB2[0],nn,dAcc,s); CK(cudaStreamSynchronize(s));
            CK(cudaMemcpy(&hs2,dAcc,8,cudaMemcpyDeviceToHost));
        }
        // SERIAL arm, stage-split + checksum pass (per-nonce sync to read events).
        // psplit=true = the reference truth; a second hoisted pass measures the
        // solver model (P-limb split template-hoisted out of the per-nonce path).
        float ts1=0,ts2=0,ts3=0,ts3full=0;
        auto stage_split=[&](bool psplit,bool doSum,float* o1,float* o2,float* o3){
            float a1=0,a2=0,a3=0;
            for(int it=0; it<W+Q; ++it){
                CK(cudaMemcpyAsync(dSeedB2,hSeeds+(size_t)it*32,32,cudaMemcpyHostToDevice,sMain));
                CK(cudaEventRecord(ea,sMain));
                gen_cub(dB2[0],dSeedB2,nn,sMain);
                CK(cudaEventRecord(eb,sMain));
                int8_gemm(lt,sMain,ws,(size_t)32<<20,dB2[0],dV,dQ,n,m,n);
                CK(cudaEventRecord(ec,sMain));
                bmx4_st(dP,dQ,dChat,sMain,psplit);
                CK(cudaEventRecord(ed,sMain));
                if(doSum && it>=W) fold(dChat,mm*8,dSumS+(it-W),sMain);
                CK(cudaStreamSynchronize(sMain));
                if(it>=W){ float a,b2,c2; CK(cudaEventElapsedTime(&a,ea,eb));CK(cudaEventElapsedTime(&b2,eb,ec));CK(cudaEventElapsedTime(&c2,ec,ed)); a1+=a;a2+=b2;a3+=c2; }
            }
            *o1=a1/Q; *o2=a2/Q; *o3=a3/Q;
        };
        stage_split(true,true,&ts1,&ts2,&ts3full);          // reference checksums hS + full S3g
        bmx4_split<<<((uint32_t)mn+T-1)/T,T,0,sMain>>>(dP,dPlimb,mn);   // template-hoist: prime dPlimb ONCE
        CK(cudaStreamSynchronize(sMain));
        { float d1,d2; stage_split(false,false,&d1,&d2,&ts3); ts1=(ts1+d1)/2; ts2=(ts2+d2)/2; }  // hoisted S3g
        // PIPELINED arm, checksum pass (same seeds; gen k+1 on gen stream behind S2+S3g of k)
        auto pipe_arm=[&](cudaStream_t sM,cudaStream_t sG,bool doSum,unsigned long long* sums,double* watts)->float{
            CK(cudaMemcpyAsync(dSeedB2,hSeeds,32,cudaMemcpyHostToDevice,sG));
            gen_cub(dB2[0],dSeedB2,nn,sG); CK(cudaEventRecord(evGen[0],sG));
            for(int it=0; it<W+Q; ++it){
                int p=it&1;
                if(it==W) CK(cudaEventRecord(t0,sM));
                CK(cudaStreamWaitEvent(sM,evGen[p],0));
                int8_gemm(lt,sM,ws,(size_t)32<<20,dB2[p],dV,dQ,n,m,n);   // S2 consumes B slot p
                CK(cudaEventRecord(evFree[p],sM));
                if(it+1<W+Q){                                             // S1b of nonce it+1, other slot
                    CK(cudaMemcpyAsync(dSeedB2+32*(p^1),hSeeds+(size_t)(it+1)*32,32,cudaMemcpyHostToDevice,sG));
                    CK(cudaStreamWaitEvent(sG,evFree[p^1],0));
                    gen_cub(dB2[p^1],dSeedB2+32*(p^1),nn,sG);
                    CK(cudaEventRecord(evGen[p^1],sG));
                }
                bmx4_st(dP,dQ,dChat,sM,false);                            // S3g (P-limb split hoisted)
                if(doSum && it>=W) fold(dChat,mm*8,sums+(it-W),sM);
            }
            CK(cudaEventRecord(t1,sM));
            double pw=0; long pwn=0;
            while(cudaStreamQuery(sM)==cudaErrorNotReady){
                if(nvml_ok){ unsigned int mw; if(nvmlDeviceGetPowerUsage(nvdev,&mw)==NVML_SUCCESS){ pw+=mw; pwn++; } } }
            CK(cudaStreamSynchronize(sM)); CK(cudaStreamSynchronize(sG));
            if(watts) *watts = pwn>0 ? pw/pwn/1000.0 : 0.0;
            float el; CK(cudaEventElapsedTime(&el,t0,t1)); return el/Q;
        };
        pipe_arm(sMain,sGen,true,dSumP,nullptr);
        std::vector<unsigned long long> hS(Q),hP(Q);
        CK(cudaMemcpy(hS.data(),dSumS,(size_t)Q*8,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hP.data(),dSumP,(size_t)Q*8,cudaMemcpyDeviceToHost));
        int mism=0; for(int i=0;i<Q;i++) if(hS[i]!=hP[i]) mism++;
        // second checksum pass: background-gen config (small grid + priority streams)
        genB=340;
        pipe_arm(sMainH,sGenL,true,dSumP,nullptr);
        CK(cudaMemcpy(hP.data(),dSumP,(size_t)Q*8,cudaMemcpyDeviceToHost));
        int mism2=0; for(int i=0;i<Q;i++) if(hS[i]!=hP[i]) mism2++;
        genB=0;
        // clean TIMING passes (no checksums): serial wall (no per-nonce sync) then pipelined
        auto serial_timed=[&](double* watts)->float{
            for(int it=0; it<W+Q; ++it){
                CK(cudaMemcpyAsync(dSeedB2,hSeeds+(size_t)it*32,32,cudaMemcpyHostToDevice,sMain));
                if(it==W) CK(cudaEventRecord(t0,sMain));
                gen_cub(dB2[0],dSeedB2,nn,sMain);
                int8_gemm(lt,sMain,ws,(size_t)32<<20,dB2[0],dV,dQ,n,m,n);
                bmx4_st(dP,dQ,dChat,sMain,false);
            }
            CK(cudaEventRecord(t1,sMain));
            double pw=0; long pwn=0;
            while(cudaStreamQuery(sMain)==cudaErrorNotReady){
                if(nvml_ok){ unsigned int mw; if(nvmlDeviceGetPowerUsage(nvdev,&mw)==NVML_SUCCESS){ pw+=mw; pwn++; } } }
            CK(cudaStreamSynchronize(sMain));
            if(watts) *watts = pwn>0 ? pw/pwn/1000.0 : 0.0;
            float el; CK(cudaEventElapsedTime(&el,t0,t1)); return el/Q;
        };
        double wS=0;
        float mser = serial_timed(&wS);
        const float ceilm = (ts1 > ts2+ts3) ? ts1 : ts2+ts3;   // perfect-hide floor from measured split
        printf("\n=== S1b/TENSOR OVERLAP PROTO | %s | n=%u m=%u window Q=%d ===\n",prop.name,n,m,Q);
        printf(" gen scan impl: CUB vs thrust B-checksum      : %s\n", hs1==hs2?"MATCH (byte-identical)":"** MISMATCH — CUB swap is WRONG, times void **");
        printf(" byte-exact: pipelined vs serial Chat checksum: %d / %d mism (full grid), %d / %d mism (bg 340blk+prio) %s\n",
               mism, Q, mism2, Q, (mism||mism2)?"** FAIL — TIMES VOID **":"(all agree)");
        if(hs1!=hs2 || mism || mism2) return 1;
        printf(" ---- per-nonce MARGINAL (S3g combine, P-limb split template-hoisted, CUB gen in BOTH arms) ----\n");
        printf(" serial    S1b + S2 + S3g            : %8.3f ms   %.0f nonce/s   %.0f W\n", mser, 1000.0/mser, wS);
        printf("   split: S1b %.3f | S2 %.3f | S3g %.3f  (S3g unhoisted %.3f -> P-split hoist saves %.3f ms/nonce)\n", ts1,ts2,ts3,ts3full,ts3full-ts3);
        printf(" ceiling   : %+.1f%% (perfect hide = max(S1b, S2+S3g) = %.3f ms)\n", 100.0*(mser/ceilm-1.0), ceilm);
        struct Cfg { const char* name; cudaStream_t sM,sG; int gb; };
        Cfg cfgs[]={ {"full-grid, no-prio ",sMain,sGen,0}, {"full-grid, prio    ",sMainH,sGenL,0},
                     {"bg 1024blk, prio   ",sMainH,sGenL,1024}, {"bg 512blk, prio    ",sMainH,sGenL,512},
                     {"bg 340blk, prio    ",sMainH,sGenL,340}, {"bg 340blk, no-prio ",sMain,sGen,340},
                     {"bg 170blk, prio    ",sMainH,sGenL,170} };
        for(auto& cf: cfgs){
            genB=cf.gb; double wp=0;
            float mp = pipe_arm(cf.sM,cf.sG,false,nullptr,&wp);
            double sp=100.0*(mser/mp-1.0), ce=100.0*(mser/ceilm-1.0);
            printf(" pipelined [%s] : %8.3f ms   %.0f nonce/s   %.0f W   %+5.1f%%  (%.0f%% of ceiling)\n",
                   cf.name, mp, 1000.0/mp, wp, sp, ce>1e-9?100.0*sp/ce:0.0);
            printf("CSVOVL,%s,%u,%u,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f\n",
                   prop.name,n,m,Q,cf.name,ts1,ts2,ts3,mser,mp,ceilm,wS,wp);
        }
        genB=0;
        if(nvml_ok) nvmlShutdown();
        return 0;
    }

    // ---- --pipe: the DEPLOY-path per-nonce pipeline. Three levers over --gate's
    //      sequential marginal, measured separately then together:
    //        (h) template-scoped P-limb split HOISTED out of the per-nonce path
    //        (o) expand-B of nonce k+1 (SHA/int ALU) OVERLAPPED with the tensor
    //            chain of nonce k (S2 GEMM -> split_qh -> fused GEMM -> reconstruct)
    //            on separate streams with double-buffered slots
    //        (d) S4 digest = chained SHA256d over the serialized sketch, ON DEVICE
    //            (the reference hosts this on CPU), co-run as parallel chains
    //      Byte-exact gates precede all timing: pipe Chat == sequential Chat, and
    //      the device digest == the host (golden-validated) sketch_digest. ----
    if(pipe){
        const int Q = nonces>4 ? nonces : 192;         // steady-state window
        int KD = kdovr>0 ? kdovr : 128;                // digest ring slots (chains in flight)
        { size_t fre=0,tot=0; CK(cudaMemGetInfo(&fre,&tot));
          while(KD>16 && (size_t)KD*mm*8+((size_t)2<<30)>fre) KD/=2; }

        // cub-scan variant of the wide XOF: preallocated temp storage, so the
        // expand stream never hits thrust's per-call cudaMalloc (which would
        // serialize it against the tensor stream and void the overlap).
        size_t cubBytes=0;
        cub::DeviceScan::ExclusiveSum(nullptr,cubBytes,dCnt,dOff,(int)maxblk);
        void* dCubTmp=nullptr; CK(cudaMalloc(&dCubTmp,cubBytes));
        auto gen_on=[&](cudaStream_t st,int8_t* out,const uint8_t* dSeed,uint32_t count){
            uint32_t nblk=count/31+64;
            stream_count<<<(nblk+T-1)/T,T,0,st>>>(dSeed,0x73,dScratch,dCnt,nblk);
            size_t tb=cubBytes;
            cub::DeviceScan::ExclusiveSum(dCubTmp,tb,dCnt,dOff,(int)nblk,st);
            stream_scatter<<<(nblk+T-1)/T,T,0,st>>>(dScratch,dOff,dCnt,out,nblk,count);
        };

        // template S0 (once): A,U,V, P=U*A, and the HOISTED P-limb split
        seed(dSeedA,7); seed(dSeedU,50); seed(dSeedV,200);
        genop(dA,dSeedA,nn); genop(dU,dSeedU,mn); genop(dV,dSeedV,nm);
        if(!int8_gemm(lt,s,ws,(size_t)32<<20,dU,dA,dP,m,n,n)){printf("cublasLt INT8 unavailable\n");return 1;}
        bmx4_split<<<((uint32_t)mn+T-1)/T,T,0,s>>>(dP,dPlimb,mn);   // template-scoped: paid once per job, not per nonce
        CK(cudaStreamSynchronize(s));

        // per-nonce seeds, device-resident up front (no per-iteration H2D)
        std::vector<uint8_t> hseeds((size_t)Q*32);
        for(int i2=0;i2<Q;i2++) for(int j=0;j<32;j++) hseeds[(size_t)i2*32+j]=(uint8_t)(100+i2+j);
        uint8_t* dSeeds; CK(cudaMalloc(&dSeeds,hseeds.size()));
        CK(cudaMemcpy(dSeeds,hseeds.data(),hseeds.size(),cudaMemcpyHostToDevice));

        // double-buffered slots (slot 0 reuses the singleton buffers)
        int8_t*   dB2[2]  ={dB,   nullptr};
        int32_t*  dQ2[2]  ={dQ,   nullptr};
        int8_t*   dQlh2[2]={dQlh, nullptr};
        int32_t*  dG2[2]  ={dG,   nullptr};
        uint64_t* dChat2[2]={dChat,nullptr};
        CK(cudaMalloc(&dB2[1],nn)); CK(cudaMalloc(&dQ2[1],nm*4)); CK(cudaMalloc(&dQlh2[1],(size_t)BMXL*nm));
        CK(cudaMalloc(&dG2[1],(size_t)BMXL*BMXL*mm*4)); CK(cudaMalloc(&dChat2[1],mm*8));

        auto tensor_chain=[&](cudaStream_t st,int sl){
            int8_gemm(lt,st,ws,(size_t)32<<20,dB2[sl],dV,dQ2[sl],n,m,n);          // S2: Q = B*V
            bmx4_split_qh<<<((uint32_t)nm+T-1)/T,T,0,st>>>(dQ2[sl],dQlh2[sl],n,m);
            int8_gemm(lt,st,ws,(size_t)32<<20,dPlimb,dQlh2[sl],dG2[sl],BMXL*m,BMXL*m,n); // fused 4-base-64 GEMM
            dim3 bl(16,16), gr((m+15)/16,(m+15)/16);
            bmx4_reconstruct<<<gr,bl,0,st>>>(dG2[sl],dChat2[sl],m);
        };

        // ---- phase A: SEQUENTIAL baseline (same hoisted work, one stream) ----
        auto seq_run=[&](int q)->float{
            cudaEvent_t a,b2; CK(cudaEventCreate(&a));CK(cudaEventCreate(&b2));
            CK(cudaEventRecord(a,s));
            for(int it=0;it<q;++it){
                gen_on(s,dB2[it&1],dSeeds+(size_t)(it%Q)*32,(uint32_t)nn);
                tensor_chain(s,it&1);
            }
            CK(cudaEventRecord(b2,s)); CK(cudaStreamSynchronize(s));
            float ms=0; CK(cudaEventElapsedTime(&ms,a,b2));
            cudaEventDestroy(a);cudaEventDestroy(b2);
            return ms/q;
        };
        seq_run(6);                                    // warmup (heuristics, clocks)
        const float seq_ms=seq_run(Q);
        std::vector<uint64_t> hSeq(mm);
        CK(cudaMemcpy(hSeq.data(),dChat2[(Q-1)&1],mm*8,cudaMemcpyDeviceToHost));

        // ---- phase B: PIPELINED (expand k+1 on sE || tensor chain k on sT2) ----
        cudaStream_t sE,sT2,sD; CK(cudaStreamCreate(&sE));CK(cudaStreamCreate(&sT2));CK(cudaStreamCreate(&sD));
        cudaEvent_t evE[2],evT[2];
        for(int i2=0;i2<2;i2++){CK(cudaEventCreateWithFlags(&evE[i2],cudaEventDisableTiming));CK(cudaEventCreateWithFlags(&evT[i2],cudaEventDisableTiming));}
        auto enqueue_pipe=[&](int q){
            gen_on(sE,dB2[0],dSeeds+0,(uint32_t)nn); CK(cudaEventRecord(evE[0],sE));
            for(int it=0;it<q;++it){
                int sl=it&1, ns2=(it+1)&1;
                CK(cudaStreamWaitEvent(sT2,evE[sl],0));            // B[sl] ready
                tensor_chain(sT2,sl);
                CK(cudaEventRecord(evT[sl],sT2));
                if(it+1<q){
                    CK(cudaStreamWaitEvent(sE,evT[ns2],0));        // B[ns2] free (tensor k-1 done reading)
                    gen_on(sE,dB2[ns2],dSeeds+(size_t)((it+1)%Q)*32,(uint32_t)nn);
                    CK(cudaEventRecord(evE[ns2],sE));
                }
            }
        };
        auto pipe_run=[&](int q)->float{
            CK(cudaDeviceSynchronize());
            cudaEvent_t a,b2; CK(cudaEventCreate(&a));CK(cudaEventCreate(&b2));
            CK(cudaEventRecord(a,sT2));
            enqueue_pipe(q);
            CK(cudaEventRecord(b2,sT2)); CK(cudaDeviceSynchronize());
            float ms=0; CK(cudaEventElapsedTime(&ms,a,b2));
            cudaEventDestroy(a);cudaEventDestroy(b2);
            return ms/q;
        };
        pipe_run(6);                                   // warmup
        const float pipe_ms=pipe_run(Q);
        std::vector<uint64_t> hPipe(mm);
        CK(cudaMemcpy(hPipe.data(),dChat2[(Q-1)&1],mm*8,cudaMemcpyDeviceToHost));
        size_t mism=0; for(size_t i2=0;i2<mm;i2++) if(hPipe[i2]!=hSeq[i2]) mism++;

        // ---- S4 digest: device chained SHA256d over the serialized sketch ----
        static const char ptag[]="BTX_MATMUL_V4";
        uint8_t sig2[32]; for(int i2=0;i2<32;i2++) sig2[i2]=(uint8_t)(32+i2);
        uint8_t hhdr[45]; memcpy(hhdr,ptag,13); memcpy(hhdr+13,sig2,32);
        uint8_t* dHdr; CK(cudaMalloc(&dHdr,45)); CK(cudaMemcpy(dHdr,hhdr,45,cudaMemcpyHostToDevice));
        uint8_t* dRing; CK(cudaMalloc(&dRing,(size_t)KD*mm*8));    // KD distinct 32/8 MiB sketches (DRAM-honest)
        for(int i2=0;i2<KD;i2++) CK(cudaMemcpyAsync(dRing+(size_t)i2*mm*8,dChat2[(Q-1)&1],mm*8,cudaMemcpyDeviceToDevice,s));
        uint8_t* dDig; CK(cudaMalloc(&dDig,(size_t)std::max(Q,KD)*32));
        CK(cudaStreamSynchronize(s));
        // byte-exact: device chain vs host (golden-validated) sketch_digest
        digest_chain<<<1,32,0,s>>>(dHdr,45,dRing,(size_t)mm*8,KD,(size_t)mm*8,1,dDig);
        CK(cudaStreamSynchronize(s));
        uint8_t ddig[32]; CK(cudaMemcpy(ddig,dDig,32,cudaMemcpyDeviceToHost));
        std::vector<uint8_t> hpay(mm*8);
        for(size_t i2=0;i2<mm;i2++){uint64_t v=hPipe[i2];for(int b3=0;b3<8;b3++)hpay[i2*8+b3]=(uint8_t)(v>>(8*b3));}
        uint8_t hdig[32]; sketch_digest(sig2,hpay.data(),hpay.size(),hdig);
        const bool dig_ok = memcmp(ddig,hdig,32)==0;
        printf("\n=== BMX4-C DEPLOY PIPELINE | %s | n=%u m=%u b=%u Q=%d KD=%d ===\n",prop.name,n,m,b,Q,KD);
        printf(" byte-exact: pipe Chat vs seq = %zu mism | device S4 digest vs host = %s %s\n",
               mism, dig_ok?"MATCH":"MISMATCH",(mism||!dig_ok)?"** FAIL - TIMES VOID **":"(PASS)");
        if(mism||!dig_ok) return 1;

        // standalone S4 throughput/latency: KD chains at once
        { digest_chain<<<(KD+31)/32,32,0,sD>>>(dHdr,45,dRing,(size_t)mm*8,KD,(size_t)mm*8,KD,dDig); CK(cudaStreamSynchronize(sD)); }
        cudaEvent_t da,db2; CK(cudaEventCreate(&da));CK(cudaEventCreate(&db2));
        CK(cudaEventRecord(da,sD));
        digest_chain<<<(KD+31)/32,32,0,sD>>>(dHdr,45,dRing,(size_t)mm*8,KD,(size_t)mm*8,KD,dDig);
        CK(cudaEventRecord(db2,sD)); CK(cudaStreamSynchronize(sD));
        float dig_ms=0; CK(cudaEventElapsedTime(&dig_ms,da,db2));
        const float s4_amort=dig_ms/KD;

        // ---- phase D: SUSTAINED co-run: pipe Q nonces + digest Q chains together ----
        CK(cudaDeviceSynchronize());
        auto t0=std::chrono::steady_clock::now();
        digest_chain<<<(Q+31)/32,32,0,sD>>>(dHdr,45,dRing,(size_t)mm*8,KD,(size_t)mm*8,Q,dDig);
        enqueue_pipe(Q);
        double pwsum=0; long pwn2=0;
        while(cudaStreamQuery(sT2)==cudaErrorNotReady || cudaStreamQuery(sD)==cudaErrorNotReady){
            if(nvml_ok){unsigned int mw; if(nvmlDeviceGetPowerUsage(nvdev,&mw)==NVML_SUCCESS){pwsum+=mw;pwn2++;}}
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CK(cudaDeviceSynchronize());
        const double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        const double sus_ms=1000.0*wall/Q, avg_w=(pwn2>0)?(pwsum/pwn2/1000.0):0.0;
        const double need_chains = (1000.0/sus_ms)*(dig_ms/1000.0);   // in-flight chains at sustained rate
        printf(" A sequential (P-split hoisted)        : %8.3f ms/nonce   %6.0f nonce/s\n", seq_ms, 1000.0/seq_ms);
        printf(" B pipelined  (expand || tensor)       : %8.3f ms/nonce   %6.0f nonce/s   [%+.1f%% vs A]\n",
               pipe_ms, 1000.0/pipe_ms, 100.0*(seq_ms/pipe_ms-1.0));
        printf(" C S4 digest, %3d chains standalone    : %8.3f ms latency -> %6.3f ms/nonce amortized (%.1f MiB/sketch)\n",
               KD, dig_ms, s4_amort, mm*8.0/1048576.0);
        printf("    in-flight at B-rate: ~%.0f chains = %.1f GiB sketch ring (ref hosts S4 on CPU: ~%.0f ms/nonce @2GB/s SHA-NI core)\n",
               need_chains, need_chains*mm*8.0/1073741824.0, (mm*8.0/64.0)/31250.0);
        printf(" D SUSTAINED co-run (pipe + S4)        : %8.3f ms/nonce   %6.0f nonce/s   %.0f W   [%+.1f%% vs A]\n",
               sus_ms, 1000.0/sus_ms, avg_w, 100.0*(seq_ms/sus_ms-1.0));
        printf("CSVPIPE,%s,%u,%u,%u,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f\n",
               prop.name,n,m,b,Q,KD,seq_ms,pipe_ms,s4_amort,dig_ms,sus_ms,avg_w);
        if(nvml_ok) nvmlShutdown();
        return 0;
    }

    // ---- --solve: the FULL miner compute path, header -> digest, all on device.
    //      Host derives seeds from a CBlockHeader with the validated §H.4/§A.2 rule
    //      (ComputeTemplateHash + DeriveTaggedSeed), the GPU M11/E8M0 sampler makes
    //      the operands, and the reused GEMM->4-base-64 combine->device-S4 digest
    //      chain produces H(sigma||Chat). Checked against ALL THREE reference
    //      goldens (n=256 n1, n=128 n1, n=256 n2). This is the exact chain that
    //      drops into SolveMatMulBmx4C in the miner. ----
    if(solve){
        // ---- host header hashing + seed derivation (byte-exact to matmul_v4_bmx4.cpp) ----
        auto wle=[&](uint8_t*o,uint64_t v,int n2){for(int i=0;i<n2;i++)o[i]=(uint8_t)(v>>(8*i));};
        // ComputeMatMulHeaderHash: SHA256(ver||prev||merkle||time||bits||nonce64||dim||seed_a||seed_b)
        auto hdrhash=[&](uint32_t ver,const uint8_t*pb,const uint8_t*mr,uint32_t tm,uint32_t bits,
                         uint64_t n64,uint16_t dim,const uint8_t*sa,const uint8_t*sb,uint8_t out[32]){
            std::vector<uint8_t> buf; uint8_t t4[8];
            wle(t4,ver,4); buf.insert(buf.end(),t4,t4+4);
            buf.insert(buf.end(),pb,pb+32); buf.insert(buf.end(),mr,mr+32);
            wle(t4,tm,4); buf.insert(buf.end(),t4,t4+4);
            wle(t4,bits,4); buf.insert(buf.end(),t4,t4+4);
            wle(t4,n64,8); buf.insert(buf.end(),t4,t4+8);
            wle(t4,dim,2); buf.insert(buf.end(),t4,t4+2);
            buf.insert(buf.end(),sa,sa+32); buf.insert(buf.end(),sb,sb+32);
            hsha256(buf.data(),buf.size(),out);
        };
        auto tagged=[&](const char*tag,size_t tl,const uint8_t*hash,const uint8_t*w,size_t wl,uint8_t out[32]){
            std::vector<uint8_t> buf; buf.insert(buf.end(),tag,tag+tl);
            buf.insert(buf.end(),hash,hash+32); if(wl)buf.insert(buf.end(),w,w+wl);
            hsha256(buf.data(),buf.size(),out);
        };
        struct Golden{ uint32_t n; uint64_t nonce; const char* dig; };
        const Golden golds[]={
            {256,0x0000000000000001ULL,"4e192d8b907ad2d1383600d6f9b794c3ebf6387d577ca82333e75f544f54a9f9"},
            {256,0x0000000000000002ULL,"91fe8b670ad84b6b37d6ce859133945f7d8181709f7dbdf8a64b8c7e25f4aeed"},
            {128,0x0000000000000001ULL,"c94923800c8a5e344c88efdb2ec5ad07d80694c903af3dae1859ec14ade67b7c"},
        };
        // the reference test header (MakeV4Header)
        auto parse=[&](std::string h,uint8_t*o){ while(h.size()<64)h+="a3"; for(int i=0;i<32;i++)o[i]=(uint8_t)std::stoi(h.substr(i*2,2),nullptr,16); };
        uint8_t PB[32],MR[32],SEEDA[32],SEEDB[32];
        parse("5151515151515151515151515151515151515151515151515151515151515151",PB);
        parse("a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3",MR);
        parse("1111111111111111111111111111111111111111111111111111111111111111",SEEDA);
        parse("2222222222222222222222222222222222222222222222222222222222222222",SEEDB);
        const uint32_t VER=0x20000004,TM=1770000000,BITS=0x207fffff;

        printf("=== ENC-BMX4C --solve (header->digest, GPU M11/E8M0 sampler) | %s ===\n", prop.name);
        int passes=0, total=0;
        for(const auto& g: golds){
            const uint32_t nn2=g.n, m2=nn2/4; (void)m2;
            uint8_t th[32], fh[32], sig1[32], sig[32];
            { uint8_t znull[32]={0}; hdrhash(VER,PB,MR,TM,BITS,0,(uint16_t)nn2,znull,znull,th); } // template: nonce=0, seeds null
            hdrhash(VER,PB,MR,TM,BITS,g.nonce,(uint16_t)nn2,SEEDA,SEEDB,fh);   // full header hash (B)
            hdrhash(VER,PB,MR,TM,BITS,g.nonce,(uint16_t)nn2,SEEDA,SEEDB,sig1); // sigma = SHA256d(header)
            hsha256(sig1,32,sig);
            uint8_t SA[32],SB[32],SU[32],SV[32]; uint8_t A=0x41,B=0x42;
            tagged("BTX_MATMUL_SEED_V42",19,th,&A,1,SA);
            tagged("BTX_MATMUL_SEED_V42",19,fh,&B,1,SB);
            tagged("BTX_MATMUL_V42_SKETCH_U",23,th,nullptr,0,SU);
            tagged("BTX_MATMUL_V42_SKETCH_V",23,th,nullptr,0,SV);

            // seeds live on device as INTERNAL byte order (SeedBytesLE reverses in-kernel)
            CK(cudaMemcpy(dSeedA,SA,32,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dSeedB,SB,32,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dSeedU,SU,32,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dSeedV,SV,32,cudaMemcpyHostToDevice));

            // GPU M11/E8M0 sampler (byte-exact to host bmxA/bmxB/bmxProj)
            const size_t cA=(size_t)nn2*nn2, cU=(size_t)m2*nn2, cV=(size_t)nn2*m2;
            const uint32_t nblkA=(uint32_t)(cA/44+64);   // ~11/16 accept -> generous block estimate
            int8_t *dMuA,*dMuB; uint8_t *dScA,*dScB;
            CK(cudaMalloc(&dMuA,cA)); CK(cudaMalloc(&dMuB,cA));
            CK(cudaMalloc(&dScA,(size_t)nn2*(nn2/32))); CK(cudaMalloc(&dScB,(size_t)(nn2/32)*nn2));
            int8_t* dScratchM; uint32_t *dCntM,*dOffM;
            CK(cudaMalloc(&dScratchM,(size_t)nblkA*64)); CK(cudaMalloc(&dCntM,nblkA*4)); CK(cudaMalloc(&dOffM,nblkA*4));
            size_t mCubBytes=0; cub::DeviceScan::ExclusiveSum(nullptr,mCubBytes,dCntM,dOffM,(int)nblkA);
            void* dMCub=nullptr; CK(cudaMalloc(&dMCub,mCubBytes));
            auto mant=[&](int8_t* out,uint8_t* dSeed,size_t cnt,const char* lbl){
                uint32_t nb=(uint32_t)(cnt/44+64);
                bmx_mant_count<<<(nb+T-1)/T,T,0,s>>>(dSeed,dScratchM,dCntM,nb);
                size_t tb=mCubBytes;
                cub::DeviceScan::ExclusiveSum(dMCub,tb,dCntM,dOffM,(int)nb,s);
                bmx_mant_scatter<<<(nb+T-1)/T,T,0,s>>>(dScratchM,dOffM,dCntM,out,nb,cnt);
                (void)lbl;
            };
            // A: mantissa cA + scale (per-col-32) -> dA
            mant(dMuA,dSeedA,cA,"A");
            bmx_scale<<<((uint32_t)((size_t)nn2*(nn2/32))+T-1)/T,T,0,s>>>(dSeedA,dScA,(size_t)nn2*(nn2/32));
            bmx_dq_A<<<((uint32_t)cA+T-1)/T,T,0,s>>>(dMuA,dScA,dA,nn2);
            // B: mantissa cA + scale (per-row-32) -> dB
            mant(dMuB,dSeedB,cA,"B");
            bmx_scale<<<((uint32_t)((size_t)(nn2/32)*nn2)+T-1)/T,T,0,s>>>(dSeedB,dScB,(size_t)(nn2/32)*nn2);
            bmx_dq_B<<<((uint32_t)cA+T-1)/T,T,0,s>>>(dMuB,dScB,dB,nn2);
            // U,V: scale-free M11 mantissa -> dU,dV
            mant(dU,dSeedU,cU,"U");
            mant(dV,dSeedV,cV,"V");
            CK(cudaStreamSynchronize(s));

            // TASK-2 GATE: GPU sampler operands == host reference bmxA/bmxB/bmxProj,
            // byte-for-byte, from the SAME derived seeds. This validates the sampler
            // independently of the GEMM->digest chain (already golden via --bmx4c).
            {
                // host bmx* does NOT reverse internally (the --bmx4c bench feeds it
                // already-reversed hex); the GPU kernel + the reference SeedBytesLE
                // DO reverse. So hand the host the reversed seed to match both.
                auto rev=[&](const uint8_t* p){ std::vector<uint8_t> v(32); for(int i=0;i<32;i++)v[i]=p[31-i]; return v; };
                std::vector<uint8_t> sa=rev(SA),sb=rev(SB),su=rev(SU),sv=rev(SV);
                std::vector<int8_t> hA=bmxA(sa,nn2), hB=bmxB(sb,nn2), hU=bmxProj(su,m2,nn2), hV=bmxProj(sv,nn2,m2);
                std::vector<int8_t> gA(cA),gB(cA),gU(cU),gV(cV);
                CK(cudaMemcpy(gA.data(),dA,cA,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(gB.data(),dB,cA,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(gU.data(),dU,cU,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(gV.data(),dV,cV,cudaMemcpyDeviceToHost));
                auto diff=[&](const std::vector<int8_t>&g,const std::vector<int8_t>&h){size_t d=0;for(size_t i=0;i<h.size();i++)if(g[i]!=h[i])d++;return d;};
                size_t dA_=diff(gA,hA),dB_=diff(gB,hB),dU_=diff(gU,hU),dV_=diff(gV,hV);
                printf("  n=%-4u nonce=%llu  sampler mism: A=%zu B=%zu U=%zu V=%zu / %zu %s\n",
                       nn2,(unsigned long long)g.nonce,dA_,dB_,dU_,dV_,(size_t)cA,
                       (dA_||dB_||dU_||dV_)?"** SAMPLER MISMATCH **":"(GPU sampler byte-exact to host reference)");
            }
            cudaFree(dMuA);cudaFree(dMuB);cudaFree(dScA);cudaFree(dScB);
            cudaFree(dScratchM);cudaFree(dCntM);cudaFree(dOffM);cudaFree(dMCub);

            // P=U*A (m x n), Q=B*V (n x m), 4-base-64 combine, device S4 digest
            bool g1=int8_gemm(lt,s,ws,(size_t)32<<20,dU,dA,dP,m2,nn2,nn2);
            bool g2=int8_gemm(lt,s,ws,(size_t)32<<20,dB,dV,dQ,nn2,m2,nn2);
            if(!g1||!g2){printf("  n=%u: cublasLt INT8 unavailable (P/Q GEMM)\n",nn2);return 1;}
            bmx4_split<<<((uint32_t)((size_t)m2*nn2)+T-1)/T,T,0,s>>>(dP,dPlimb,(size_t)m2*nn2);
            bmx4_split_qh<<<((uint32_t)((size_t)nn2*m2)+T-1)/T,T,0,s>>>(dQ,dQlh,nn2,m2);
            if(!int8_gemm(lt,s,ws,(size_t)32<<20,dPlimb,dQlh,dG,BMXL*m2,BMXL*m2,nn2)){
                printf("  n=%u: fused combine GEMM fail (M=N=%u K=%u)\n",nn2,BMXL*m2,nn2);return 1;}
            { dim3 bl(16,16),gr((m2+15)/16,(m2+15)/16); bmx4_reconstruct<<<gr,bl,0,s>>>(dG,dChat,m2); }
            CK(cudaStreamSynchronize(s));

            // device S4 digest chain: header = tag||sigma (sigma internal LE)
            static const char tagd[]="BTX_MATMUL_V4"; uint8_t hh[45]; memcpy(hh,tagd,13);
            memcpy(hh+13,sig,32);   // ComputeSketchDigest hashes sigma.data() (SHA256d output order) directly
            uint8_t *dH2,*dO2; CK(cudaMalloc(&dH2,45)); CK(cudaMalloc(&dO2,32));
            CK(cudaMemcpy(dH2,hh,45,cudaMemcpyHostToDevice));
            const size_t mm2=(size_t)m2*m2;
            digest_chain<<<1,32,0,s>>>(dH2,45,(const uint8_t*)dChat,0,1,mm2*8,1,dO2);
            CK(cudaStreamSynchronize(s));
            uint8_t dd[32]; CK(cudaMemcpy(dd,dO2,32,cudaMemcpyDeviceToHost));
            char got[65]; for(int i=0;i<32;i++) snprintf(got+i*2,3,"%02x",dd[31-i]);   // display order

            total++;
            std::string want(g.dig);
            bool ok = want==got;
            if(ok) passes++;
            printf("  n=%-4u nonce=%llu  digest=%s  %s\n", nn2, (unsigned long long)g.nonce, got,
                   ok?"MATCH":"** MISMATCH **");
            cudaFree(dH2);cudaFree(dO2);
        }
        printf("=> %d/%d goldens matched (header->digest, GPU sampler, ready for SolveMatMulBmx4C)\n",passes,total);
        return passes==total?0:1;
    }

    // ---- --bmx4c: reproduce the reference ENC-BMX4C golden digest (n=256 nonce=1
    //      = 4e192d8b...). Uses the emit's DERIVED seeds; validates the new operand
    //      generation (M11 + E8M0) end-to-end through the reused GEMM/combine/digest. ----
    if(bmx4c){
        printf("=== ENC-BMX4C replica | %s | n=%u m=%u (golden n=256 nonce=1) ===\n", prop.name, n, m);
        // derived seeds for MakeV4Header(nonce=1, n=256), from the reference refcheck-emit:
        const auto SA=hex2b("907e62a8f1c9f7b5d13ef3b82414ef6a307663d8e7aa590460575de0fda1443d");
        const auto SB=hex2b("473af78e22929dc60dde29a0412897e7fbf6eeb81318f7c80ee5b6129d552456");
        const auto SU=hex2b("7eeb5a36ff36b56b6e896ccff560eb70cb4f5addfb6af238a76692450513eb4f");
        const auto SV=hex2b("85d52e063a9a1a49a840075c60bc778325fe521876e9eae3a9897509be4780cf");
        const auto SIG=hex2b("8308b2fa6265b55fcea5b7b2fce143f3b56e91d10ccad0cb71d7d03dea9e316a");
        std::vector<int8_t> Ahat=bmxA(SA,n), Bhat=bmxB(SB,n), Uu=bmxProj(SU,m,n), Vv=bmxProj(SV,n,m);
        auto row8=[&](const char* lbl,const std::vector<int8_t>& v,const char* gold){ printf("%-10s",lbl); for(int i=0;i<8;i++)printf(" %d",(int)v[i]); printf("   golden %s\n",gold); };
        row8("Ahat[0..7]",Ahat,"-6 -6 3 0 3 4 3 6");
        row8("Bhat[0..7]",Bhat,"-4 16 0 -3 48 -24 16 0");
        row8("U[0..7]",Uu,"1 4 -2 2 -3 -1 0 4");
        row8("V[0..7]",Vv,"-2 2 -1 -2 -4 0 0 0");
        CK(cudaMemcpy(dA,Ahat.data(),nn,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB,Bhat.data(),nn,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dU,Uu.data(),mn,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dV,Vv.data(),nm,cudaMemcpyHostToDevice));
        bool g1=int8_gemm(lt,s,ws,(size_t)32<<20,dU,dA,dP,m,n,n);   // P = U * Ahat  (m x n)
        bool g2=int8_gemm(lt,s,ws,(size_t)32<<20,dB,dV,dQ,n,m,n);   // Q = Bhat * V  (n x m)
        if(!g1||!g2){printf("cublasLt INT8 unavailable\n");return 1;}
        launch_combine(dP,dQ,dChat);                                // = ComputeCombineModQ
        CK(cudaStreamSynchronize(s));
        std::vector<int32_t> hP(mn),hQ(nm); CK(cudaMemcpy(hP.data(),dP,mn*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hQ.data(),dQ,nm*4,cudaMemcpyDeviceToHost));
        printf("P[0..3]    %d %d %d %d   golden 643 -727 567 887\n",hP[0],hP[1],hP[2],hP[3]);
        printf("Q[0..3]    %d %d %d %d   golden 78 -472 512 -1528\n",hQ[0],hQ[1],hQ[2],hQ[3]);
        std::vector<uint64_t> chat(mm); CK(cudaMemcpy(chat.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
        printf("Chat[0..3] %llu %llu %llu %llu\n           golden 11697362 2305843009209742944 2323758 11768734\n",
               (unsigned long long)chat[0],(unsigned long long)chat[1],(unsigned long long)chat[2],(unsigned long long)chat[3]);
        std::vector<uint8_t> payload(mm*8);
        for(size_t i=0;i<mm;i++){uint64_t v=chat[i];for(int b=0;b<8;b++)payload[i*8+b]=(uint8_t)(v>>(8*b));}
        std::vector<uint8_t> sigrev(SIG.rbegin(),SIG.rend());   // ComputeSketchDigest uses sigma.data() = internal LE
        uint8_t dig[32]; sketch_digest(sigrev.data(),payload.data(),payload.size(),dig);
        // uint256::GetHex() displays internal bytes reversed; print dig reversed to match.
        char got[65]; for(int i=0;i<32;i++) snprintf(got+i*2,3,"%02x",dig[31-i]);
        static const char* GOLD="4e192d8b907ad2d1383600d6f9b794c3ebf6387d577ca82333e75f544f54a9f9";
        printf("DIGEST     %s\n           golden %s\n=> %s\n", got, GOLD,
               std::string(got)==GOLD?"MATCH -- BMX4-C digest is BYTE-EXACT to the reference golden":"MISMATCH (see stage rows above)");
        // Cross-check: the BMX4-C NATIVE 4-base-64 combine must reach the SAME golden.
        bmx4_combine(dP,dQ,dChat); CK(cudaStreamSynchronize(s));
        std::vector<uint64_t> chat2(mm); CK(cudaMemcpy(chat2.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
        std::vector<uint8_t> payload2(mm*8);
        for(size_t i=0;i<mm;i++){uint64_t v=chat2[i];for(int b=0;b<8;b++)payload2[i*8+b]=(uint8_t)(v>>(8*b));}
        uint8_t dig2[32]; sketch_digest(sigrev.data(),payload2.data(),payload2.size(),dig2);
        char got2[65]; for(int i=0;i<32;i++) snprintf(got2+i*2,3,"%02x",dig2[31-i]);
        printf("DIGEST(4b64) %s\n=> %s\n", got2,
               std::string(got2)==GOLD?"MATCH -- 4-base-64 native combine is BYTE-EXACT (deploy path)":"MISMATCH -- 4-base-64 port is WRONG");
        // The device chained-SHA S4 digest (--pipe's solver stage) must ALSO hit
        // the reference golden, end-to-end from the device Chat bytes.
        {
            static const char tagd[]="BTX_MATMUL_V4";
            uint8_t hh[45]; memcpy(hh,tagd,13); memcpy(hh+13,sigrev.data(),32);
            uint8_t *dH2,*dO2; CK(cudaMalloc(&dH2,45)); CK(cudaMalloc(&dO2,32));
            CK(cudaMemcpy(dH2,hh,45,cudaMemcpyHostToDevice));
            digest_chain<<<1,32,0,s>>>(dH2,45,(const uint8_t*)dChat,0,1,mm*8,1,dO2);
            CK(cudaStreamSynchronize(s));
            uint8_t dd[32]; CK(cudaMemcpy(dd,dO2,32,cudaMemcpyDeviceToHost));
            char g3[65]; for(int i=0;i<32;i++) snprintf(g3+i*2,3,"%02x",dd[31-i]);
            printf("DIGEST(dev-S4) %s\n=> %s\n", g3, std::string(g3)==GOLD?
                "MATCH -- device chained-SHA S4 digest is BYTE-EXACT to the reference golden":"MISMATCH -- device S4 digest WRONG");
        }
        return 0;
    }

    // ---- --emit: reproduce the reference digest for the FIXED seeds shared with
    //      `refcheck emit <n>`, for a byte-exact cross-check. ----
    if(emit){
        seed(dSeedA,0); seed(dSeedB,64); seed(dSeedU,128); seed(dSeedV,192);
        // bit-exact wide stream (matches f50f0f8); emit digest should equal the reference
        gen_stream(dA,dSeedA,nn,0x73); gen_stream(dB,dSeedB,nn,0x73);
        gen_stream(dU,dSeedU,mn,0x73); gen_stream(dV,dSeedV,nm,0x73);
        bool g1=int8_gemm(lt,s,ws,(size_t)32<<20,dU,dA,dP,m,n,n);
        bool g2=int8_gemm(lt,s,ws,(size_t)32<<20,dB,dV,dQ,n,m,n);
        if(!g1||!g2){printf("cublasLt INT8 unavailable\n");return 1;}
        combine(dP,dQ,dChat);
        CK(cudaStreamSynchronize(s));
        std::vector<uint64_t> chat(mm); CK(cudaMemcpy(chat.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
        int8_t a8[8]; CK(cudaMemcpy(a8,dA,8,cudaMemcpyDeviceToHost));
        std::vector<uint8_t> payload(mm*8);
        for(size_t i=0;i<mm;i++){uint64_t v=chat[i];for(int b=0;b<8;b++)payload[i*8+b]=(uint8_t)(v>>(8*b));}
        uint8_t sigma[32]; for(int i=0;i<32;i++) sigma[i]=(uint8_t)(32+i);
        uint8_t dig[32]; sketch_digest(sigma,payload.data(),payload.size(),dig);
        printf("GPU n=%u m=%u\n",n,m);
        printf("A[0..7] ="); for(int i=0;i<8;i++) printf(" %d",(int)a8[i]); printf("\n");
        printf("CHAT[0] =%llu\n",(unsigned long long)chat[0]);
        printf("DIGEST  ="); for(int i=0;i<32;i++) printf("%02x",dig[i]); printf("\n");
        return 0;
    }

    for(int it=0; it<nonces; ++it){
        // vary seeds per nonce (stand-in for nNonce64 changing the header)
        seed(dSeedA,1+it); seed(dSeedB,100+it); seed(dSeedU,50+it); seed(dSeedV,200+it);

        CK(cudaEventRecord(e0,s));
        genop(dA,dSeedA,nn); genop(dB,dSeedB,nn); genop(dU,dSeedU,mn); genop(dV,dSeedV,nm);
        CK(cudaEventRecord(e1,s));
        // P=U*A (M=m,K=n,N=n) ; Q=B*V (M=n,K=n,N=m)
        bool g1=int8_gemm(lt,s,ws,(size_t)32<<20,dU,dA,dP,m,n,n);
        bool g2=int8_gemm(lt,s,ws,(size_t)32<<20,dB,dV,dQ,n,m,n);
        if(!g1||!g2){printf("cublasLt INT8 unavailable; aborting perf\n");return 1;}
        CK(cudaEventRecord(e2,s));
        combine(dP,dQ,dChat);
        CK(cudaEventRecord(e3,s));
        CK(cudaStreamSynchronize(s));
        float a,bb,c; CK(cudaEventElapsedTime(&a,e0,e1));CK(cudaEventElapsedTime(&bb,e1,e2));CK(cudaEventElapsedTime(&c,e2,e3));
        if(it>0){ t_gen+=a;t_gemm+=bb;t_comb+=c; // drop first (warmup)
            if(nvml_ok){ unsigned int mw; if(nvmlDeviceGetPowerUsage(nvdev,&mw)==NVML_SUCCESS){ pw_sum+=mw; pw_n++; } } }

        if(verify && it==0){
            gemm_scalar<<<(mn+T-1)/T,T,0,s>>>(dU,dA,dPs,m,n,n);
            gemm_scalar<<<(nm+T-1)/T,T,0,s>>>(dB,dV,dQs,n,m,n);
            launch_combine(dPs,dQs,dChatS);
            CK(cudaStreamSynchronize(s));
            std::vector<uint64_t> h1(mm),h2(mm);
            CK(cudaMemcpy(h1.data(),dChat,mm*8,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(h2.data(),dChatS,mm*8,cudaMemcpyDeviceToHost));
            size_t mism=0; for(size_t i=0;i<mm;i++) if(h1[i]!=h2[i]) mism++;
            printf("verify: cuBLASLt-vs-scalar Chat mismatches = %zu / %zu %s\n",mism,mm,mism?"FAIL":"OK");
        }
    }
    int N=nonces-1; if(N<1)N=1;
    t_gen/=N;t_gemm/=N;t_comb/=N; float tot=t_gen+t_gemm+t_comb;
    // wide stream yields ~31.4 accepted elements per SHA-256 block (+slack blocks)
    uint64_t sha = wide ? ((2ull*nn+2ull*mn)/31 + 256) : (2ull*nn+2ull*mn);
    uint64_t macs=2ull*nn*m;
    double nps=1000.0/tot;
    double avg_w=(nvml_ok && pw_n>0)?(pw_sum/pw_n/1000.0):0.0;
    double joules=avg_w*(tot/1000.0); // board joules per nonce
    printf("\n stage            ms/nonce   %%\n");
    printf(" 1 operand-gen(SHA) %7.3f  %5.1f%%   %.2f M SHA-256/nonce (%s)\n",t_gen,100*t_gen/tot,sha/1e6, wide?"wide XOF":"per-element");
    printf(" 2 INT8 GEMMs       %7.3f  %5.1f%%   %.2e MAC\n",t_gemm,100*t_gemm/tot,(double)macs);
    printf(" 3 Fq combine       %7.3f  %5.1f%%\n",t_comb,100*t_comb/tot);
    printf(" ------------------------------------------------\n");
    printf(" TOTAL              %7.3f            %.0f nonce/s\n",tot,nps);
    printf(" INT8 matmul share = %.1f%%   |   power = %.0f W   |   J/nonce = %.3f\n", 100*t_gemm/tot, avg_w, joules);
    // machine-parseable row for cross-GPU aggregation:
    // CSV,gpu,n,xof,combine,sha_ms,gemm_ms,comb_ms,nonce_s,watts,joules_per_nonce,int8_pct
    printf("CSV,%s,%u,%s,%s,%.4f,%.4f,%.4f,%.1f,%.1f,%.4f,%.2f\n",
           prop.name,n,wide?"wide":"legacy",c13?"c13":"tiled",t_gen,t_gemm,t_comb,nps,avg_w,joules,100.0*t_gemm/tot);
    if(nvml_ok) nvmlShutdown();
    return 0;
}
