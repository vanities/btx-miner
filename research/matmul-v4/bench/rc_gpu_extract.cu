// rc_gpu_extract.cu -- Phase B of our v4.5 ENC_RC solver: the MX Extract on GPU.
//
// WHY THIS ONE FIRST: profiling numair's production solver on pc showed his harness pegging ONE
// CPU core at 100% with the GPU at 0.8% util / 49 W -- ~40+ min per production episode against a
// ~44 ms GPU compute floor. The per-element MX Extract is the dominant term in that CPU time and
// is embarrassingly parallel (one independent 32-wide tile per (i,bj)), so it is both the biggest
// single win and the easiest thing to move.
//
// Correctness gate: every tile must equal the Phase-A CPU oracle (rc_cpu_solver.cpp) bit-for-bit.
// That oracle already reproduces his frozen golden b339d0ff, so matching it means matching him.
//
// Extract, per (i,bj) tile of 32 int64 raws:
//   ChaCha20(key = prf_key), nonce_first = bj ^ 'MXBL', nonce_second = (i<<32)|bj, counter = remix
//   per keystream byte, nibbles at shifts {0,4}:
//       mixed = (nibble ^ ((mix32(raw[filled]) * 0x9E3779B9) >> 28)) & 0xF
//       M11 rejection-sample (rejects 1,3,8,9,11); on accept mu[filled++] = value
//   e = SHA256("BTX_MATEXPAND_MXSCALE_V44LT" || prf || LE32(i) || LE32(bj))[0] & 3
//   out[t] = mu[t] << e
//
// build: nvcc -O3 -arch=sm_XX rc_gpu_extract.cu -o rc_gpu_extract
// usage: rc_gpu_extract [rows] [cols]      (defaults 4096 x 4096 -> 524288 tiles)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

using u8 = uint8_t; using u32 = uint32_t; using u64 = uint64_t;
using i8 = int8_t;  using i32 = int32_t; using i64 = int64_t;

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static constexpr u32 kBlockLen    = 32;
static constexpr u32 kLaneMxBlock = 0x4D58424Cu;   // 'MXBL'

// ============================================================ device SHA-256
__constant__ u32 dK256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ u32 dror(u32 x, int n) { return (x >> n) | (x << (32 - n)); }

// Single-block SHA-256 over a <=55-byte message (our scale preimage is 32+32+4+4 = 72 -> 2 blocks).
__device__ void dsha256(const u8* msg, u32 len, u8 out[32])
{
    u32 h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[128];
    u32 total = len;
    for (u32 i = 0; i < len; ++i) blk[i] = msg[i];
    blk[len] = 0x80;
    u32 padded = ((len + 9 + 63) / 64) * 64;
    for (u32 i = len + 1; i < padded; ++i) blk[i] = 0;
    const u64 bits = u64(total) * 8;
    for (int i = 0; i < 8; ++i) blk[padded - 1 - i] = u8(bits >> (8 * i));
    for (u32 b = 0; b < padded; b += 64) {
        u32 w[64];
#pragma unroll
        for (int i = 0; i < 16; ++i)
            w[i] = (u32(blk[b+i*4]) << 24) | (u32(blk[b+i*4+1]) << 16) |
                   (u32(blk[b+i*4+2]) << 8) | u32(blk[b+i*4+3]);
        for (int i = 16; i < 64; ++i) {
            u32 s0 = dror(w[i-15],7) ^ dror(w[i-15],18) ^ (w[i-15] >> 3);
            u32 s1 = dror(w[i-2],17) ^ dror(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        u32 a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            u32 S1 = dror(e,6) ^ dror(e,11) ^ dror(e,25);
            u32 ch = (e & f) ^ (~e & g);
            u32 t1 = hh + S1 + ch + dK256[i] + w[i];
            u32 S0 = dror(a,2) ^ dror(a,13) ^ dror(a,22);
            u32 mj = (a & bb) ^ (a & c) ^ (bb & c);
            u32 t2 = S0 + mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=bb; bb=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=bb; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i*4]=u8(h[i]>>24); out[i*4+1]=u8(h[i]>>16);
        out[i*4+2]=u8(h[i]>>8); out[i*4+3]=u8(h[i]);
    }
}

// ============================================================ device ChaCha20
__device__ __forceinline__ u32 drotl(u32 x, int n) { return (x << n) | (x >> (32 - n)); }

__device__ void dchacha_block(const u32 key[8], u32 nonce_first, u64 nonce_second,
                              u32 counter, u8 out[64])
{
    u32 in[16];
    in[0]=0x61707865; in[1]=0x3320646e; in[2]=0x79622d32; in[3]=0x6b206574;
#pragma unroll
    for (int i = 0; i < 8; ++i) in[4+i] = key[i];
    in[12] = counter;
    in[13] = nonce_first;
    in[14] = u32(nonce_second);
    in[15] = u32(nonce_second >> 32);
    u32 x[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = in[i];
    for (int r = 0; r < 10; ++r) {
#define QR(a,b,c,d) { x[a]+=x[b]; x[d]=drotl(x[d]^x[a],16); \
                      x[c]+=x[d]; x[b]=drotl(x[b]^x[c],12); \
                      x[a]+=x[b]; x[d]=drotl(x[d]^x[a], 8); \
                      x[c]+=x[d]; x[b]=drotl(x[b]^x[c], 7); }
        QR(0,4,8,12) QR(1,5,9,13) QR(2,6,10,14) QR(3,7,11,15)
        QR(0,5,10,15) QR(1,6,11,12) QR(2,7,8,13) QR(3,4,9,14)
#undef QR
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        u32 v = x[i] + in[i];
        out[i*4]=u8(v); out[i*4+1]=u8(v>>8); out[i*4+2]=u8(v>>16); out[i*4+3]=u8(v>>24);
    }
}

// ============================================================ M11 table (device constant)
// value[] / accepted[] for the 16 E2M1 codes; rejects {1,3,8,9,11}.
__constant__ i8 dM11v[16] = {0,0,1,0,2,3,4,6, 0,0,-1,0,-2,-3,-4,-6};
__constant__ u8 dM11a[16] = {1,0,1,0,1,1,1,1, 0,0, 1,0, 1, 1, 1, 1};

__device__ __forceinline__ u32 dmix_from_i64(i64 y)
{
    if (y >= (i64)INT32_MIN && y <= (i64)INT32_MAX) return u32((i32)y);
    const u64 u = (u64)y;
    return u32(u) ^ u32(u >> 32);
}

// One thread per (i,bj) tile -- fully independent, which is exactly why this moves to GPU well.
__global__ void k_extract_tiles(const u8* __restrict__ prf,   // 32 bytes
                                const i64* __restrict__ raw,  // rows*cols
                                i8* __restrict__ out,         // rows*cols
                                u32 rows, u32 cols)
{
    const u32 nblk = cols / kBlockLen;
    const u64 tile = u64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tile >= u64(rows) * nblk) return;
    const u32 i  = u32(tile / nblk);
    const u32 bj = u32(tile % nblk);
    const size_t base = size_t(i) * cols + size_t(bj) * kBlockLen;

    u32 key[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
        key[k] = u32(prf[k*4]) | (u32(prf[k*4+1])<<8) | (u32(prf[k*4+2])<<16) | (u32(prf[k*4+3])<<24);

    i8 mu[kBlockLen];
    u32 filled = 0, remix = 0;
    while (filled < kBlockLen) {
        u8 ks[64];
        dchacha_block(key, bj ^ kLaneMxBlock, (u64(i) << 32) | u64(bj), remix, ks);
        for (u32 b = 0; b < 64 && filled < kBlockLen; ++b) {
            for (int shift = 0; shift <= 4; shift += 4) {
                if (filled >= kBlockLen) break;
                const u8 nib = u8((ks[b] >> shift) & 0x0F);
                const u32 raw_u = dmix_from_i64(raw[base + filled]);
                const u8 mixed = u8((nib ^ u8((raw_u * 0x9E3779B9u) >> 28)) & 0x0F);
                if (dM11a[mixed]) mu[filled++] = dM11v[mixed];
            }
        }
        ++remix;
    }
    // scale: SHA256(tag || prf || LE32(i) || LE32(bj))[0] & 3
    u8 msg[27 + 32 + 8];
    const char tag[] = "BTX_MATEXPAND_MXSCALE_V44LT";
#pragma unroll
    for (int k = 0; k < 27; ++k) msg[k] = u8(tag[k]);
    for (int k = 0; k < 32; ++k) msg[27 + k] = prf[k];
    for (int k = 0; k < 4; ++k) { msg[59 + k] = u8(i >> (8*k)); msg[63 + k] = u8(bj >> (8*k)); }
    u8 h[32]; dsha256(msg, 27 + 32 + 8, h);
    const u8 e = u8(h[0] & 0x3);
#pragma unroll
    for (u32 t = 0; t < kBlockLen; ++t) out[base + t] = i8(i32(mu[t]) * (i32{1} << e));
}

// ============================================================ host CPU oracle (Phase A, verbatim)
struct Sha256H {
    u32 s[8]{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 buf[64]{}; size_t len=0; u64 bits=0;
    static u32 ror(u32 x,int n){return (x>>n)|(x<<(32-n));}
    void block(const u8* p){
        static const u32 K[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        u32 w[64];
        for(int i=0;i<16;++i) w[i]=(u32(p[i*4])<<24)|(u32(p[i*4+1])<<16)|(u32(p[i*4+2])<<8)|u32(p[i*4+3]);
        for(int i=16;i<64;++i){u32 s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3),
                                    s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
                               w[i]=w[i-16]+s0+w[i-7]+s1;}
        u32 a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        for(int i=0;i<64;++i){
            u32 S1=ror(e,6)^ror(e,11)^ror(e,25), ch=(e&f)^(~e&g), t1=h+S1+ch+K[i]+w[i];
            u32 S0=ror(a,2)^ror(a,13)^ror(a,22), mj=(a&b)^(a&c)^(b&c), t2=S0+mj;
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
    }
    void write(const void* dd,size_t n){const u8* p=(const u8*)dd; bits+=u64(n)*8;
        while(n){size_t t=64-len; if(t>n)t=n; memcpy(buf+len,p,t); len+=t;p+=t;n-=t;
                 if(len==64){block(buf);len=0;}}}
    void finalize(u8 o[32]){u64 b=bits; u8 pad=0x80; write(&pad,1); u8 z=0;
        while(len!=56) write(&z,1); u8 be[8];
        for(int i=0;i<8;++i) be[i]=u8(b>>(56-8*i)); write(be,8);
        for(int i=0;i<8;++i){o[i*4]=u8(s[i]>>24);o[i*4+1]=u8(s[i]>>16);o[i*4+2]=u8(s[i]>>8);o[i*4+3]=u8(s[i]);}}
};
struct ChaChaH {
    u32 in[16]{};
    explicit ChaChaH(const u8 k[32]){
        in[0]=0x61707865;in[1]=0x3320646e;in[2]=0x79622d32;in[3]=0x6b206574;
        for(int i=0;i<8;++i) in[4+i]=u32(k[i*4])|(u32(k[i*4+1])<<8)|(u32(k[i*4+2])<<16)|(u32(k[i*4+3])<<24);
    }
    void seek(u32 nf,u64 ns,u32 c){in[12]=c;in[13]=nf;in[14]=u32(ns);in[15]=u32(ns>>32);}
    static u32 rotl(u32 x,int n){return (x<<n)|(x>>(32-n));}
    void keystream(u8* o,size_t n){
        while(n){u32 x[16]; memcpy(x,in,sizeof(x));
            for(int r=0;r<10;++r){
                auto qr=[&](int a,int b,int c,int d){
                    x[a]+=x[b];x[d]=rotl(x[d]^x[a],16); x[c]+=x[d];x[b]=rotl(x[b]^x[c],12);
                    x[a]+=x[b];x[d]=rotl(x[d]^x[a],8);  x[c]+=x[d];x[b]=rotl(x[b]^x[c],7);};
                qr(0,4,8,12);qr(1,5,9,13);qr(2,6,10,14);qr(3,7,11,15);
                qr(0,5,10,15);qr(1,6,11,12);qr(2,7,8,13);qr(3,4,9,14);}
            u8 blk[64];
            for(int i=0;i<16;++i){u32 v=x[i]+in[i];
                blk[i*4]=u8(v);blk[i*4+1]=u8(v>>8);blk[i*4+2]=u8(v>>16);blk[i*4+3]=u8(v>>24);}
            size_t t=n<64?n:64; memcpy(o,blk,t); o+=t;n-=t; ++in[12];}
    }
};
static const i8 hM11v[16]={0,0,1,0,2,3,4,6, 0,0,-1,0,-2,-3,-4,-6};
static const u8 hM11a[16]={1,0,1,0,1,1,1,1, 0,0, 1,0, 1, 1, 1, 1};
static u32 hmix(i64 y){ if(y>=(i64)INT32_MIN&&y<=(i64)INT32_MAX) return u32((i32)y);
                        u64 u=(u64)y; return u32(u)^u32(u>>32); }
static void cpu_extract_tile(const u8 prf[32], u32 i, u32 bj, const i64* raw, i8* out)
{
    i8 mu[32]; u32 filled=0, remix=0;
    while (filled < 32) {
        ChaChaH cc(prf); cc.seek(bj ^ kLaneMxBlock, (u64(i)<<32)|u64(bj), remix);
        u8 ks[64]; cc.keystream(ks,64);
        for (size_t b=0;b<64&&filled<32;++b)
            for (int sh=0; sh<=4; sh+=4) {
                if (filled>=32) break;
                const u8 nib=u8((ks[b]>>sh)&0x0F);
                const u8 mixed=u8((nib ^ u8((hmix(raw[filled])*0x9E3779B9u)>>28))&0x0F);
                if (hM11a[mixed]) mu[filled++]=hM11v[mixed];
            }
        ++remix;
    }
    Sha256H h; const char tag[]="BTX_MATEXPAND_MXSCALE_V44LT";
    h.write(tag,27); h.write(prf,32);
    u8 il[4],bl[4]; for(int k=0;k<4;++k){il[k]=u8(i>>(8*k));bl[k]=u8(bj>>(8*k));}
    h.write(il,4); h.write(bl,4);
    u8 o[32]; h.finalize(o); const u8 e=u8(o[0]&3);
    for (u32 t=0;t<32;++t) out[t]=i8(i32(mu[t])*(i32{1}<<e));
}

int main(int argc, char** argv)
{
    const u32 rows = argc>1 ? (u32)atoi(argv[1]) : 4096;
    const u32 cols = argc>2 ? (u32)atoi(argv[2]) : 4096;
    const u32 nblk = cols / kBlockLen;
    const size_t n = size_t(rows) * cols;
    const size_t ntile = size_t(rows) * nblk;

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d | Extract %ux%u = %zu tiles (%zu elements)\n",
           p.name, p.major, p.minor, rows, cols, ntile, n);

    u8 prf[32]; for (int k=0;k<32;++k) prf[k] = u8(0x13 + k*7);
    std::vector<i64> raw(n);
    for (size_t t=0;t<n;++t) raw[t] = i64((t*2654435761u) % 100000) - 50000;

    u8* dprf; i64* draw; i8* dout;
    CK(cudaMalloc(&dprf,32)); CK(cudaMalloc(&draw,n*sizeof(i64))); CK(cudaMalloc(&dout,n));
    CK(cudaMemcpy(dprf,prf,32,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(draw,raw.data(),n*sizeof(i64),cudaMemcpyHostToDevice));

    const int T=256; const u32 blocks=(u32)((ntile+T-1)/T);
    k_extract_tiles<<<blocks,T>>>(dprf,draw,dout,rows,cols);   // warmup
    CK(cudaDeviceSynchronize());

    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    CK(cudaEventRecord(a));
    k_extract_tiles<<<blocks,T>>>(dprf,draw,dout,rows,cols);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms,a,b);

    std::vector<i8> got(n); CK(cudaMemcpy(got.data(),dout,n,cudaMemcpyDeviceToHost));

    // correctness: sample tiles against the Phase-A CPU oracle
    size_t checked=0, bad=0;
    for (size_t t=0; t<ntile && checked<4096; t += (ntile/4096 ? ntile/4096 : 1)) {
        const u32 i=u32(t/nblk), bj=u32(t%nblk);
        const size_t base=size_t(i)*cols+size_t(bj)*kBlockLen;
        i8 want[32]; cpu_extract_tile(prf,i,bj,raw.data()+base,want);
        for (u32 k2=0;k2<32;++k2) if (want[k2]!=got[base+k2]) ++bad;
        ++checked;
    }
    printf("  oracle check: %zu tiles sampled, %zu mismatched byte%s -> %s\n",
           checked, bad, bad==1?"":"s", bad? "FAIL":"PASS");

    // CPU throughput for the same work, single-threaded (his solver's regime)
    const size_t cpu_tiles = ntile < 20000 ? ntile : 20000;
    std::vector<i8> tmp(32);
    cudaEvent_t c0,c1; (void)c0; (void)c1;
    const clock_t t0=clock();
    for (size_t t=0;t<cpu_tiles;++t) {
        const u32 i=u32(t/nblk), bj=u32(t%nblk);
        cpu_extract_tile(prf,i,bj,raw.data()+size_t(i)*cols+size_t(bj)*kBlockLen,tmp.data());
    }
    const double cpu_s=double(clock()-t0)/CLOCKS_PER_SEC;
    const double cpu_full_ms = cpu_s * 1000.0 * double(ntile)/double(cpu_tiles);

    printf("  GPU: %8.2f ms   (%.1f Mtile/s)\n", ms, ntile/1e6/(ms/1000.0));
    printf("  CPU: %8.2f ms   (1 core, extrapolated from %zu tiles)\n", cpu_full_ms, cpu_tiles);
    printf("  speedup: %.0fx\n", cpu_full_ms/ms);
    return bad ? 1 : 0;
}
