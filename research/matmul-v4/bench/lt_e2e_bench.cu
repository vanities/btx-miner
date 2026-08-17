// End-to-end v4.4-LT solve bench: the FULL per-nonce cost, not the compute ceiling.
//   per nonce: operand-gen (GPU SHA256 mant_stream, nonce-fresh W_bhat) -> MatExpand
//   (Y=G*W, tensor-B32=Y*H, MX-block Extract) -> Q=Bhat*V -> 3-base-128 mod-q combine
//   -> S4 digest (GPU SHA256d over the ~33 MiB Chat). Template-scoped work (G/H/Ahat/U/V/P)
//   is derived ONCE per job (hoisted), like a real deploy solver.
// SHA256 + digest_chain kernels are the proven c13_bench.cu ones; operand-gen models the
// LT mant_stream SHA cost (per-block, ~44 M11 values / SHA256). Perf-representative, not
// byte-exact (a perf harness). build: nvcc -O3 -arch=sm_120 lt_e2e_bench.cu -lcublasLt -lnvidia-ml
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvml.h>
using std::vector; using u8=uint8_t; using i8=int8_t; using i32=int32_t; using u32=uint32_t; using u64=uint64_t; using i64=int64_t;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("CUDA %s:%d %s->%s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));return 2;}}while(0)

__device__ __constant__ u32 K256[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7};
__device__ __forceinline__ u32 ror(u32 x,int r){return (x>>r)|(x<<(32-r));}
// SHA-256 of msg (len<=55) -> out[32], one block. (c13_bench.cu)
__device__ void sha256_1blk(const u8* msg,int len,u8 out[32]){
    u8 blk[64];
    #pragma unroll
    for(int i=0;i<64;i++) blk[i]=0;
    for(int i=0;i<len;i++) blk[i]=msg[i];
    blk[len]=0x80; u64 bits=(u64)len*8; for(int i=0;i<8;i++) blk[63-i]=(u8)(bits>>(8*i));
    u32 w[64];
    #pragma unroll
    for(int i=0;i<16;i++) w[i]=(blk[i*4]<<24)|(blk[i*4+1]<<16)|(blk[i*4+2]<<8)|blk[i*4+3];
    #pragma unroll
    for(int i=16;i<64;i++){u32 s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3),s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
    u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma unroll
    for(int i=0;i<64;i++){u32 S1=ror(e,6)^ror(e,11)^ror(e,25),ch=(e&f)^((~e)&g),t1=h+S1+ch+K256[i]+w[i],S0=ror(a,2)^ror(a,13)^ror(a,22),mj=(a&b)^(a&c)^(b&c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+S0+mj;}
    u32 H[8]={0x6a09e667+a,0xbb67ae85+b,0x3c6ef372+c,0xa54ff53a+d,0x510e527f+e,0x9b05688c+f,0x1f83d9ab+g,0x5be0cd19+h};
    #pragma unroll
    for(int i=0;i<8;i++){out[i*4]=H[i]>>24;out[i*4+1]=H[i]>>16;out[i*4+2]=H[i]>>8;out[i*4+3]=H[i];}
}
__device__ __forceinline__ void sha256_compress_w(u32* H,const u32* w0){
    u32 w[64];
    #pragma unroll
    for(int i=0;i<16;i++) w[i]=w0[i];
    #pragma unroll
    for(int i=16;i<64;i++){u32 s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3),s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
    u32 a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
    #pragma unroll
    for(int i=0;i<64;i++){u32 S1=ror(e,6)^ror(e,11)^ror(e,25),ch=(e&f)^((~e)&g),t1=h+S1+ch+K256[i]+w[i],S0=ror(a,2)^ror(a,13)^ror(a,22),mj=(a&b)^(a&c)^(b&c);h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+S0+mj;}
    H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
}

// M11 mantissa alphabet.
__constant__ bool dM_ACC[16]; __constant__ i8 dM_VAL[16];
static const bool M_ACC[16]={1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1};
static const i8   M_VAL[16]={0,0,1,0,2,3,4,6,0,0,-1,0,-2,-3,-4,-6};

// LT operand-gen: SHA256(seedLE[32] || 0x6D || blk_le64) -> ~44 M11 values (2 nibbles/byte).
// One thread = one SHA block, writes per_hash values at base=blk*per_hash (perf model of the
// mant_stream SHA cost; shortfall padded). seedLE is host-reversed (SeedBytesLE).
__global__ void lt_mant_stream(i8* out,u32 salt,u32 count,u32 per_hash){
    u32 blk=blockIdx.x*blockDim.x+threadIdx.x; u32 base=blk*per_hash; if(base>=count) return;
    u8 msg[41];
    #pragma unroll
    for(int i=0;i<32;i++) msg[i]=(u8)((salt*2654435761u)>>((i&3)*8))^i;   // perf-model seed from salt (no H2D)
    msg[32]=0x6D; for(int i=0;i<8;i++) msg[33+i]=(u8)((u64)blk>>(8*i));
    u8 h[32]; sha256_1blk(msg,41,h);
    u32 w=0;
    for(int i=0;i<32 && w<per_hash && base+w<count;++i){
        u8 lo=h[i]&0xF, hi=(h[i]>>4)&0xF;
        if(dM_ACC[lo] && w<per_hash && base+w<count) out[base+w++]=dM_VAL[lo];
        if(dM_ACC[hi] && w<per_hash && base+w<count) out[base+w++]=dM_VAL[hi];
    }
    for(;w<per_hash && base+w<count;++w) out[base+w]=0;
}

// ---- ChaCha extract (MX-block perf model: 1 PRF / 32-block scale + cheap mantissa) ----
__device__ __forceinline__ u32 drotl(u32 x,int n){return (x<<n)|(x>>(32-n));}
__device__ void dchacha(const u32 key[8],u32 ctr,u32 n0,u32 n1,u32 n2,u8 out[64]){
    u32 s[16]={0x61707865,0x3320646e,0x79622d32,0x6b206574,key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7],ctr,n0,n1,n2},x[16];
    #pragma unroll
    for(int i=0;i<16;i++)x[i]=s[i];
    auto QR=[&](int a,int b,int c,int d){x[a]+=x[b];x[d]=drotl(x[d]^x[a],16);x[c]+=x[d];x[b]=drotl(x[b]^x[c],12);x[a]+=x[b];x[d]=drotl(x[d]^x[a],8);x[c]+=x[d];x[b]=drotl(x[b]^x[c],7);};
    #pragma unroll
    for(int i=0;i<10;i++){QR(0,4,8,12);QR(1,5,9,13);QR(2,6,10,14);QR(3,7,11,15);QR(0,5,10,15);QR(1,6,11,12);QR(2,7,8,13);QR(3,4,9,14);}
    #pragma unroll
    for(int i=0;i<16;i++){u32 v=x[i]+s[i];out[i*4]=v;out[i*4+1]=v>>8;out[i*4+2]=v>>16;out[i*4+3]=v>>24;}
}
__device__ u64 dprf(const u32 key[8],i32 raw,u32 i,u32 j,u32 rm,u32 lane){u32 n0=(u32)raw^lane;u64 ns=((u64)i<<32)|(u64)j;u8 ks[64];dchacha(key,rm,n0,(u32)(ns&0xffffffff),(u32)(ns>>32),ks);u64 v=0;for(int k=0;k<8;k++)v|=(u64)ks[k]<<(8*k);return v;}
__global__ void k_mx_scales(const u32* key,u8* scales,u32 nblk){u32 b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=nblk)return;u64 sc=dprf(key,(i32)b,b,0,0,0x4D58424Cu);scales[b]=(u8)(sc&0x3);}
__global__ void k_extract_mx(const i32* B32,const u8* scales,i8* Bhat,u32 n){u32 idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=n*n)return;i32 raw=B32[idx];u8 e=scales[idx>>5];u8 nb=(u8)((raw^(raw>>7))&0xF);Bhat[idx]=(i8)((i32)dM_VAL[nb]*(1<<e));}

// ---- tensor-B32 (3-limb int8 split of Y) ----
__global__ void k_splitY3(const i32* Y,i8* Ys,u32 n,u32 w){size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=(size_t)n*w)return;u32 i=(u32)(idx/w),k=(u32)(idx%w);i32 x=Y[idx];
 #pragma unroll
 for(int l=0;l<2;l++){i32 d=((x+128)&255)-128;Ys[(size_t)(l*n+i)*w+k]=(i8)d;x=(x-d)/256;}Ys[(size_t)(2*n+i)*w+k]=(i8)x;}
__global__ void k_recombine3(const i32* Bs,i32* B32,u32 n){size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=(size_t)n*n)return;u32 i=(u32)(idx/n),j=(u32)(idx%n);i64 v=(i64)Bs[(size_t)i*n+j]+256LL*(i64)Bs[(size_t)(n+i)*n+j]+65536LL*(i64)Bs[(size_t)(2*n+i)*n+j];B32[idx]=(i32)v;}

// ---- 3-base-128 mod-q combine ----
__device__ __forceinline__ u64 dfqred(unsigned __int128 x){const u64 Q=(((u64)1)<<61)-1;u64 lo=(u64)(x&Q),hi=(u64)(x>>61);u64 s=lo+hi;s=(s&Q)+(s>>61);if(s>=Q)s-=Q;return s;}
__device__ __forceinline__ u64 dfq(i32 x){const u64 Q=(((u64)1)<<61)-1;return x>=0?(u64)x:Q-(u64)(-(i64)x);}
__device__ __forceinline__ u64 fqadd(u64 a,u64 b){const u64 Q=(((u64)1)<<61)-1;u64 s=a+b;if(s>=Q)s-=Q;return s;}
__device__ __forceinline__ u64 fqmul(u64 a,u64 b){return dfqred((unsigned __int128)a*b);}
constexpr int LIMB3=3;
__global__ void k_split3(const i32* X,i8* planes,size_t count){size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=count)return;i32 x=X[idx];
 #pragma unroll
 for(int l=0;l<LIMB3-1;l++){i32 d=((x+64)&127)-64;planes[(size_t)l*count+idx]=(i8)d;x=(x-d)/128;}planes[(size_t)(LIMB3-1)*count+idx]=(i8)x;}
__global__ void k_split_qh3(const i32* X,i8* qlh,u32 n,u32 mcol){size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;size_t count=(size_t)n*mcol;if(idx>=count)return;u32 r=(u32)(idx/mcol),c=(u32)(idx%mcol);i32 x=X[idx];
 #pragma unroll
 for(int l=0;l<LIMB3-1;l++){i32 d=((x+64)&127)-64;qlh[(size_t)r*(LIMB3*mcol)+(size_t)l*mcol+c]=(i8)d;x=(x-d)/128;}qlh[(size_t)r*(LIMB3*mcol)+(size_t)(LIMB3-1)*mcol+c]=(i8)x;}
__global__ void k_reconstruct3(const i32* Gbig,u64* Chat,u32 m){u32 a=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(a>=m||c>=m)return;const size_t M3=(size_t)LIMB3*m;u64 acc=0;
 #pragma unroll
 for(int i=0;i<LIMB3;i++)for(int j=0;j<LIMB3;j++){i32 g=Gbig[(size_t)(i*m+a)*M3+(size_t)(j*m+c)];acc=fqadd(acc,fqmul(((u64)1)<<(7*(i+j)),dfq(g)));}Chat[(size_t)a*m+c]=acc;}

// S4 digest: SHA256d(tag||sigma[45] || LE64(Chat)) per chain. (c13_bench.cu)
__global__ void digest_chain(const u8* __restrict__ hdr,int hlen,const u8* __restrict__ pays,size_t stride,int nbuf,size_t plen,int nchains,u8* __restrict__ out){
    int tid=blockIdx.x*blockDim.x+threadIdx.x; if(tid>=nchains) return;
    const u8* pay=pays+(size_t)(tid%nbuf)*stride;
    u32 H[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    const u64 total=(u64)hlen+plen; const u64 nb=(total+1+8+63)/64;
    const bool a45=(hlen==45)&&(((uintptr_t)pay&3u)==0);
    for(u64 bi=0;bi<nb;++bi){const u64 base=bi*64;u32 w[16];
        if(a45&&base>=64&&base+64<=total){const u32* pw=(const u32*)(pay+(base-48));u32 wa[17];
            #pragma unroll
            for(int k2=0;k2<17;k2++) wa[k2]=__ldg(pw+k2);
            #pragma unroll
            for(int i=0;i<16;i++) w[i]=__byte_perm(wa[i],wa[i+1],0x3456);
        }else{
            #pragma unroll
            for(int i=0;i<16;i++){u32 v32=0;for(int k2=0;k2<4;k2++){u64 pos=base+i*4+k2;u8 v;if(pos<(u64)hlen)v=hdr[pos];else if(pos<total)v=__ldg(&pay[pos-hlen]);else if(pos==total)v=0x80;else if(pos>=nb*64-8)v=(u8)((total*8)>>(8*(nb*64-1-pos)));else v=0;v32=(v32<<8)|v;}w[i]=v32;}}
        sha256_compress_w(H,w);}
    u8 d1[32];
    #pragma unroll
    for(int i=0;i<8;i++){d1[i*4]=H[i]>>24;d1[i*4+1]=H[i]>>16;d1[i*4+2]=H[i]>>8;d1[i*4+3]=H[i];}
    u8 d2[32]; sha256_1blk(d1,32,d2); for(int i=0;i<32;i++) out[(size_t)tid*32+i]=d2[i];
}

// 4-WAY interleaved digest: one thread drives 4 independent SHA256d chains, interleaving their
// compressions so the SM has ILP to hide the serial per-chain latency (the digest is latency-bound,
// not bandwidth-bound: sm~100% but ~40% power at any ring). Same memory as digest_chain.
__global__ void digest_chain4(const u8* __restrict__ hdr,int hlen,const u8* __restrict__ pays,size_t stride,size_t plen,int nchains,u8* __restrict__ out){
    const int base=(blockIdx.x*blockDim.x+threadIdx.x)*4; if(base>=nchains) return;
    u32 H[4][8]; const u8* pay[4]; bool live[4];
    #pragma unroll
    for(int L=0;L<4;L++){ live[L]=(base+L)<nchains; pay[L]=pays+(size_t)(base+L)*stride;
        H[L][0]=0x6a09e667;H[L][1]=0xbb67ae85;H[L][2]=0x3c6ef372;H[L][3]=0xa54ff53a;H[L][4]=0x510e527f;H[L][5]=0x9b05688c;H[L][6]=0x1f83d9ab;H[L][7]=0x5be0cd19; }
    const u64 total=(u64)hlen+plen; const u64 nb=(total+1+8+63)/64;
    for(u64 bi=0;bi<nb;++bi){ const u64 b=bi*64;
        u32 w[4][16];
        #pragma unroll
        for(int L=0;L<4;L++){ if(!live[L]) continue; const u8* p=pay[L];
            if(base+0>=0 && b>=64 && b+64<=total && (((uintptr_t)p&3u)==0)){ const u32* pw=(const u32*)(p+(b-48)); u32 wa[17];
                #pragma unroll
                for(int k2=0;k2<17;k2++) wa[k2]=__ldg(pw+k2);
                #pragma unroll
                for(int i=0;i<16;i++) w[L][i]=__byte_perm(wa[i],wa[i+1],0x3456);
            } else {
                #pragma unroll
                for(int i=0;i<16;i++){ u32 v=0; for(int k2=0;k2<4;k2++){ u64 pos=b+i*4+k2; u8 v8; if(pos<(u64)hlen)v8=hdr[pos]; else if(pos<total)v8=__ldg(&p[pos-hlen]); else if(pos==total)v8=0x80; else if(pos>=nb*64-8)v8=(u8)((total*8)>>(8*(nb*64-1-pos))); else v8=0; v=(v<<8)|v8; } w[L][i]=v; }
            }
        }
        #pragma unroll
        for(int L=0;L<4;L++) if(live[L]) sha256_compress_w(H[L],w[L]);   // 4 independent compressions -> ILP
    }
    #pragma unroll
    for(int L=0;L<4;L++){ if(!live[L]) continue; u8 d1[32];
        #pragma unroll
        for(int i=0;i<8;i++){d1[i*4]=H[L][i]>>24;d1[i*4+1]=H[L][i]>>16;d1[i*4+2]=H[L][i]>>8;d1[i*4+3]=H[L][i];}
        u8 d2[32]; sha256_1blk(d1,32,d2); for(int i=0;i<32;i++) out[(size_t)(base+L)*32+i]=d2[i]; }
}

static bool gemm8(cublasLtHandle_t lt,void*ws,size_t wsz,const i8*dL,const i8*dR,i32*dO,u32 rows,u32 inner,u32 cols,cudaStream_t st=0){
    cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;cublasLtMatmulPreference_t pf=nullptr;bool ok=false;
    do{if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I))break;cublasOperation_t N=CUBLAS_OP_N;cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&N,sizeof N);cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N,sizeof N);cublasLtOrder_t ro=CUBLASLT_ORDER_ROW;auto mk=[&](cublasLtMatrixLayout_t*L,cudaDataType t,u64 r,u64 c,i64 ld){if(cublasLtMatrixLayoutCreate(L,t,r,c,ld))return false;return 0==cublasLtMatrixLayoutSetAttribute(*L,CUBLASLT_MATRIX_LAYOUT_ORDER,&ro,sizeof ro);};if(!mk(&la,CUDA_R_8I,rows,inner,inner)||!mk(&lb,CUDA_R_8I,inner,cols,cols)||!mk(&lc,CUDA_R_32I,rows,cols,cols))break;cublasLtMatmulPreferenceCreate(&pf);cublasLtMatmulPreferenceSetAttribute(pf,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof wsz);cublasLtMatmulHeuristicResult_t h{};int got=0;if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pf,1,&h,&got)||!got)break;i32 al=1,be=0;if(cublasLtMatmul(lt,op,&al,dL,la,dR,lb,&be,dO,lc,dO,lc,&h.algo,ws,wsz,st))break;ok=true;}while(0);
    if(pf)cublasLtMatmulPreferenceDestroy(pf);if(lc)cublasLtMatrixLayoutDestroy(lc);if(lb)cublasLtMatrixLayoutDestroy(lb);if(la)cublasLtMatrixLayoutDestroy(la);if(op)cublasLtMatmulDescDestroy(op);return ok;
}

int main(int argc,char**argv){
    const u32 n=argc>1?atoi(argv[1]):4096, m=n/2, w=1024;
    const double secs=argc>2?atof(argv[2]):15.0;
    const u32 PH=44;   // ~M11 values per SHA256 (11/16 accept * 64 nibbles)
    cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d | LT n=%u m=%u w=%u (END-TO-END: operand-gen + compute + S4 digest)\n",p.name,p.major,p.minor,n,m,w);
    nvmlInit(); nvmlDevice_t nv; nvmlDeviceGetHandleByIndex(0,&nv);
    CK(cudaMemcpyToSymbol(dM_ACC,M_ACC,16));CK(cudaMemcpyToSymbol(dM_VAL,M_VAL,16));
    cublasLtHandle_t lt;cublasLtCreate(&lt);void*ws;const size_t WS=(size_t)256<<20;CK(cudaMalloc(&ws,WS));

    // device buffers
    i8 *dG,*dH,*dW,*dAh,*dU,*dV,*dBh,*dYstack,*dPlimb,*dQlh; i32 *dY,*dB32,*dBstack,*dP,*dQ,*dGbig; u64 *dC; u32 *dkey; u8 *dScales,*dseedLE,*dhdr,*dout;
    CK(cudaMalloc(&dG,(size_t)n*n));CK(cudaMalloc(&dH,(size_t)w*n));CK(cudaMalloc(&dW,(size_t)n*w));CK(cudaMalloc(&dAh,(size_t)n*n));
    CK(cudaMalloc(&dU,(size_t)m*n));CK(cudaMalloc(&dV,(size_t)n*m));CK(cudaMalloc(&dBh,(size_t)n*n));
    CK(cudaMalloc(&dY,(size_t)n*w*4));CK(cudaMalloc(&dB32,(size_t)n*n*4));CK(cudaMalloc(&dYstack,(size_t)3*n*w));CK(cudaMalloc(&dBstack,(size_t)3*n*n*4));
    CK(cudaMalloc(&dP,(size_t)m*n*4));CK(cudaMalloc(&dQ,(size_t)n*m*4));CK(cudaMalloc(&dC,(size_t)m*m*8));
    CK(cudaMalloc(&dPlimb,(size_t)LIMB3*m*n));CK(cudaMalloc(&dQlh,(size_t)n*LIMB3*m));CK(cudaMalloc(&dGbig,(size_t)LIMB3*m*LIMB3*m*4));
    CK(cudaMalloc(&dScales,(size_t)((n*n+31)/32)));CK(cudaMalloc(&dkey,32));CK(cudaMalloc(&dseedLE,32));CK(cudaMalloc(&dhdr,45));CK(cudaMalloc(&dout,32));
    u32 hkey[8]={1,2,3,4,5,6,7,8};CK(cudaMemcpy(dkey,hkey,32,cudaMemcpyHostToDevice));
    { u8 hdr[45]; const char*t="BTX_MATMUL_V4"; memcpy(hdr,t,13); memset(hdr+13,0x5a,32); CK(cudaMemcpy(dhdr,hdr,45,cudaMemcpyHostToDevice)); }

    auto opgen=[&](i8* out,u32 salt,size_t count){ u32 nb=(u32)((count+PH-1)/PH); lt_mant_stream<<<(nb+255)/256,256>>>(out,salt,(u32)count,PH); };
    // ---- template setup (ONCE per job; hoisted, not in the per-nonce cost) ----
    opgen(dG,1,(size_t)n*n); opgen(dH,2,(size_t)w*n); opgen(dU,5,(size_t)m*n); opgen(dV,6,(size_t)n*m);
    opgen(dW,3,(size_t)n*w); gemm8(lt,ws,WS,dG,dW,dY,n,n,w);   // build Ahat once (reuse dW/dY scratch)
    k_splitY3<<<((size_t)n*w+255)/256,256>>>(dY,dYstack,n,w); gemm8(lt,ws,WS,dYstack,dH,dBstack,3*n,w,n); k_recombine3<<<((size_t)n*n+255)/256,256>>>(dBstack,dB32,n);
    k_mx_scales<<<(((n*n+31)/32)+255)/256,256>>>(dkey,dScales,(n*n+31)/32); k_extract_mx<<<(n*n+255)/256,256>>>(dB32,dScales,dAh,n);  // dAh = Ahat
    gemm8(lt,ws,WS,dU,dAh,dP,m,n,n);   // P = U*Ahat (template)
    dim3 rbl(16,16),rgr((m+15)/16,(m+15)/16);

    // ---- one full per-nonce solve ----
    auto opgen_bhat=[&](u32 nonce){ opgen(dW,0x1000u+nonce,(size_t)n*w); };   // nonce-fresh W_bhat
    auto matexpand_bhat=[&](){ gemm8(lt,ws,WS,dG,dW,dY,n,n,w); k_splitY3<<<((size_t)n*w+255)/256,256>>>(dY,dYstack,n,w); gemm8(lt,ws,WS,dYstack,dH,dBstack,3*n,w,n); k_recombine3<<<((size_t)n*n+255)/256,256>>>(dBstack,dB32,n); k_mx_scales<<<(((n*n+31)/32)+255)/256,256>>>(dkey,dScales,(n*n+31)/32); k_extract_mx<<<(n*n+255)/256,256>>>(dB32,dScales,dBh,n); };
    auto combine=[&](){ k_split3<<<((size_t)m*n+255)/256,256>>>(dP,dPlimb,(size_t)m*n); k_split_qh3<<<((size_t)n*m+255)/256,256>>>(dQ,dQlh,n,m); gemm8(lt,ws,WS,dPlimb,dQlh,dGbig,LIMB3*m,n,LIMB3*m); k_reconstruct3<<<rgr,rbl>>>(dGbig,dC,m); };
    auto digest=[&](){ digest_chain<<<1,1>>>(dhdr,45,(const u8*)dC,(size_t)m*m*8,1,(size_t)m*m*8,1,dout); };
    auto nonce=[&](u32 k){ opgen_bhat(k); matexpand_bhat(); gemm8(lt,ws,WS,dBh,dV,dQ,n,n,m); combine(); digest(); };
    nonce(0); CK(cudaDeviceSynchronize());

    // ---- per-stage breakdown (20 iters) ----
    { cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);const int R=20;
      auto tt=[&](const char*nm,auto fn){cudaDeviceSynchronize();cudaEventRecord(a);for(int r=0;r<R;r++)fn();cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);printf("  stage %-12s %8.3f ms/nonce\n",nm,ms/R);};
      tt("operand-gen",[&]{ opgen_bhat(0); });
      tt("Y=G*W",      [&]{ gemm8(lt,ws,WS,dG,dW,dY,n,n,w); });
      tt("B32-tensor", [&]{ k_splitY3<<<((size_t)n*w+255)/256,256>>>(dY,dYstack,n,w); gemm8(lt,ws,WS,dYstack,dH,dBstack,3*n,w,n); k_recombine3<<<((size_t)n*n+255)/256,256>>>(dBstack,dB32,n); });
      tt("extract",    [&]{ k_mx_scales<<<(((n*n+31)/32)+255)/256,256>>>(dkey,dScales,(n*n+31)/32); k_extract_mx<<<(n*n+255)/256,256>>>(dB32,dScales,dBh,n); });
      tt("Q=Bhat*V",   [&]{ gemm8(lt,ws,WS,dBh,dV,dQ,n,n,m); });
      tt("combine",    [&]{ combine(); });
      tt("S4 digest",  [&]{ digest(); });
    }

    // ---- sustained end-to-end nonce/s: SINGLE digest ring (max chains = max latency hiding). The
    //      double-buffered overlap was tested and loses: halving chains costs more digest parallelism
    //      than the produce/digest overlap buys back. So spend all memory on one ring. ----
    // ring size: explicit arg, or AUTO-SIZE to this card's free VRAM (ring 0/absent).
    // Auto is the fair cross-card method: every GPU filled to its own capacity, which is
    // exactly what the 33.5 MiB/chain Chat gates. 0.88 leaves headroom for cublasLt workspaces.
    const u32 RING_ARG = argc>3?(u32)atoi(argv[3]):0;
    size_t freeB=0,totB=0; CK(cudaMemGetInfo(&freeB,&totB));
    const size_t per_chain=(size_t)m*m*8+32;
    u32 RING = RING_ARG ? RING_ARG : (u32)((double)freeB*0.88/(double)per_chain);
    if(RING<1) RING=1;
    printf("  vram: %.1f/%.1f GiB free | chain %.2f MiB | ring %u (%s)\n",
           freeB/1073741824.0, totB/1073741824.0, per_chain/1048576.0, RING,
           RING_ARG?"explicit":"AUTO");
    u64* dCring; CK(cudaMalloc(&dCring,(size_t)RING*m*m*8)); u8* doutR; CK(cudaMalloc(&doutR,(size_t)RING*32));
    auto produce1=[&](u32 k,u64* cslot){
        u32 cw=(u32)((size_t)n*w); lt_mant_stream<<<((cw+PH-1)/PH+255)/256,256>>>(dW,0x1000u+k,cw,PH);   // nonce-fresh W_bhat
        gemm8(lt,ws,WS,dG,dW,dY,n,n,w);
        k_splitY3<<<((size_t)n*w+255)/256,256>>>(dY,dYstack,n,w); gemm8(lt,ws,WS,dYstack,dH,dBstack,3*n,w,n); k_recombine3<<<((size_t)n*n+255)/256,256>>>(dBstack,dB32,n);
        k_mx_scales<<<(((n*n+31)/32)+255)/256,256>>>(dkey,dScales,(n*n+31)/32); k_extract_mx<<<(n*n+255)/256,256>>>(dB32,dScales,dBh,n);
        gemm8(lt,ws,WS,dBh,dV,dQ,n,n,m);
        k_split3<<<((size_t)m*n+255)/256,256>>>(dP,dPlimb,(size_t)m*n); k_split_qh3<<<((size_t)n*m+255)/256,256>>>(dQ,dQlh,n,m); gemm8(lt,ws,WS,dPlimb,dQlh,dGbig,LIMB3*m,n,LIMB3*m); k_reconstruct3<<<rgr,rbl>>>(dGbig,cslot,m);
    };
    printf("  single digest ring = %u chains (%.1f GiB Chat)\n", RING, (double)RING*m*m*8/(1024.0*1024*1024));
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);long cnt=0;double pw=0;int pn=0;cudaEventRecord(t0);double el=0;
    while(el<secs){
        for(u32 k=0;k<RING;k++) produce1((u32)cnt+k, dCring+(size_t)k*m*m);
        digest_chain<<<(RING+255)/256,256>>>(dhdr,45,(const u8*)dCring,(size_t)m*m*8,(int)RING,(size_t)m*m*8,(int)RING,doutR);  // 1-way wins: 4-way interleave collapses occupancy (tested: 139 vs 381)
        cnt+=RING; CK(cudaDeviceSynchronize());cudaEventRecord(t1);cudaEventSynchronize(t1);float ms;cudaEventElapsedTime(&ms,t0,t1);el=ms/1000.0;unsigned mw;if(nvmlDeviceGetPowerUsage(nv,&mw)==NVML_SUCCESS){pw+=mw;pn++;} }
    printf("LT END-TO-END (ring %u): %ld nonces in %.1fs => %.1f nonce/s  |  ~%.0f W\n",RING,cnt,el,cnt/el,pn?pw/pn/1000.0:0.0);
    nvmlShutdown(); return 0;
}
