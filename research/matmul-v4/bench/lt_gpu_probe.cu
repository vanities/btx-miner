// matador LT GPU solver — byte-exact gate to db1136f2 (n=64 test header) on sm_120.
// GPU does the LT compute (MatExpand GEMMs + ChaCha Extract + P/Q + mod-q combine);
// host derives seeds + M11 projectors (proven byte-exact) + final SHA256d digest.
// This is the GPU port gate; kernels here become the deploy path we then optimize.
// build: nvcc -O3 -arch=sm_120 lt_gpu_probe.cu -lcublasLt -o lt_gpu
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
using std::vector; using u8=uint8_t; using u32=uint32_t; using u64=uint64_t; using i32=int32_t; using i8=int8_t; using i64=int64_t; using u16=uint16_t;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("CUDA %s:%d %s->%s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e));return 2;}}while(0)

// ---------------- host SHA256 (KAT-gated) + helpers (from lt_cpu_solver) ----------------
struct SHA256{u32 h[8];u8 buf[64];u64 len=0;size_t n=0;static u32 ror(u32 x,int r){return(x>>r)|(x<<(32-r));}
 SHA256(){h[0]=0x6a09e667;h[1]=0xbb67ae85;h[2]=0x3c6ef372;h[3]=0xa54ff53a;h[4]=0x510e527f;h[5]=0x9b05688c;h[6]=0x1f83d9ab;h[7]=0x5be0cd19;}
 void blk(const u8*p){static const u32 K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  u32 w[64];for(int i=0;i<16;i++)w[i]=(p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];for(int i=16;i<64;i++){u32 s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3),s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
  u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];for(int i=0;i<64;i++){u32 S1=ror(e,6)^ror(e,11)^ror(e,25),ch=(e&f)^((~e)&g),t1=hh+S1+ch+K[i]+w[i],S0=ror(a,2)^ror(a,13)^ror(a,22),mj=(a&b)^(a&c)^(b&c),t2=S0+mj;hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
  h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;}
 void write(const void*data,size_t l){const u8*p=(const u8*)data;len+=l;while(l){size_t t=64-n;if(t>l)t=l;memcpy(buf+n,p,t);n+=t;p+=t;l-=t;if(n==64){blk(buf);n=0;}}}
 void final(u8 o[32]){u64 b=len*8;u8 pad=0x80;write(&pad,1);u8 z=0;while(n!=56)write(&z,1);u8 lb[8];for(int i=0;i<8;i++)lb[i]=b>>(56-8*i);write(lb,8);for(int i=0;i<8;i++){o[i*4]=h[i]>>24;o[i*4+1]=h[i]>>16;o[i*4+2]=h[i]>>8;o[i*4+3]=h[i];}}};
static void sha256(const void*d,size_t l,u8 o[32]){SHA256 s;s.write(d,l);s.final(o);}
struct H32{u8 d[32];};
static H32 tagged(const char*tag,const H32&h){SHA256 s;s.write(tag,strlen(tag));s.write(h.d,32);H32 o;s.final(o.d);return o;}
static const bool M_ACC[16]={1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1};
static const i8 M_VAL[16]={0,0,1,0,2,3,4,6,0,0,-1,0,-2,-3,-4,-6};
static vector<i8> mant_stream(const H32&seed,size_t count){u8 sb[32];for(int i=0;i<32;i++)sb[i]=seed.d[31-i];vector<i8> out(count);size_t f=0;u64 blk=0;while(f<count){SHA256 s;s.write(sb,32);u8 dm=0x6D;s.write(&dm,1);u8 b8[8];for(int i=0;i<8;i++)b8[i]=blk>>(8*i);s.write(b8,8);u8 hs[32];s.final(hs);for(int i=0;i<32&&f<count;i++){u8 nb[2]={(u8)(hs[i]&0xF),(u8)((hs[i]>>4)&0xF)};for(u8 nib:nb){if(M_ACC[nib]){out[f++]=M_VAL[nib];if(f==count)break;}}}++blk;}return out;}
static H32 hdr_hash(u32 ver,const u8*prev,const u8*merk,u32 tm,u32 bits,u64 nc,u16 dim,const u8*sa,const u8*sb){SHA256 s;u8 t[8];t[0]=ver;t[1]=ver>>8;t[2]=ver>>16;t[3]=ver>>24;s.write(t,4);s.write(prev,32);s.write(merk,32);t[0]=tm;t[1]=tm>>8;t[2]=tm>>16;t[3]=tm>>24;s.write(t,4);t[0]=bits;t[1]=bits>>8;t[2]=bits>>16;t[3]=bits>>24;s.write(t,4);for(int k=0;k<8;k++)t[k]=nc>>(8*k);s.write(t,8);t[0]=dim;t[1]=dim>>8;s.write(t,2);s.write(sa,32);s.write(sb,32);H32 o;s.final(o.d);return o;}

// ---------------- device kernels ----------------
__device__ __forceinline__ u32 drotl(u32 x,int n){return (x<<n)|(x>>(32-n));}
__device__ void dchacha(const u32 key[8],u32 ctr,u32 n0,u32 n1,u32 n2,u8 out[64]){
    u32 s[16]={0x61707865,0x3320646e,0x79622d32,0x6b206574,key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7],ctr,n0,n1,n2};
    u32 x[16];
    #pragma unroll
    for(int i=0;i<16;i++)x[i]=s[i];
    auto QR=[&](int a,int b,int c,int d){x[a]+=x[b];x[d]=drotl(x[d]^x[a],16);x[c]+=x[d];x[b]=drotl(x[b]^x[c],12);x[a]+=x[b];x[d]=drotl(x[d]^x[a],8);x[c]+=x[d];x[b]=drotl(x[b]^x[c],7);};
    #pragma unroll
    for(int i=0;i<10;i++){QR(0,4,8,12);QR(1,5,9,13);QR(2,6,10,14);QR(3,7,11,15);QR(0,5,10,15);QR(1,6,11,12);QR(2,7,8,13);QR(3,4,9,14);}
    #pragma unroll
    for(int i=0;i<16;i++){u32 v=x[i]+s[i];out[i*4]=v;out[i*4+1]=v>>8;out[i*4+2]=v>>16;out[i*4+3]=v>>24;}
}
__constant__ bool dM_ACC[16]; __constant__ i8 dM_VAL[16];
__device__ u64 dprf(const u32 key[8],i32 raw,u32 i,u32 j,u32 remix,u32 lane){u32 n0=(u32)raw^lane;u64 ns=((u64)i<<32)|(u64)j;u8 ks[64];dchacha(key,remix,n0,(u32)(ns&0xffffffff),(u32)(ns>>32),ks);u64 v=0;for(int k=0;k<8;k++)v|=(u64)ks[k]<<(8*k);return v;}
// B32 = Y(n x w, int32) * H(w x n, int8) ; naive int32xint8 (kept for the byte-exact self-check)
__global__ void k_b32(const i32* Y,const i8* H,i32* B32,u32 n,u32 w){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n*n) return; u32 i=idx/n,j=idx%n; i64 acc=0;
    for(u32 k=0;k<w;k++) acc+=(i64)Y[i*w+k]*(i64)H[k*n+j]; B32[idx]=(i32)acc;
}
// tensor-core B32: split Y(int32, |Y|<2^18) into 3 STACKED balanced-base-256 int8 limb planes (3n x w),
// one INT8 GEMM (3n)xw . wxn, recombine B0+256*B1+65536*B2 (i64 then cast; exact since true B32 < 2^31).
__global__ void k_splitY3(const i32* Y,i8* Ys,u32 n,u32 w){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)n*w) return; u32 i=(u32)(idx/w),k=(u32)(idx%w); i32 x=Y[idx];
    #pragma unroll
    for(int l=0;l<2;l++){i32 d=((x+128)&255)-128; Ys[(size_t)(l*n+i)*w+k]=(i8)d; x=(x-d)/256;}
    Ys[(size_t)(2*n+i)*w+k]=(i8)x;
}
__global__ void k_recombine3(const i32* Bs,i32* B32,u32 n){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)n*n) return; u32 i=(u32)(idx/n),j=(u32)(idx%n);
    i64 v=(i64)Bs[(size_t)i*n+j] + 256LL*(i64)Bs[(size_t)(n+i)*n+j] + 65536LL*(i64)Bs[(size_t)(2*n+i)*n+j];
    B32[idx]=(i32)v;
}
__global__ void k_extract(const i32* B32,i8* Bhat,const u32* key,u32 n){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n*n) return; u32 i=idx/n,j=idx%n; i32 raw=B32[idx];
    for(u32 remix=0;;++remix){u64 mixed=dprf(key,raw,i,j,remix,0x4D414E54u);
        for(int sh=0;sh<64;sh+=4){u8 nib=(mixed>>sh)&0xF; if(dM_ACC[nib]){u64 sc=dprf(key,raw,i,j,remix,0x53434C45u);u8 e=sc&0x3;Bhat[idx]=(i8)((i32)dM_VAL[nib]*(1<<e));return;}}}
}
// direct mod q=2^61-1 combine: C[a*m+c] = sum_k P[a*n+k]*Q[k*m+c] mod q
__device__ __forceinline__ u64 dfqred(unsigned __int128 x){const u64 Q=(((u64)1)<<61)-1;u64 lo=(u64)(x&Q),hi=(u64)(x>>61);u64 s=lo+hi;s=(s&Q)+(s>>61);if(s>=Q)s-=Q;return s;}
__device__ __forceinline__ u64 dfq_i32(i32 x){const u64 Q=(((u64)1)<<61)-1;return x>=0?(u64)x:Q-(u64)(-(i64)x);}
__global__ void k_combine(const i32* P,const i32* Q,u64* C,u32 n,u32 m){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=m*m) return; u32 a=idx/m,c=idx%m; const u64 QP=(((u64)1)<<61)-1; u64 acc=0;
    for(u32 k=0;k<n;k++){i32 p=P[a*n+k];if(!p)continue;u64 pf=dfq_i32(p);u64 qf=dfq_i32(Q[k*m+c]);u64 pr=dfqred((unsigned __int128)pf*qf);acc+=pr;if(acc>=QP)acc-=QP;}
    C[idx]=acc;
}
// 3-base-128 TENSOR combine (byte-exact to mod-q; LT P,Q are 18-bit): split P/Q into 3 int8 limb planes
// (digits [-64,63]), one INT8 GEMM (3m)xn . nx(3m), reconstruct with weight 128^(i+j)=2^(7(i+j)) mod q.
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
 for(int i=0;i<LIMB3;i++)for(int j=0;j<LIMB3;j++){i32 g=Gbig[(size_t)(i*m+a)*M3+(size_t)(j*m+c)];acc=fqadd(acc,fqmul(((u64)1)<<(7*(i+j)),dfq_i32(g)));}Chat[(size_t)a*m+c]=acc;}

// cuBLASLt s8xs8->s32 row-major (L rows x inner, R inner x cols)
static bool gemm8(cublasLtHandle_t lt,void*ws,size_t wsz,const i8*dL,const i8*dR,i32*dO,u32 rows,u32 inner,u32 cols){
    cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;cublasLtMatmulPreference_t pf=nullptr;bool ok=false;
    do{if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I))break;cublasOperation_t N=CUBLAS_OP_N;cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&N,sizeof N);cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N,sizeof N);cublasLtOrder_t ro=CUBLASLT_ORDER_ROW;auto mk=[&](cublasLtMatrixLayout_t*L,cudaDataType t,u64 r,u64 c,i64 ld){if(cublasLtMatrixLayoutCreate(L,t,r,c,ld))return false;return 0==cublasLtMatrixLayoutSetAttribute(*L,CUBLASLT_MATRIX_LAYOUT_ORDER,&ro,sizeof ro);};if(!mk(&la,CUDA_R_8I,rows,inner,inner)||!mk(&lb,CUDA_R_8I,inner,cols,cols)||!mk(&lc,CUDA_R_32I,rows,cols,cols))break;cublasLtMatmulPreferenceCreate(&pf);cublasLtMatmulPreferenceSetAttribute(pf,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof wsz);cublasLtMatmulHeuristicResult_t h{};int got=0;if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pf,1,&h,&got)||!got)break;i32 al=1,be=0;if(cublasLtMatmul(lt,op,&al,dL,la,dR,lb,&be,dO,lc,dO,lc,&h.algo,ws,wsz,0))break;ok=true;}while(0);
    if(pf)cublasLtMatmulPreferenceDestroy(pf);if(lc)cublasLtMatrixLayoutDestroy(lc);if(lb)cublasLtMatrixLayoutDestroy(lb);if(la)cublasLtMatrixLayoutDestroy(la);if(op)cublasLtMatmulDescDestroy(op);return ok;
}

int main(){
    {u8 o[32];sha256("abc",3,o);if(o[0]!=0xba){printf("KAT fail\n");return 2;}}
    cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));printf("device: %s sm_%d%d\n",p.name,p.major,p.minor);
    const u32 n=64,m=n/2,w=128;
    u8 prev[32],merk[32],sa[32],sb[32],zero[32];memset(prev,0x51,32);memset(merk,0xa3,32);memset(sa,0x11,32);memset(sb,0x22,32);memset(zero,0,32);
    H32 hh=hdr_hash(0x20000004u,prev,merk,1770000000u,0x207fffffu,0xdeadbeefULL,64,sa,sb);
    H32 th=hdr_hash(0x20000004u,prev,merk,1770000000u,0x207fffffu,0,64,zero,zero);
    H32 sigma;sha256(hh.d,32,sigma.d);
    // host M11 projectors (proven)
    auto host_mat=[&](const H32&tmpl,const H32&seed_w,vector<i8>&G,vector<i8>&H,vector<i8>&W){
        G=mant_stream(tagged("BTX_MATEXPAND_G_V44LT",tmpl),(size_t)n*n);
        H=mant_stream(tagged("BTX_MATEXPAND_H_V44LT",tmpl),(size_t)w*n);
        W=mant_stream(seed_w,(size_t)n*w);};
    // --- device MatExpand: Y=G*W (cuBLASLt), B32=Y*H (naive), Extract (ChaCha) ---
    cublasLtHandle_t lt;cublasLtCreate(&lt);void*ws;CK(cudaMalloc(&ws,(size_t)32<<20));
    CK(cudaMemcpyToSymbol(dM_ACC,M_ACC,16));CK(cudaMemcpyToSymbol(dM_VAL,M_VAL,16));
    auto matexpand=[&](const H32&tmpl,const H32&seed_w)->vector<i8>{
        vector<i8> G,H,W;host_mat(tmpl,seed_w,G,H,W);
        i8*dG,*dW,*dH;i32*dY,*dB32;i8*dBhat;u32*dkey;
        i8*dYstack;i32*dBstack;   // tensor-B32: 3 stacked int8 limb planes of Y + stacked GEMM output
        cudaMalloc(&dG,(size_t)n*n);cudaMalloc(&dW,(size_t)n*w);cudaMalloc(&dH,(size_t)w*n);cudaMalloc(&dY,(size_t)n*w*4);cudaMalloc(&dB32,(size_t)n*n*4);cudaMalloc(&dBhat,(size_t)n*n);cudaMalloc(&dkey,32);
        cudaMalloc(&dYstack,(size_t)3*n*w);cudaMalloc(&dBstack,(size_t)3*n*n*4);
        cudaMemcpy(dG,G.data(),(size_t)n*n,cudaMemcpyHostToDevice);cudaMemcpy(dW,W.data(),(size_t)n*w,cudaMemcpyHostToDevice);cudaMemcpy(dH,H.data(),(size_t)w*n,cudaMemcpyHostToDevice);
        gemm8(lt,ws,(size_t)32<<20,dG,dW,dY,n,n,w);                    // Y = G*W  (n x w)
        // B32 = Y*H via 3-limb int8 tensor GEMM (byte-exact; end-to-end gated by db1136f2)
        k_splitY3<<<((size_t)n*w+255)/256,256>>>(dY,dYstack,n,w);
        gemm8(lt,ws,(size_t)32<<20,dYstack,dH,dBstack,3*n,w,n);
        k_recombine3<<<((size_t)n*n+255)/256,256>>>(dBstack,dB32,n);
        H32 prf;{SHA256 s;const char*t="BTX_MATEXPAND_PRF_V44LT";s.write(t,strlen(t));s.write(seed_w.d,32);s.final(prf.d);}
        u32 key[8];for(int i=0;i<8;i++)key[i]=prf.d[i*4]|(prf.d[i*4+1]<<8)|(prf.d[i*4+2]<<16)|((u32)prf.d[i*4+3]<<24);
        cudaMemcpy(dkey,key,32,cudaMemcpyHostToDevice);
        k_extract<<<(n*n+255)/256,256>>>(dB32,dBhat,dkey,n);
        vector<i8> Bhat((size_t)n*n);cudaMemcpy(Bhat.data(),dBhat,(size_t)n*n,cudaMemcpyDeviceToHost);
        cudaFree(dG);cudaFree(dW);cudaFree(dH);cudaFree(dY);cudaFree(dB32);cudaFree(dBhat);cudaFree(dkey);cudaFree(dYstack);cudaFree(dBstack);return Bhat;};
    vector<i8> Ahat=matexpand(th,tagged("BTX_MATEXPAND_WA_V44LT",th));
    vector<i8> Bhat=matexpand(th,tagged("BTX_MATEXPAND_W_V44LT",hh));
    // U,V + P=U*Ahat, Q=Bhat*V on device, combine on device
    vector<i8> U=mant_stream(tagged("BTX_MATMUL_V44LT_SKETCH_U",th),(size_t)m*n);
    vector<i8> V=mant_stream(tagged("BTX_MATMUL_V44LT_SKETCH_V",th),(size_t)n*m);
    i8*dU,*dV,*dAh,*dBh;i32*dP,*dQ;u64*dC;
    i8*dPlimb3,*dQlh3;i32*dGbig3;   // 3-base-128 tensor combine
    cudaMalloc(&dU,(size_t)m*n);cudaMalloc(&dV,(size_t)n*m);cudaMalloc(&dAh,(size_t)n*n);cudaMalloc(&dBh,(size_t)n*n);cudaMalloc(&dP,(size_t)m*n*4);cudaMalloc(&dQ,(size_t)n*m*4);cudaMalloc(&dC,(size_t)m*m*8);
    cudaMalloc(&dPlimb3,(size_t)LIMB3*m*n);cudaMalloc(&dQlh3,(size_t)n*LIMB3*m);cudaMalloc(&dGbig3,(size_t)LIMB3*m*LIMB3*m*4);
    cudaMemcpy(dU,U.data(),(size_t)m*n,cudaMemcpyHostToDevice);cudaMemcpy(dV,V.data(),(size_t)n*m,cudaMemcpyHostToDevice);
    cudaMemcpy(dAh,Ahat.data(),(size_t)n*n,cudaMemcpyHostToDevice);cudaMemcpy(dBh,Bhat.data(),(size_t)n*n,cudaMemcpyHostToDevice);
    gemm8(lt,ws,(size_t)32<<20,dU,dAh,dP,m,n,n);      // P = U*Ahat (m x n)
    gemm8(lt,ws,(size_t)32<<20,dBh,dV,dQ,n,n,m);      // Q = Bhat*V (n x m)
    // C = P*Q mod q via 3-base-128 tensor combine (byte-exact; gated end-to-end by db1136f2)
    k_split3<<<((size_t)m*n+255)/256,256>>>(dP,dPlimb3,(size_t)m*n);
    k_split_qh3<<<((size_t)n*m+255)/256,256>>>(dQ,dQlh3,n,m);
    gemm8(lt,ws,(size_t)32<<20,dPlimb3,dQlh3,dGbig3,LIMB3*m,n,LIMB3*m);
    { dim3 rbl(16,16),rgr((m+15)/16,(m+15)/16); k_reconstruct3<<<rgr,rbl>>>(dGbig3,dC,m); }
    CK(cudaDeviceSynchronize());
    vector<u64> Chat((size_t)m*m);cudaMemcpy(Chat.data(),dC,(size_t)m*m*8,cudaMemcpyDeviceToHost);
    // host digest = SHA256d("BTX_MATMUL_V4" || sigma || LE64(Chat))
    vector<u8> buf;const char*tag="BTX_MATMUL_V4";buf.insert(buf.end(),tag,tag+strlen(tag));buf.insert(buf.end(),sigma.d,sigma.d+32);
    for(u64 x:Chat)for(int k=0;k<8;k++)buf.push_back((u8)(x>>(8*k)));
    u8 d1[32],d2[32];sha256(buf.data(),buf.size(),d1);sha256(d1,32,d2);
    char hex[65];for(int k=0;k<32;k++)sprintf(hex+2*k,"%02x",d2[31-k]);
    const char*EXP="db1136f2974d45d9757262978ab074ef53ba54c368df9829f565ee2d26da0da9";
    printf("digest=%s\nEXPECT=%s\n%s\n",hex,EXP,strcmp(hex,EXP)==0?"=> LT GPU SOLVER BYTE-EXACT (db1136f2)":"=> MISMATCH");
    return strcmp(hex,EXP)==0?0:1;
}
