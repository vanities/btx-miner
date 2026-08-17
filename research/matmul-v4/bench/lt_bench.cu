// matador LT solver — sustained per-nonce COMPUTE bench at production n=4096, sm_120.
// Times the LT per-nonce GPU work: Y=G*W -> B32=Y*H -> ChaCha Extract -> Q=Bhat*V ->
// mod-q combine. Template-scoped P=U*Ahat done once. Digest (S4) EXCLUDED here (SHA,
// separate stage; noted). Operands filled with a realistic M11 pattern (values in
// {0,+-1..+-6}) so GEMM/zero-skip timing is representative. Reports nonce/s + TOPS.
// The point: does LT's big dense m=2048 GEMM (Q=Bhat*V, ~34 GMAC) favor datacenter?
// build: nvcc -O3 -arch=sm_120 lt_bench.cu -lcublasLt -lnvidia-ml -o lt_bench
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvml.h>
using u8=uint8_t; using i8=int8_t; using i32=int32_t; using u32=uint32_t; using u64=uint64_t; using i64=int64_t;
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){printf("CUDA %s->%s\n",#x,cudaGetErrorString(e));return 2;}}while(0)

__global__ void k_b32(const i32* Y,const i8* H,i32* B32,u32 n,u32 w){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n*n) return; u32 i=idx/n,j=idx%n; i64 acc=0;
    for(u32 k=0;k<w;k++) acc+=(i64)Y[i*w+k]*(i64)H[k*n+j]; B32[idx]=(i32)acc;
}
// int32-accumulate naive: B32 fits int32 (max |B32| = 147456*6*128 ~= 113M < 2^31), so the i64 above is waste
__global__ void k_b32_i32(const i32* Y,const i8* H,i32* B32,u32 n,u32 w){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n*n) return; u32 i=idx/n,j=idx%n; i32 acc=0;
    for(u32 k=0;k<w;k++) acc+=Y[i*w+k]*(i32)H[k*n+j]; B32[idx]=acc;
}
// split Y(n x w int32, |Y|<2^18) into 3 STACKED int8 limb planes (3n x w), balanced base-256 (digits in [-128,127])
__global__ void k_splitY3(const i32* Y,i8* Ys,u32 n,u32 w){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)n*w) return; u32 i=(u32)(idx/w),k=(u32)(idx%w); i32 x=Y[idx];
    #pragma unroll
    for(int l=0;l<2;l++){i32 d=((x+128)&255)-128; Ys[(size_t)(l*n+i)*w+k]=(i8)d; x=(x-d)/256;}
    Ys[(size_t)(2*n+i)*w+k]=(i8)x;
}
// recombine stacked GEMM output Bs(3n x n int32) -> B32 = B0 + 256*B1 + 65536*B2 (i64 then cast, exact since true B32<2^31)
__global__ void k_recombine3(const i32* Bs,i32* B32,u32 n){
    size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=(size_t)n*n) return; u32 i=(u32)(idx/n),j=(u32)(idx%n);
    i64 v=(i64)Bs[(size_t)i*n+j] + 256LL*(i64)Bs[(size_t)(n+i)*n+j] + 65536LL*(i64)Bs[(size_t)(2*n+i)*n+j];
    B32[idx]=(i32)v;
}
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
__constant__ i8 dM_VAL[16]; __constant__ bool dM_ACC[16];
__device__ u64 dprf(const u32 key[8],i32 raw,u32 i,u32 j,u32 rm,u32 lane){u32 n0=(u32)raw^lane;u64 ns=((u64)i<<32)|(u64)j;u8 ks[64];dchacha(key,rm,n0,(u32)(ns&0xffffffff),(u32)(ns>>32),ks);u64 v=0;for(int k=0;k<8;k++)v|=(u64)ks[k]<<(8*k);return v;}
__global__ void k_extract(const i32* B32,i8* Bhat,const u32* key,u32 n){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n*n) return; u32 i=idx/n,j=idx%n; i32 raw=B32[idx];
    for(u32 rm=0;;++rm){u64 mx=dprf(key,raw,i,j,rm,0x4D414E54u);for(int sh=0;sh<64;sh+=4){u8 nb=(mx>>sh)&0xF;if(dM_ACC[nb]){u64 sc=dprf(key,raw,i,j,rm,0x53434C45u);u8 e=sc&0x3;Bhat[idx]=(i8)((i32)dM_VAL[nb]*(1<<e));return;}}}
}
// --- PR#89 (695dd45) MX-block Extract PERF MODEL (NOT byte-exact): ~32x PRF dilution ---
// numair retired the per-element ChaCha rejection loop for a block-scale (MXFP4 E2M1) Extract:
// ONE PRF per 32-element block for the scale, per-element mantissa from the block tile (cheap, no
// ChaCha). Models ONLY the op-count/perf profile of his new Extract; byte-exact MX replication is
// a separate task (needs his exact block-scale math). This is the "did diluting the PRF un-strand
// the datacenter" measurement -- it shrinks the consumer-favoring ALU floor ~32x.
__global__ void k_mx_scales(const u32* key,u8* scales,u32 nblk){
    u32 b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nblk) return;
    u64 sc=dprf(key,(i32)b,b,0,0,0x4D58424Cu); scales[b]=(u8)(sc&0x3);   // 'MXBL' — 1 ChaCha / 32-block
}
__global__ void k_extract_mx(const i32* B32,const u8* scales,i8* Bhat,u32 n){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n*n) return; i32 raw=B32[idx];
    u8 e=scales[idx>>5]; u8 nb=(u8)((raw^(raw>>7))&0xF);                  // per-element mantissa, no ChaCha
    Bhat[idx]=(i8)((i32)dM_VAL[nb]*(1<<e));
}
__device__ __forceinline__ u64 dfqred(unsigned __int128 x){const u64 Q=(((u64)1)<<61)-1;u64 lo=(u64)(x&Q),hi=(u64)(x>>61);u64 s=lo+hi;s=(s&Q)+(s>>61);if(s>=Q)s-=Q;return s;}
__device__ __forceinline__ u64 dfq(i32 x){const u64 Q=(((u64)1)<<61)-1;return x>=0?(u64)x:Q-(u64)(-(i64)x);}
__global__ void k_combine(const i32* P,const i32* Q,u64* C,u32 n,u32 m){
    u32 idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=m*m) return; u32 a=idx/m,c=idx%m; const u64 QP=(((u64)1)<<61)-1;u64 acc=0;
    for(u32 k=0;k<n;k++){i32 p=P[a*n+k];if(!p)continue;u64 pr=dfqred((unsigned __int128)dfq(p)*dfq(Q[k*m+c]));acc+=pr;if(acc>=QP)acc-=QP;}
    C[idx]=acc;
}
// ---- 4-base-64 TENSOR combine (byte-exact to mod-q; from BMX4-C) ----
constexpr int LIMBS=4;
__device__ __forceinline__ u64 fqadd(u64 a,u64 b){const u64 Q=(((u64)1)<<61)-1;u64 s=a+b;if(s>=Q)s-=Q;return s;}
__device__ __forceinline__ u64 fqmul(u64 a,u64 b){return dfqred((unsigned __int128)a*b);}
__global__ void k_split(const i32* X,i8* planes,size_t count){size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(idx>=count)return;i32 x=X[idx];
 #pragma unroll
 for(int l=0;l<LIMBS-1;l++){i32 d=((x+32)&63)-32;planes[(size_t)l*count+idx]=(i8)d;x=(x-d)/64;}planes[(size_t)(LIMBS-1)*count+idx]=(i8)x;}
__global__ void k_split_qh(const i32* X,i8* qlh,u32 n,u32 mcol){size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;size_t count=(size_t)n*mcol;if(idx>=count)return;u32 r=(u32)(idx/mcol),c=(u32)(idx%mcol);i32 x=X[idx];
 #pragma unroll
 for(int l=0;l<LIMBS-1;l++){i32 d=((x+32)&63)-32;qlh[(size_t)r*(LIMBS*mcol)+(size_t)l*mcol+c]=(i8)d;x=(x-d)/64;}qlh[(size_t)r*(LIMBS*mcol)+(size_t)(LIMBS-1)*mcol+c]=(i8)x;}
// --- 3-base-128 combine: LT P,Q are 18-bit (|P|=|U.Ahat|<=6*6*4096=147456<2^20), so 3 base-128 limbs
//     (digits [-64,63], weight 128^(i+j)=2^(7(i+j))) cover the SAME range as 4 base-64 with a 0.56x combine GEMM ---
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
__global__ void k_reconstruct(const i32* Gbig,u64* Chat,u32 m){u32 a=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(a>=m||c>=m)return;const size_t M4=(size_t)LIMBS*m;u64 acc=0;
 #pragma unroll
 for(int i=0;i<LIMBS;i++)for(int j=0;j<LIMBS;j++){i32 g=Gbig[(size_t)(i*m+a)*M4+(size_t)(j*m+c)];acc=fqadd(acc,fqmul(((u64)1)<<(6*(i+j)),dfq(g)));}Chat[(size_t)a*m+c]=acc;}
static bool gemm8(cublasLtHandle_t lt,void*ws,size_t wsz,const i8*dL,const i8*dR,i32*dO,u32 rows,u32 inner,u32 cols){
    cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t la=nullptr,lb=nullptr,lc=nullptr;cublasLtMatmulPreference_t pf=nullptr;bool ok=false;
    do{if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I))break;cublasOperation_t N=CUBLAS_OP_N;cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&N,sizeof N);cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&N,sizeof N);cublasLtOrder_t ro=CUBLASLT_ORDER_ROW;auto mk=[&](cublasLtMatrixLayout_t*L,cudaDataType t,u64 r,u64 c,i64 ld){if(cublasLtMatrixLayoutCreate(L,t,r,c,ld))return false;return 0==cublasLtMatrixLayoutSetAttribute(*L,CUBLASLT_MATRIX_LAYOUT_ORDER,&ro,sizeof ro);};if(!mk(&la,CUDA_R_8I,rows,inner,inner)||!mk(&lb,CUDA_R_8I,inner,cols,cols)||!mk(&lc,CUDA_R_32I,rows,cols,cols))break;cublasLtMatmulPreferenceCreate(&pf);cublasLtMatmulPreferenceSetAttribute(pf,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof wsz);cublasLtMatmulHeuristicResult_t h{};int got=0;if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,lc,lc,pf,1,&h,&got)||!got)break;i32 al=1,be=0;if(cublasLtMatmul(lt,op,&al,dL,la,dR,lb,&be,dO,lc,dO,lc,&h.algo,ws,wsz,0))break;ok=true;}while(0);
    if(pf)cublasLtMatmulPreferenceDestroy(pf);if(lc)cublasLtMatrixLayoutDestroy(lc);if(lb)cublasLtMatrixLayoutDestroy(lb);if(la)cublasLtMatrixLayoutDestroy(la);if(op)cublasLtMatmulDescDestroy(op);return ok;
}

int main(int argc,char**argv){
    const u32 n=argc>1?atoi(argv[1]):4096, m=n/2, w=1024;   // PR#89 770031e: MatExpand panel w 128->1024
    const double secs=argc>2?atof(argv[2]):30.0;
    cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d | LT n=%u m=%u (deep b=2) w=%u | target %.0fs\n",p.name,p.major,p.minor,n,m,w,secs);
    nvmlInit(); nvmlDevice_t nv; nvmlDeviceGetHandleByIndex(0,&nv);
    static const i8 MV[16]={0,0,1,0,2,3,4,6,0,0,-1,0,-2,-3,-4,-6}; static const bool MA[16]={1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1};
    CK(cudaMemcpyToSymbol(dM_VAL,MV,16)); CK(cudaMemcpyToSymbol(dM_ACC,MA,16));
    // realistic M11-pattern operand fill (values in {0,+-1..+-6}, ~15% zeros like M11)
    auto fill=[&](std::vector<i8>&v,size_t cnt,u32 salt){v.resize(cnt);for(size_t k=0;k<cnt;k++){u32 h=(u32)(k*2654435761u+salt*40503u);int nib=(h>>13)&0xF;v[k]=MA[nib]?MV[nib]:0;}};
    std::vector<i8> G,H,W,Ah,U,V; fill(G,(size_t)n*n,1);fill(H,(size_t)w*n,2);fill(W,(size_t)n*w,3);fill(Ah,(size_t)n*n,4);fill(U,(size_t)m*n,5);fill(V,(size_t)n*m,6);
    i8*dG,*dW,*dH,*dAh,*dU,*dV,*dBh;i32*dY,*dB32,*dP,*dQ;u64*dC;u32*dkey;
    i8*dPlimb,*dQlh;i32*dGbig;
    i8*dYstack;i32*dBstack;   // tensor-B32: 3 stacked int8 limb planes of Y + stacked GEMM output
    CK(cudaMalloc(&dYstack,(size_t)3*n*w));CK(cudaMalloc(&dBstack,(size_t)3*n*n*4));
    i8*dPlimb3,*dQlh3;i32*dGbig3;   // 3-base-128 combine (0.56x GEMM vs 4-base-64)
    CK(cudaMalloc(&dPlimb3,(size_t)LIMB3*m*n));CK(cudaMalloc(&dQlh3,(size_t)n*LIMB3*m));CK(cudaMalloc(&dGbig3,(size_t)LIMB3*m*LIMB3*m*4));
    CK(cudaMalloc(&dPlimb,(size_t)LIMBS*m*n));CK(cudaMalloc(&dQlh,(size_t)n*LIMBS*m));CK(cudaMalloc(&dGbig,(size_t)LIMBS*m*LIMBS*m*4));
    CK(cudaMalloc(&dG,(size_t)n*n));CK(cudaMalloc(&dW,(size_t)n*w));CK(cudaMalloc(&dH,(size_t)w*n));CK(cudaMalloc(&dAh,(size_t)n*n));CK(cudaMalloc(&dU,(size_t)m*n));CK(cudaMalloc(&dV,(size_t)n*m));CK(cudaMalloc(&dBh,(size_t)n*n));
    CK(cudaMalloc(&dY,(size_t)n*w*4));CK(cudaMalloc(&dB32,(size_t)n*n*4));CK(cudaMalloc(&dP,(size_t)m*n*4));CK(cudaMalloc(&dQ,(size_t)n*m*4));CK(cudaMalloc(&dC,(size_t)m*m*8));CK(cudaMalloc(&dkey,32));
    cudaMemcpy(dG,G.data(),(size_t)n*n,cudaMemcpyHostToDevice);cudaMemcpy(dW,W.data(),(size_t)n*w,cudaMemcpyHostToDevice);cudaMemcpy(dH,H.data(),(size_t)w*n,cudaMemcpyHostToDevice);cudaMemcpy(dAh,Ah.data(),(size_t)n*n,cudaMemcpyHostToDevice);cudaMemcpy(dU,U.data(),(size_t)m*n,cudaMemcpyHostToDevice);cudaMemcpy(dV,V.data(),(size_t)n*m,cudaMemcpyHostToDevice);
    u8* dScales; CK(cudaMalloc(&dScales,(size_t)((n*n+31)/32)));   // MX-block scales (1 per 32 elems)
    u32 hkey[8]={1,2,3,4,5,6,7,8};cudaMemcpy(dkey,hkey,32,cudaMemcpyHostToDevice);
    cublasLtHandle_t lt;cublasLtCreate(&lt);void*ws;CK(cudaMalloc(&ws,(size_t)256<<20));size_t WS=(size_t)256<<20;
    gemm8(lt,ws,WS,dU,dAh,dP,m,n,n);   // template P=U*Ahat (once)
    dim3 rbl(16,16),rgr((m+15)/16,(m+15)/16);
    auto tcombine=[&](){ k_split<<<((size_t)m*n+255)/256,256>>>(dP,dPlimb,(size_t)m*n); k_split_qh<<<((size_t)n*m+255)/256,256>>>(dQ,dQlh,n,m); gemm8(lt,ws,WS,dPlimb,dQlh,dGbig,LIMBS*m,n,LIMBS*m); k_reconstruct<<<rgr,rbl>>>(dGbig,dC,m); };
    auto tcombine3=[&](){ k_split3<<<((size_t)m*n+255)/256,256>>>(dP,dPlimb3,(size_t)m*n); k_split_qh3<<<((size_t)n*m+255)/256,256>>>(dQ,dQlh3,n,m); gemm8(lt,ws,WS,dPlimb3,dQlh3,dGbig3,LIMB3*m,n,LIMB3*m); k_reconstruct3<<<rgr,rbl>>>(dGbig3,dC,m); };
    // tensor-B32: split Y -> 3 stacked int8 limbs, ONE big INT8 GEMM (3n)xw . wxn, recombine (byte-exact to naive B32)
    auto b32tensor=[&](){ k_splitY3<<<((size_t)n*w+255)/256,256>>>(dY,dYstack,n,w); gemm8(lt,ws,WS,dYstack,dH,dBstack,3*n,w,n); k_recombine3<<<((size_t)n*n+255)/256,256>>>(dBstack,dB32,n); };
    // MX-block Extract (PR#89 perf model): 1 PRF per 32-block scale + cheap per-element mantissa
    auto extract_mx=[&](){ u32 nb=(n*n+31)/32; k_mx_scales<<<(nb+255)/256,256>>>(dkey,dScales,nb); k_extract_mx<<<(n*n+255)/256,256>>>(dB32,dScales,dBh,n); };
    auto nonce=[&](){ gemm8(lt,ws,WS,dG,dW,dY,n,n,w); b32tensor(); extract_mx(); gemm8(lt,ws,WS,dBh,dV,dQ,n,n,m); tcombine3(); };
    nonce(); CK(cudaDeviceSynchronize());   // warmup (dY now valid)
    // self-check: tensor B32 must byte-match the naive B32 (guards the perf claim before we trust it)
    { std::vector<i32> hn((size_t)n*n),ht((size_t)n*n);
      k_b32<<<(n*n+255)/256,256>>>(dY,dH,dB32,n,w); CK(cudaMemcpy(hn.data(),dB32,(size_t)n*n*4,cudaMemcpyDeviceToHost));
      b32tensor(); CK(cudaMemcpy(ht.data(),dB32,(size_t)n*n*4,cudaMemcpyDeviceToHost));
      size_t mism=0; for(size_t z=0;z<(size_t)n*n;z++) if(hn[z]!=ht[z]) mism++;
      printf("  B32 tensor-vs-naive: %s (%zu mismatches)\n", mism?"MISMATCH":"byte-exact", mism);
      if(mism){ printf("  ABORT: tensor B32 not byte-exact\n"); return 1; } }
    // self-check: 3-base-128 combine must byte-match the 4-base-64 combine (dP/dQ valid post-warmup)
    { std::vector<u64> c4((size_t)m*m),c3((size_t)m*m);
      tcombine();  CK(cudaMemcpy(c4.data(),dC,(size_t)m*m*8,cudaMemcpyDeviceToHost));
      tcombine3(); CK(cudaMemcpy(c3.data(),dC,(size_t)m*m*8,cudaMemcpyDeviceToHost));
      size_t mism=0; for(size_t z=0;z<(size_t)m*m;z++) if(c4[z]!=c3[z]) mism++;
      printf("  combine 3-base-128-vs-4-base-64: %s (%zu mismatches)\n", mism?"MISMATCH":"byte-exact", mism);
      if(mism){ printf("  ABORT: 3-limb combine not byte-exact\n"); return 1; } }
    // ---- per-stage breakdown (20 iters each) ----
    { cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); const int R=20;
      auto tt=[&](const char* nm, auto fn){ cudaDeviceSynchronize(); cudaEventRecord(a); for(int r=0;r<R;r++) fn(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms,a,b); printf("  stage %-10s %7.3f ms/nonce\n", nm, ms/R); };
      tt("Y=G*W",   [&]{ gemm8(lt,ws,WS,dG,dW,dY,n,n,w); });
      tt("B32-i64",  [&]{ k_b32<<<(n*n+255)/256,256>>>(dY,dH,dB32,n,w); });
      tt("B32-i32",  [&]{ k_b32_i32<<<(n*n+255)/256,256>>>(dY,dH,dB32,n,w); });
      tt("B32-tens", [&]{ b32tensor(); });
      tt("Extract-cc", [&]{ k_extract<<<(n*n+255)/256,256>>>(dB32,dBh,dkey,n); });   // legacy ChaCha (retired PR#89)
      tt("Extract-mx", [&]{ extract_mx(); });   // PR#89 MX-block model: ~32x PRF dilution
      tt("Q=Bhat*V",[&]{ gemm8(lt,ws,WS,dBh,dV,dQ,n,n,m); });
      tt("combineTC",[&]{ tcombine(); });   // 4-base-64 tensor combine (split->big INT8 GEMM->reconstruct)
      tt("combineTC3",[&]{ tcombine3(); }); // 3-base-128 tensor combine (0.56x GEMM, byte-exact)
    }
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
    long cnt=0; double pw=0; int pn=0; cudaEventRecord(t0);
    double el=0; while(el<secs){ for(int b=0;b<4;b++){nonce();++cnt;} CK(cudaDeviceSynchronize()); cudaEventRecord(t1);cudaEventSynchronize(t1);float ms;cudaEventElapsedTime(&ms,t0,t1);el=ms/1000.0; unsigned mw;if(nvmlDeviceGetPowerUsage(nv,&mw)==NVML_SUCCESS){pw+=mw;pn++;} }
    double rate=cnt/el; double gmac=(double)m*n*n + (double)n*n*m + (double)n*n*w + (double)n*w*n; // Q+? approx per-nonce INT8 MACs (Y,B32,Q)
    printf("LT compute (digest EXCL): %ld nonces in %.1fs => %.1f nonce/s  |  ~%.0f W  |  Q-GEMM %ux%ux%u\n",
           cnt, el, rate, pn?pw/pn/1000.0:0.0, n,m,n);
    printf("  note: per-nonce dense INT8 = Y(%ux%ux%u)+B32(naive)+Q(%ux%ux%u); combine mod-q ALU; S4 digest excluded\n", n,w,n>0?n:0, n,n,m);
    nvmlShutdown(); return 0;
}
