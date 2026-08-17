// Standalone Blackwell FP4 (mxf4) accumulator-exactness probe — the on-silicon
// M-t24 measurement that BTX PR #89's own tool does NOT wire for the GPU
// (matmul-v4-report reports "device native kernel wired: no"). Mirrors their
// backend's RunMxf4Qualification (src/cuda/matmul_v4_bmx4_accel.cu) exactly.
//
// Question: does this device's block-scaled FP4 tensor core accumulate EXACTLY
// to t=24 bits, or round earlier (Hopper-style t~14)? If exact -> BMX4-C native
// FP4 path is eligible on this card. If not -> fail-closed to INT8.
//
// PROBE 1 (t-discrimination): all-(+3) rail GEMM, product=9 everywhere, K=
// floor(2^24/9) 32-aligned. Accumulator walks 9,18,27,... to EXACTLY 9K =
// 2^24-64 = 16,777,152, hitting odd partial sums that a t<24 grid cannot hold.
// PROBE 2 (layout/packing): fixed pseudorandom M11 GEMM vs exact int64 host ref.
//
// build (pc, CUDA 13.3):  nvcc -O3 -arch=sm_120 fp4_t24.cu -lcublasLt -o fp4_t24
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

#if !defined(CUBLAS_VERSION) || CUBLAS_VERSION < 120800
int main(){ printf("FP4 t24: cuBLASLt < 12.8 has no block-scaled FP4 API (CUBLAS_VERSION=%d). Need CUDA>=12.8.\n",
                   (int)CUBLAS_VERSION); return 2; }
#else

// M11 -> pinned E2M1 nibble (matmul_v4_bmx4_accel.cu EncodeE2M1Nibble)
static uint8_t Enc(int8_t mu){switch(mu){case 0:return 0x0;case 1:return 0x2;case -1:return 0xA;case 2:return 0x4;
  case -2:return 0xC;case 3:return 0x5;case -3:return 0xD;case 4:return 0x6;case -4:return 0xE;case 6:return 0x7;
  case -6:return 0xF;default:return 0xFF;}}
static inline void Pack(uint8_t* buf,size_t idx,uint8_t nib){ if(idx&1) buf[idx>>1]|=(nib<<4); else buf[idx>>1]|=nib; }
static size_t UnitScaleBytes(size_t outer,size_t K){ size_t rows=((outer+127)/128)*128, kb=(K+31)/32, cols=((kb+3)/4)*4; return rows*cols; }

// One block-scaled FP4 GEMM: D(FP32, MxN col-major) = A(MxK)*B(KxN), TN, UE8M0
// unit scales, COMPUTE_32F, alpha=1/beta=0, FAST_ACCUM never set. (their RunMxf4Gemm)
static bool Mxf4Gemm(cublasLtHandle_t lt,cudaStream_t s,void* ws,size_t wsz,
                     const void* dA,const void* dB,const void* dSFa,const void* dSFb,float* dD,
                     uint32_t M,uint32_t N,uint32_t K,std::string& err,int fastaccum=0){
  cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t la=nullptr,lb=nullptr,ld=nullptr; cublasLtMatmulPreference_t pref=nullptr; bool ok=false;
  do{
    if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)){err="DescCreate";break;}
    cublasOperation_t opt=CUBLAS_OP_T, opn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opt,sizeof(opt));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof(opn));
    int8_t fa=(int8_t)fastaccum; cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fa,sizeof(fa));
    cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dSFa,sizeof(dSFa))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dSFb,sizeof(dSFb))){err="scaleattr";break;}
    if(cublasLtMatrixLayoutCreate(&la,CUDA_R_4F_E2M1,K,M,K)||cublasLtMatrixLayoutCreate(&lb,CUDA_R_4F_E2M1,K,N,K)||
       cublasLtMatrixLayoutCreate(&ld,CUDA_R_32F,M,N,M)){err="layout";break;}
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
    cublasLtMatmulHeuristicResult_t h{}; int got=0;
    if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,ld,ld,pref,1,&h,&got)||got==0){err="no mxf4 algorithm (device/toolkit lacks full-precision-accum block-scaled FP4)";break;}
    float alpha=1.f,beta=0.f;
    if(cublasLtMatmul(lt,op,&alpha,dA,la,dB,lb,&beta,dD,ld,dD,ld,&h.algo,ws,wsz,s)){err="matmul failed";break;}
    ok=true;
  }while(0);
  if(pref)cublasLtMatmulPreferenceDestroy(pref); if(ld)cublasLtMatrixLayoutDestroy(ld);
  if(lb)cublasLtMatrixLayoutDestroy(lb); if(la)cublasLtMatrixLayoutDestroy(la); if(op)cublasLtMatmulDescDestroy(op);
  return ok;
}

// Control: does the SAME cuBLASLt heuristic path find an INT8 tensor GEMM? If
// yes, the harness is sound and "FP4 none" is a genuine FP4-specific device gap.
static bool Int8Avail(cublasLtHandle_t lt,size_t wsz,uint32_t M,uint32_t N,uint32_t K){
  cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t la=nullptr,lb=nullptr,ld=nullptr; cublasLtMatmulPreference_t pref=nullptr; bool ok=false;
  if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32I,CUDA_R_32I)) return false;
  cublasOperation_t opt=CUBLAS_OP_T,opn=CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opt,sizeof(opt));
  cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof(opn));
  cublasLtMatrixLayoutCreate(&la,CUDA_R_8I,K,M,K); cublasLtMatrixLayoutCreate(&lb,CUDA_R_8I,K,N,K); cublasLtMatrixLayoutCreate(&ld,CUDA_R_32I,M,N,M);
  cublasLtMatmulPreferenceCreate(&pref); cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
  cublasLtMatmulHeuristicResult_t h{}; int got=0;
  if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,ld,ld,pref,1,&h,&got)==0 && got>0) ok=true;
  cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(ld); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(la); cublasLtMatmulDescDestroy(op);
  return ok;
}

// Does cuBLASLt offer NVFP4 (E2M1 + VEC16_UE4M3 16-elt scales) instead of MXFP4?
// cuBLASLt's FP4 GEMM (sample LtNvfp4Matmul) is NVFP4; BMX4-C needs MXFP4 (UE8M0).
// This tests whether the FP4 *engine* exists but only for the other scale format.
static bool Nvfp4Avail(cublasLtHandle_t lt,void* ws,size_t wsz,uint32_t M,uint32_t N,uint32_t K){
  cublasLtMatmulDesc_t op=nullptr; cublasLtMatrixLayout_t la=nullptr,lb=nullptr,ld=nullptr; cublasLtMatmulPreference_t pref=nullptr; bool ok=false;
  uint8_t *dA=nullptr,*dB=nullptr,*dSF=nullptr; float* dD=nullptr;
  if(cudaMalloc(&dA,(size_t)M*K/2)||cudaMalloc(&dB,(size_t)N*K/2)||cudaMalloc(&dD,(size_t)M*N*4)) return false;
  size_t kb=(K+15)/16, outer=(M>N?M:N); size_t sb=(((outer+127)/128)*128)*(((kb+3)/4)*4); cudaMalloc(&dSF, sb?sb:256);
  do{
    if(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F)) break;
    cublasOperation_t opt=CUBLAS_OP_T,opn=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&opt,sizeof(opt));
    cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&opn,sizeof(opn));
    cublasLtMatmulMatrixScale_t sm=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    if(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&sm,sizeof(sm))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&dSF,sizeof(dSF))||
       cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&dSF,sizeof(dSF))) break;
    if(cublasLtMatrixLayoutCreate(&la,CUDA_R_4F_E2M1,K,M,K)||cublasLtMatrixLayoutCreate(&lb,CUDA_R_4F_E2M1,K,N,K)||cublasLtMatrixLayoutCreate(&ld,CUDA_R_32F,M,N,M)) break;
    cublasLtMatmulPreferenceCreate(&pref); cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz));
    cublasLtMatmulHeuristicResult_t h{}; int got=0;
    if(cublasLtMatmulAlgoGetHeuristic(lt,op,la,lb,ld,ld,pref,1,&h,&got)==0 && got>0) ok=true;
  }while(0);
  if(pref)cublasLtMatmulPreferenceDestroy(pref); if(ld)cublasLtMatrixLayoutDestroy(ld); if(lb)cublasLtMatrixLayoutDestroy(lb); if(la)cublasLtMatrixLayoutDestroy(la); if(op)cublasLtMatmulDescDestroy(op);
  cudaFree(dA);cudaFree(dB);cudaFree(dD);cudaFree(dSF);
  return ok;
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("=== Blackwell FP4 (mxf4) t=24 accumulator-exactness probe ===\n");
  printf("device: %s  sm_%d%d  cuBLASLt %d\n", p.name, p.major, p.minor, (int)CUBLAS_VERSION);
  cublasLtHandle_t lt; if(cublasLtCreate(&lt)){printf("cublasLtCreate failed\n");return 2;}
  cudaStream_t s; CK(cudaStreamCreate(&s));
  size_t wsz=(size_t)64<<20; void* ws; CK(cudaMalloc(&ws,wsz));

  // ---- CONTROL: prove the harness + cuBLASLt path are sound via INT8 ----
  printf("\n[control] INT8 tensor GEMM (CUDA_R_8I, COMPUTE_32I) 512x512x4096: %s\n",
         Int8Avail(lt,wsz,512,512,4096)?"AVAILABLE (harness sound; 5090 has an INT8 tensor path -> the FP4 result below is FP4-specific)":"NONE (harness/driver problem -- FP4 result not trustworthy)");

  // ---- AVAILABILITY MATRIX: is block-scaled FP4 offered at all, and only the
  //      rounding (fast_accum) kind or also full-precision (fast_accum=0)? ----
  printf("\n[availability] cuBLASLt block-scaled FP4 (CUDA_R_4F_E2M1 + UE8M0) algorithm search:\n");
  struct Sh{uint32_t M,N,K;}; Sh shs[]={{512,512,4096},{1024,1024,8192},{256,256,2048},{32,32,1864128}};
  bool any_exact=false;
  for(auto sh:shs){
    for(int fa=0;fa<=1;fa++){
      uint8_t *tA,*tB,*tSF; float* tD;
      if(cudaMalloc(&tA,(size_t)sh.M*sh.K/2)||cudaMalloc(&tB,(size_t)sh.N*sh.K/2)||cudaMalloc(&tD,(size_t)sh.M*sh.N*4)) continue;
      size_t sb=UnitScaleBytes(sh.M>sh.N?sh.M:sh.N,sh.K); cudaMalloc(&tSF,sb);
      cudaMemset(tA,0x55,(size_t)sh.M*sh.K/2); cudaMemset(tB,0x55,(size_t)sh.N*sh.K/2); cudaMemset(tSF,0x7F,sb);
      std::string e; bool av=Mxf4Gemm(lt,s,ws,wsz,tA,tB,tSF,tSF,tD,sh.M,sh.N,sh.K,e,fa);
      if(av && fa==0) any_exact=true;
      printf("  %5ux%5ux%-8u  %-16s : %s%s\n", sh.M,sh.N,sh.K, fa?"fast_accum(round)":"full-precision", av?"AVAILABLE":"none", av?"":("  ["+e+"]").c_str());
      cudaFree(tA);cudaFree(tB);cudaFree(tD);cudaFree(tSF);
    }
  }
  printf("  => full-precision (exact) block-scaled MXFP4 (UE8M0) on this card: %s\n", any_exact?"AVAILABLE (exactness test below is meaningful)":"NONE (BMX4-C's MXFP4 recipe cannot dispatch -> falls to INT8)");
  // Is the FP4 engine present but only for NVFP4 (UE4M3 16-blk), not BMX4-C's MXFP4 (UE8M0 32-blk)?
  bool nv=Nvfp4Avail(lt,ws,wsz,512,512,4096);
  printf("  [contrast] NVFP4 (E2M1 + VEC16_UE4M3) 512x512x4096: %s\n",
         nv?"AVAILABLE -> cuBLASLt HAS an FP4 GEMM, but only NVFP4; BMX4-C's MXFP4 (UE8M0) is UNSERVED"
           :"none (no FP4 GEMM of either scale format here)");

  // ---- PROBE 1: all-(+3) rail (nibble 0x5, packing-independent memset 0x55) ----
  const uint32_t M=32,N=32; const uint32_t K=1864128; // 32-aligned floor(2^24/9)
  const double expect = 9.0*(double)K; // 16,777,152 = 2^24 - 64
  uint8_t *dA,*dB,*dSF; float* dD;
  CK(cudaMalloc(&dA,(size_t)M*K/2)); CK(cudaMalloc(&dB,(size_t)N*K/2)); CK(cudaMalloc(&dD,(size_t)M*N*4));
  size_t sfb=UnitScaleBytes(128,K); CK(cudaMalloc(&dSF,sfb));
  CK(cudaMemset(dA,0x55,(size_t)M*K/2)); CK(cudaMemset(dB,0x55,(size_t)N*K/2)); CK(cudaMemset(dSF,0x7F,sfb)); // 0x7F=E8M0 2^0
  std::string err;
  if(!Mxf4Gemm(lt,s,ws,wsz,dA,dB,dSF,dSF,dD,M,N,K,err)){ printf("PROBE1: mxf4 GEMM unavailable: %s\n  => native FP4 path NOT usable on this card via cuBLASLt.\n",err.c_str()); return 1; }
  CK(cudaStreamSynchronize(s));
  std::vector<float> d((size_t)M*N); CK(cudaMemcpy(d.data(),dD,d.size()*4,cudaMemcpyDeviceToHost));
  double got=d[0]; bool allsame=true; for(float v:d) if((double)v!=got) allsame=false;
  bool p1 = ((double)d[0]==expect); for(float v:d) if((double)v!=expect) p1=false;
  printf("\n[PROBE 1] t=24 discrimination (all +3 rail, K=%u)\n", K);
  printf("  expected exact 2^24-64 = %.0f\n", expect);
  printf("  FP4 tensor-core returned = %.1f  (uniform across outputs: %s)\n", got, allsame?"yes":"NO");
  printf("  delta from exact = %.1f\n", got-expect);
  if(p1) printf("  => PASS: accumulator is EXACT to t=24. Native FP4 path ELIGIBLE.\n");
  else {
    double ratio = got>0? got/expect : 0;
    printf("  => FAIL: FP4 accumulator ROUNDS below 2^24 (t<24). Native path INELIGIBLE (fail-closed to INT8).\n");
    printf("     (returned/expected = %.6f; the rounding regime the committed path forbids)\n", ratio);
  }

  // ---- PROBE 2: fixed pseudorandom M11 (M=N=64,K=4096) vs exact int64 ----
  const uint32_t M2=64,N2=64,K2=4096,RAIL=512;
  const int8_t A11[11]={0,1,-1,2,-2,3,-3,4,-4,6,-6};
  std::vector<int8_t> a2((size_t)M2*K2), b2((size_t)K2*N2);
  uint64_t st=0x9e3779b97f4a7c15ull; auto rnd=[&](){st^=st<<13;st^=st>>7;st^=st<<17;return st;};
  for(uint32_t r=0;r<M2;r++)for(uint32_t k=0;k<K2;k++) a2[(size_t)r*K2+k]=(k<RAIL)?6:A11[rnd()%11];
  for(uint32_t k=0;k<K2;k++)for(uint32_t c=0;c<N2;c++) b2[(size_t)k*N2+c]=(k<RAIL)?6:A11[rnd()%11];
  // pack A (KxM col-major, i.e. a2 is MxK row-major -> element (m,k) at k-major col-major = k + m*K)
  std::vector<uint8_t> pa((size_t)M2*K2/2,0), pb((size_t)K2*N2/2,0);
  for(uint32_t m=0;m<M2;m++)for(uint32_t k=0;k<K2;k++){ uint8_t nb=Enc(a2[(size_t)m*K2+k]); Pack(pa.data(),(size_t)m*K2+k,nb);} // col m, row k: idx=k+m*K
  for(uint32_t c=0;c<N2;c++)for(uint32_t k=0;k<K2;k++){ uint8_t nb=Enc(b2[(size_t)k*N2+c]); Pack(pb.data(),(size_t)c*K2+k,nb);} // col c, row k
  uint8_t *dA2,*dB2,*dSF2; float* dD2;
  CK(cudaMalloc(&dA2,pa.size())); CK(cudaMalloc(&dB2,pb.size())); CK(cudaMalloc(&dD2,(size_t)M2*N2*4));
  size_t sfb2=UnitScaleBytes(128,K2); CK(cudaMalloc(&dSF2,sfb2));
  CK(cudaMemcpy(dA2,pa.data(),pa.size(),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB2,pb.data(),pb.size(),cudaMemcpyHostToDevice));
  CK(cudaMemset(dSF2,0x7F,sfb2));
  printf("\n[PROBE 2] layout/packing cross-check (M11 pseudorandom, K=%u) vs exact int64\n", K2);
  if(!Mxf4Gemm(lt,s,ws,wsz,dA2,dB2,dSF2,dSF2,dD2,M2,N2,K2,err)){ printf("  mxf4 GEMM unavailable: %s\n", err.c_str()); }
  else{
    CK(cudaStreamSynchronize(s));
    std::vector<float> d2((size_t)M2*N2); CK(cudaMemcpy(d2.data(),dD2,d2.size()*4,cudaMemcpyDeviceToHost));
    size_t mism=0; int64_t maxabs=0;
    for(uint32_t m=0;m<M2;m++)for(uint32_t c=0;c<N2;c++){
      int64_t acc=0; for(uint32_t k=0;k<K2;k++) acc+=(int64_t)a2[(size_t)m*K2+k]*(int64_t)b2[(size_t)k*N2+c];
      if(acc<0?-acc:acc>maxabs) maxabs=acc<0?-acc:acc;
      // D is col-major MxN: entry (m,c) at m + c*M
      if((double)d2[(size_t)c*M2+m]!=(double)acc) mism++;
    }
    printf("  mismatches vs exact int64 = %zu / %u  (max |true dot| = %lld)  %s\n",
           mism, M2*N2, (long long)maxabs, mism? "LAYOUT/ROUNDING DEFECT":"OK (packing+layout+accum consistent)");
  }
  printf("\nVERDICT: native FP4 path %s on %s.\n", p1?"ELIGIBLE (t=24 exact)":"INELIGIBLE (t<24 rounds / unavailable)", p.name);
  return p1?0:1;
}
#endif
