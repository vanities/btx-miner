// FP4 (MXFP4 E2M1) vs INT8 tensor-throughput CEILING on Blackwell, via hand-written
// inline PTX (cuBLASLt exposes no MXFP4). Same m16n8k32 tile run both ways, SM-
// saturating launch, aggregate TOPS -> the FP4:INT8 rate ratio the operand matmul
// (Q=B*V, M11 operands = E2M1 exact-int subset) could ride. This is the CEILING:
// a real tiled GEMM lands below it, so if FP4 is not clearly >INT8 here, the
// datacenter-FP4 thesis is dead regardless of kernel effort.
//
//   FP4 : mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32
//         .row.col.f32.e2m1.e2m1.f32.ue8m0   (needs -gencode compute_120a/100a)
//   INT8: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
// Both tiles = 16*8*32 = 4096 MAC = 8192 FLOP/mma. Values are irrelevant to
// throughput (uniform rails); this measures raw tensor issue rate, not a GEMM.
//
// build (5090): nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_int8_ceiling.cu -o fp4c
//       (B200): nvcc -O3 -gencode arch=compute_100a,code=sm_100a fp4_int8_ceiling.cu -o fp4c
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

constexpr int ILP = 4;   // independent accumulator groups per warp (hide mma latency)

// Each warp runs N chained mmas across ILP independent accumulators -> throughput.
__global__ void fp4_tput(float* sink, int N){
  uint32_t a0=0x14141414u,a1=0x14141414u,a2=0x14141414u,a3=0x14141414u; // +3 rail (E2M1 nibble<<2)
  uint32_t b0=0x14141414u,b1=0x14141414u;
  uint32_t sA=0x7F7F7F7Fu, sB=0x7F7F7F7Fu;                              // UE8M0 2^0
  float d[ILP*4];
  #pragma unroll
  for(int i=0;i<ILP*4;i++) d[i]=0.0f;
  for(int n=0;n<N;n++){
    #pragma unroll
    for(int g=0;g<ILP;g++){
      asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[g*4+0]),"+f"(d[g*4+1]),"+f"(d[g*4+2]),"+f"(d[g*4+3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
          "r"(sA),"n"(0),"n"(0), "r"(sB),"n"(0),"n"(0));
    }
  }
  float acc=0; for(int i=0;i<ILP*4;i++) acc+=d[i];
  if(threadIdx.x==0 && acc<0) sink[blockIdx.x]=acc;   // keep live, never taken
}

// Dedicated 4-bit path: kind::mxf4, m16n8k64 (true 4-bit packing, DOUBLE the K per
// issue). If the tensor core issues this at the same rate as k32, it is 2x INT8 --
// the "real" FP4 lever. scale_vec::2X selects two per-32 UE8M0 block scales for k64.
__global__ void mxf4_tput(float* sink, int N){
  uint32_t a0=0x55555555u,a1=0x55555555u,a2=0x55555555u,a3=0x55555555u; // 4-bit packed E2M1 (nibble 0x5 = +3)
  uint32_t b0=0x55555555u,b1=0x55555555u;
  uint32_t sA=0x7F7F7F7Fu, sB=0x7F7F7F7Fu;
  float d[ILP*4];
  #pragma unroll
  for(int i=0;i<ILP*4;i++) d[i]=0.0f;
  for(int n=0;n<N;n++){
    #pragma unroll
    for(int g=0;g<ILP;g++){
      asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(d[g*4+0]),"+f"(d[g*4+1]),"+f"(d[g*4+2]),"+f"(d[g*4+3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
          "r"(sA),"n"(0),"n"(0), "r"(sB),"n"(0),"n"(0));
    }
  }
  float acc=0; for(int i=0;i<ILP*4;i++) acc+=d[i];
  if(threadIdx.x==0 && acc<0) sink[blockIdx.x]=acc;
}

__global__ void int8_tput(int32_t* sink, int N){
  uint32_t a0=0x01010101u,a1=0x01010101u,a2=0x01010101u,a3=0x01010101u; // s8 = +1 rail
  uint32_t b0=0x01010101u,b1=0x01010101u;
  int32_t d[ILP*4];
  #pragma unroll
  for(int i=0;i<ILP*4;i++) d[i]=0;
  for(int n=0;n<N;n++){
    #pragma unroll
    for(int g=0;g<ILP;g++){
      asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[g*4+0]),"+r"(d[g*4+1]),"+r"(d[g*4+2]),"+r"(d[g*4+3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
    }
  }
  int32_t acc=0; for(int i=0;i<ILP*4;i++) acc+=d[i];
  if(threadIdx.x==0 && acc<0) sink[blockIdx.x]=acc;
}

template<class K, class T>
static double run_tops(K kern, T* sink, int blocks, int threads, int N, int reps, int kdim){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  kern<<<blocks,threads>>>(sink,N);                 // warmup
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  for(int r=0;r<reps;r++) kern<<<blocks,threads>>>(sink,N);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms=0; cudaEventElapsedTime(&ms,a,b);
  const int warps = threads/32;
  const double mmas = (double)reps*blocks*warps*(double)N*ILP;
  const double flop = mmas * (16.0*8.0*(double)kdim*2.0);   // MACs = 16*8*K
  cudaEventDestroy(a); cudaEventDestroy(b);
  return flop / (ms*1e-3) / 1e12;                   // TOPS
}

int main(int argc,char**argv){
  int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
  const int N   = argc>1?atoi(argv[1]):20000;
  const int reps= argc>2?atoi(argv[2]):20;
  const int bpsm= argc>3?atoi(argv[3]):4;           // blocks per SM (occupancy)
  const int threads=256;
  const int blocks = p.multiProcessorCount*bpsm;
  printf("device: %s  sm_%d%d  SMs=%d  launch=%dx%d  N=%d reps=%d\n",
         p.name,p.major,p.minor,p.multiProcessorCount,blocks,threads,N,reps);

  float*  fsink; CK(cudaMalloc(&fsink,blocks*sizeof(float)));
  int32_t*isink; CK(cudaMalloc(&isink,blocks*sizeof(int32_t)));

  double int8 = run_tops(int8_tput,  isink, blocks, threads, N, reps, 32);
  double f8f4 = run_tops(fp4_tput,   fsink, blocks, threads, N, reps, 32);  // kind::mxf8f6f4 (8-bit container)
  double mxf4 = 0.0;
  mxf4_tput<<<1,32>>>(fsink,1);                       // probe: does kind::mxf4 even run on this arch?
  cudaError_t probe = cudaDeviceSynchronize();
  if(probe==cudaSuccess) mxf4 = run_tops(mxf4_tput, fsink, blocks, threads, N, reps, 64);  // k64
  else printf(" [kind::mxf4 unavailable on this arch: %s]\n", cudaGetErrorString(probe));

  printf("\n INT8               (m16n8k32 s8)   : %8.1f TOPS\n", int8);
  printf(" MXFP4 mxf8f6f4     (8-bit container): %8.1f TOPS   %.2fx INT8\n", f8f4, f8f4/int8);
  if(mxf4>0) printf(" MXFP4 mxf4         (true 4-bit k64) : %8.1f TOPS   %.2fx INT8\n", mxf4, mxf4/int8);
  printf(" ---------------------------------------\n");
  double best = mxf4>f8f4?mxf4:f8f4;
  printf(" best FP4 : INT8 = %.2fx  %s\n", best/int8,
         best/int8>=1.8?"(~2x -- FP4 has real headroom)":
         best/int8>=1.3?"(partial -- FP4 helps but under 2x)":
                       "(FP4 <= INT8 -- NO tensor-rate advantage)");
  printf("CSVFP4,%s,%d,%.1f,%.1f,%.1f,%.3f\n", p.name, p.multiProcessorCount, int8, f8f4, mxf4, best/int8);
  return 0;
}
