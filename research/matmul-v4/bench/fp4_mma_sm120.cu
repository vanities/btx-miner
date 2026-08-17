// Hand-written block-scaled FP4 (E2M1) tensor-core MMA on consumer Blackwell
// (RTX 5090, sm_120a) via inline PTX. This is the path cuBLASLt does NOT expose
// for sm_120 (fp4_t24.cu proved cuBLASLt returns zero FP4 algorithms on the
// 5090) — but the hardware has the cores, reachable directly through
// mma.sync ... kind::mxf8f6f4.block_scale. If this runs, matador can drive the
// 5090's FP4 units itself, so BMX4-C's FP4 lever is NOT datacenter-exclusive.
//
// Instruction + operand layout + the compute_120a gencode requirement are from
// the reverse-engineered public writeups (florianmattana.com FP4-attention-sm120,
// NVIDIA devforum block-scale threads). Container packing for kind::mxf8f6f4:
// each E2M1 element sits in an 8-bit lane with the nibble in bits 5..2 (nibble<<2).
//
// PROBE (layout-agnostic): every A and B element = +3 (E2M1 nibble 0x5), every
// block scale = 2^0 (UE8M0 byte 0x7F). Then each m16n8k32 MMA adds 32*(3*3)=288
// to every output element; chaining N MMAs (C=D feedback) gives 288*N in FP32.
// Because the rail is uniform, the (undocumented) fragment layout is irrelevant:
// any accumulator register must read exactly 288*N. N=58254 -> 16,777,152 = 2^24-64.
//
// build (pc, CUDA 13.3):
//   nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_mma_sm120.cu -o fp4_mma_sm120
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

// One warp. Fresh accumulator, N chained block-scaled FP4 MMAs, uniform +3 rail.
// out[0..3] <- this thread's 4 accumulator regs after N MMAs (all should be 288*N).
__global__ void fp4_rail(float* out, int N, float seed){
  // +3 in every E2M1 lane. kind::mxf8f6f4 uses an 8-bit container, nibble<<2:
  // 0x5<<2 = 0x14 per byte -> 0x14141414 per 32-bit register.
  uint32_t a0=0x14141414u,a1=0x14141414u,a2=0x14141414u,a3=0x14141414u; // A: 4 regs (16 FP4)
  uint32_t b0=0x14141414u,b1=0x14141414u;                               // B: 2 regs (8 FP4)
  uint32_t sA=0x7F7F7F7Fu, sB=0x7F7F7F7Fu;                              // UE8M0 2^0 in every byte
  // seed the accumulator: an ODD seed keeps the running sum odd through the whole
  // (even-step) rail, so at magnitude 2^24 it needs all 24 mantissa bits -> a t<=23
  // accumulator (ULP=2 above 2^23) provably rounds it. Layout-agnostic.
  float d0=seed,d1=seed,d2=seed,d3=seed;
  for(int i=0;i<N;i++){
    asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
        "r"(b0),"r"(b1),
        "r"(sA), "n"(0), "n"(0),
        "r"(sB), "n"(0), "n"(0));
  }
  if(threadIdx.x==0){ out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3; }
}

int main(){
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  printf("=== hand-written block-scaled FP4 MMA on %s (sm_%d%d) ===\n", p.name, p.major, p.minor);
  float* d; CK(cudaMalloc(&d, 4*sizeof(float)));
  struct { int N; double seed; double expect; const char* note; } cases[] = {
    {1,      0.0,   288.0,        "one MMA = 32*(3*3)"},
    {100,    0.0,   28800.0,      "chained x100"},
    {58254,  0.0,   16777152.0,   "even rail 2^24-64 (t>=19: even-step, low bits zero)"},
    {58261,  0.0,   16779168.0,   "even rail just over 2^24"},
    // THE M-t24 GATE: odd seed keeps the sum odd through the whole rail, so the
    // final value 16,777,145 = 2^24-71 needs all 24 mantissa bits at magnitude
    // 2^24. A true t=24 (FP32) accumulator returns it exactly; a t<=23 accumulator
    // (ULP=2 above 2^23) provably rounds it to an even neighbour. numair's target.
    {58254, -7.0,   16777145.0,   "ODD-TARGET 2^24-71  <== true t=24 discriminator (t<=23 -> even)"},
  };
  bool allok=true, mt24=false, mt24_ran=false;
  for(auto c: cases){
    CK(cudaMemset(d,0,4*sizeof(float)));
    fp4_rail<<<1,32>>>(d, c.N, (float)c.seed);
    cudaError_t e=cudaGetLastError();
    if(e!=cudaSuccess){ printf("  launch(N=%d) FAILED: %s\n", c.N, cudaGetErrorString(e)); allok=false; continue; }
    CK(cudaDeviceSynchronize());
    float h[4]; CK(cudaMemcpy(h,d,4*sizeof(float),cudaMemcpyDeviceToHost));
    bool ok = ((double)h[0]==c.expect);
    bool odd_case = (c.seed!=0.0);
    printf("  N=%-6d seed=%+.0f  d0=%16.1f  expect=%16.1f  %s   [%s]\n", c.N, c.seed, h[0], c.expect, ok?"OK":"MISMATCH", c.note);
    if(odd_case){ mt24_ran=true; mt24=ok;
      long long got=(long long)h[0];
      printf("       -> returned %lld is %s; %s\n", got, (got&1)?"ODD":"EVEN",
             ok ? "accumulator is EXACT to t=24 (true FP32)."
                : "t<=23: the odd value was rounded away -> native BMX4-C path INELIGIBLE, must fall back to INT8."); }
    if(!ok) allok=false;
  }
  printf("\n=> M-t24 VERDICT (%s): the 5090 FP4 accumulator %s\n",
         p.name, mt24_ran ? (mt24 ? "is EXACT to t=24 -> native BMX4-C FP4 path ELIGIBLE on this silicon."
                                  : "ROUNDS below t=24 -> INELIGIBLE (fall back to INT8).")
                          : "odd-target case did not run.");
  printf("   (Reachability + rails: %s. cuBLASLt exposes no MXFP4; this is a hand-written mma.sync probe.)\n",
         allok ? "all exact" : "see mismatches above");
  return (allok && (!mt24_ran || mt24)) ? 0 : 1;
}
