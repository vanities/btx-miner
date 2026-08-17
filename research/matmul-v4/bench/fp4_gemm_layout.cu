// Byte-exact native FP4 GEMM -- feasibility gate, step 2: NON-UNIFORM m16n8k64 tile.
//
// Step 1 (fp4_gemm_exact.cu) proved the mechanism (encoding + block scale + FP32->int) is exact
// with uniform operands. This step validates the FRAGMENT LAYOUT: load a known non-uniform M11
// tile per the standard mma m16n8k64 .row.col e2m1 mapping, run one mma, and compare C to a host
// int8 reference. If C matches, the layout is right and a tiled GEMM is just tiling + chunked
// draining on top. Wrong layout = wrong values (not a crash), so this is validated, not assumed.
//
// build (5090): nvcc -O3 -gencode arch=compute_120a,code=sm_120a fp4_gemm_layout.cu -o fp4gl
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s -> %s\n",#x,cudaGetErrorString(e));return 2;} }while(0)

__host__ __device__ static inline uint8_t e2m1_code(int v){
    int m=v<0?-v:v; uint8_t c;
    switch(m){case 0:c=0;break;case 1:c=0b010;break;case 2:c=0b100;break;
              case 3:c=0b101;break;case 4:c=0b110;break;case 6:c=0b111;break;default:c=0;break;}
    return v<0?(c|0b1000):c;
}
// M11 value cycle for a deterministic non-uniform fill (all E2M1-exact).
__host__ __device__ static inline int m11_at(int idx){
    static const int V[8]={1,2,3,4,6,-1,-2,-3};
    return V[idx&7];
}

// A: 16x64 row-major, B: 64x8 col-major, both M11 via m11_at(row*64+k) / m11_at(k*8+col).
// Block scales unit (validate raw layout first; scale already proven in step 1).
__global__ void fp4_tile(int32_t* out /*16*8*/){
    const int t=threadIdx.x, gr=t/4, tc=t%4;
    auto packA=[&](int row,int kbase)->uint32_t{
        uint32_t r=0;
        #pragma unroll
        for(int i=0;i<8;i++) r |= (uint32_t)e2m1_code(m11_at(row*64+kbase+i))<<(i*4);
        return r;
    };
    auto packB=[&](int col,int kbase)->uint32_t{
        uint32_t r=0;
        #pragma unroll
        for(int i=0;i<8;i++) r |= (uint32_t)e2m1_code(m11_at((kbase+i)*8+col))<<(i*4);
        return r;
    };
    uint32_t a0=packA(gr,tc*8), a1=packA(gr+8,tc*8), a2=packA(gr,tc*8+32), a3=packA(gr+8,tc*8+32);
    uint32_t b0=packB(gr,tc*8), b1=packB(gr,tc*8+32);
    uint32_t sU=0x7F7F7F7Fu;                                  // E8M0 2^0 everywhere
    float d[4]={0,0,0,0};
    asm volatile(
      "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
        "r"(sU),"n"(0),"n"(0), "r"(sU),"n"(0),"n"(0));
    // C layout: c0=(gr,tc*2) c1=(gr,tc*2+1) c2=(gr+8,tc*2) c3=(gr+8,tc*2+1)
    out[(gr)*8   + tc*2]   = (int32_t)llrintf(d[0]);
    out[(gr)*8   + tc*2+1] = (int32_t)llrintf(d[1]);
    out[(gr+8)*8 + tc*2]   = (int32_t)llrintf(d[2]);
    out[(gr+8)*8 + tc*2+1] = (int32_t)llrintf(d[3]);
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("device: %s sm_%d%d\n",p.name,p.major,p.minor);
    // host int8 reference: C[i,j] = sum_k A[i,k]*B[k,j]
    int32_t ref[16*8];
    for(int i=0;i<16;i++)for(int j=0;j<8;j++){ long s=0;
        for(int k=0;k<64;k++) s += (long)m11_at(i*64+k)*m11_at(k*8+j); ref[i*8+j]=(int32_t)s; }
    int32_t* d; CK(cudaMalloc(&d,16*8*sizeof(int32_t))); CK(cudaMemset(d,0,16*8*sizeof(int32_t)));
    fp4_tile<<<1,32>>>(d);
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("LAUNCH ERR %s\n",cudaGetErrorString(e)); return 2; }
    int32_t got[16*8]; CK(cudaMemcpy(got,d,sizeof(got),cudaMemcpyDeviceToHost));
    int fails=0;
    for(int i=0;i<16;i++)for(int j=0;j<8;j++) if(got[i*8+j]!=ref[i*8+j]) fails++;
    printf("first row ref: "); for(int j=0;j<8;j++) printf("%5d ",ref[j]); printf("\n");
    printf("first row got: "); for(int j=0;j<8;j++) printf("%5d ",got[j]); printf("\n");
    printf("mismatched of 128: %d\n", fails);
    printf("\n%s\n", fails==0 ? "LAYOUT CORRECT -- non-uniform FP4 tile == int8 reference, byte-exact"
                              : "LAYOUT WRONG -- adjust A/B/C fragment mapping");
    return fails==0?0:1;
}
