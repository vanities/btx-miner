// lattice_commit_probe.cu -- per-nonce ring-Ajtai commit cost on real silicon.
//
// Measures the succinct-proof front-runner's eligibility-gate cost: the
// ring-Ajtai commitment over the m^2 sketch coefficients (Dilithium ring
// Z_8380417[X]/(X^256+1), gadget base-2^4 digits, module rank 4), exactly
// the math of research/matmul-v4/succinct-proof/lattice_pc.py. The op-count
// model said ~0.64x today's digest hashing; this probe replaces that with
// measured milliseconds at profile C (m=1024) and D (m=2048) scale.
//
// Two arms, because the op-count model is blind to memory traffic:
//   arm A-mem: public matrix A precomputed in global memory (1.07 GB at D)
//              -> per-nonce commit READS it all: bandwidth-bound arm.
//   arm A-fly: A regenerated in-registers from a counter-mix PRF
//              -> compute-bound arm (production would use a specified PRF;
//                 this stands in for its cost class).
//
// Cross-validation: --selftest (K=4 chunks) prints the serialized t; the
// companion lattice_check.py recomputes it with lattice_pc.py's NTT/gadget
// and the same A formula. Byte-identical => the kernel implements the same
// commitment.
//
// Build: nvcc -O3 -arch=sm_120 lattice_commit_probe.cu -o lattice_probe
// Run:   ./lattice_probe --selftest
//        ./lattice_probe --m 2048 [--iters 100]

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>

#define D 256
#define QP 8380417u
#define NOUT 4
#define GDIGITS 16
#define TPB 128           // threads per block = butterflies per NTT stage
#define GRID 2048

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s -> %s\n",__FILE__,__LINE__,#x,cudaGetErrorString(e)); exit(1);} }while(0)

__constant__ uint32_t c_zetas[D];

// ---- host helpers -----------------------------------------------------------
static uint32_t brv8(uint32_t x){ uint32_t r=0; for(int i=0;i<8;i++) r|=((x>>i)&1u)<<(7-i); return r; }
static uint64_t pw(uint64_t b,uint64_t e,uint64_t m){ uint64_t r=1; b%=m; while(e){ if(e&1) r=r*b%m; b=b*b%m; e>>=1; } return r; }

// ---- device math ------------------------------------------------------------
__device__ __host__ __forceinline__ uint32_t mix32(uint32_t x){
    x ^= x>>16; x *= 0x85EBCA6Bu; x ^= x>>13; x *= 0xC2B2AE35u; x ^= x>>16; return x;
}
// probe PRF for A (NTT-domain entries, uniform-ish mod QP). Replicated in
// lattice_check.py; the exact PRF is irrelevant to timing class.
__device__ __host__ __forceinline__ uint32_t gen_a(uint32_t seed,uint32_t r,uint32_t j,uint32_t i){
    return mix32(seed ^ (r*0x9E3779B9u) ^ mix32(j*0x85EBCA6Bu ^ i)) % QP;
}
__device__ __forceinline__ uint32_t mulmod(uint32_t a,uint32_t b){
    return (uint32_t)((uint64_t)a*b % QP);   // QP is a compile-time constant
}

// deterministic test coefficients (61-bit), replicated in lattice_check.py
__device__ __host__ __forceinline__ uint64_t gen_coeff(uint32_t i){
    uint64_t hi = mix32(i*2654435761u), lo = mix32(i ^ 0xDEADBEEFu);
    return ((hi<<32)|lo) & (((uint64_t)1<<61)-1);
}

__global__ void fill_coeffs(uint64_t* c, uint32_t n){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) c[i] = gen_coeff(i);
}
__global__ void fill_A(uint32_t* A, uint32_t K, uint32_t seed){
    // layout A[((r*K)+j)*D + i]
    size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t total = (size_t)NOUT*K*D;
    if(idx>=total) return;
    uint32_t i = idx % D, j = (idx/D) % K, r = idx/((size_t)D*K);
    A[idx] = gen_a(seed,r,j,i);
}

// ---- the commit kernel ------------------------------------------------------
// grid-stride over K chunks; per block: load 16 coeffs -> 256 digits in
// shared, NTT-256, then MAC against A (from memory or regenerated) into
// shared u64 accumulators; one atomic flush per block at the end.
// Overflow: per-chunk raw products <= (QP-1)^2 < 2^46; chunks/block at D =
// K/GRID = 128 -> local acc < 2^53 (fits u64); flush adds (< QP) x GRID
// blocks < 2^34 into global u64.
__global__ void commit_kernel(const uint64_t* __restrict__ coeffs,
                              const uint32_t* __restrict__ Aglob,
                              unsigned long long* t_acc,
                              uint32_t K, uint32_t seed, int a_from_mem)
{
    __shared__ uint32_t sa[D];
    __shared__ unsigned long long acc[NOUT][D];
    const int tid = threadIdx.x;
    for(int r=0;r<NOUT;r++){ acc[r][tid]=0ull; acc[r][tid+TPB]=0ull; }
    __syncthreads();

    for(uint32_t j = blockIdx.x; j < K; j += gridDim.x){
        // gadget-decompose 16 coefficients into 256 base-2^4 digits
        {
            uint32_t cl = tid>>3;                      // coeff 0..15 in chunk
            uint64_t c = coeffs[(size_t)j*16 + cl];
            uint32_t d0 = (tid&7)*2;                   // two digits per thread
            sa[cl*GDIGITS + d0]   = (uint32_t)((c >> (4*d0))     & 15u);
            sa[cl*GDIGITS + d0+1] = (uint32_t)((c >> (4*(d0+1))) & 15u);
        }
        __syncthreads();
        // negacyclic NTT-256 (Dilithium layout; zeta index = D/(2L) + group)
        for(int L=TPB; L>=1; L>>=1){
            int group = tid / L, pos = tid - group*L;
            int ai = group*2*L + pos;
            uint32_t z = c_zetas[(TPB/L) + group];
            uint32_t t = mulmod(z, sa[ai+L]);
            uint32_t u = sa[ai];
            sa[ai+L] = (u + QP - t) % QP;
            sa[ai]   = (u + t) % QP;
            __syncthreads();
        }
        // pointwise MAC into the NOUT accumulators
        #pragma unroll
        for(int r=0;r<NOUT;r++){
            #pragma unroll
            for(int h=0;h<2;h++){
                int i = tid + h*TPB;
                uint32_t a = a_from_mem ? Aglob[((size_t)r*K + j)*D + i]
                                        : gen_a(seed, r, j, i);
                acc[r][i] += (unsigned long long)a * sa[i];
            }
        }
        __syncthreads();
    }
    for(int r=0;r<NOUT;r++)
        for(int h=0;h<2;h++){
            int i = tid + h*TPB;
            atomicAdd(&t_acc[(size_t)r*D + i], acc[r][i] % QP);
        }
}

int main(int argc, char** argv){
    uint32_t m = 2048; int iters = 100; int selftest = 0;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--m") && i+1<argc) m = atoi(argv[++i]);
        if(!strcmp(argv[i],"--iters") && i+1<argc) iters = atoi(argv[++i]);
        if(!strcmp(argv[i],"--selftest")) selftest = 1;
    }
    const uint32_t SEED = 0x42545831u; // "BTX1"
    uint32_t n_coeffs = selftest ? 64u : m*m;
    uint32_t K = (n_coeffs*GDIGITS)/D;

    // zeta table, identical to lattice_pc.py _zetas()
    uint32_t zh[D]; for(int i=0;i<D;i++) zh[i]=(uint32_t)pw(1753,brv8(i),QP);
    CK(cudaMemcpyToSymbol(c_zetas, zh, sizeof(zh)));

    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
    printf("=== ring-Ajtai commit probe | %s | m=%u n_coeffs=%u K=%u chunks ===\n",
           prop.name, m, n_coeffs, K);

    uint64_t* dC; CK(cudaMalloc(&dC,(size_t)n_coeffs*8));
    fill_coeffs<<<(n_coeffs+255)/256,256>>>(dC,n_coeffs);
    unsigned long long* dT; CK(cudaMalloc(&dT,(size_t)NOUT*D*8));

    size_t a_bytes = (size_t)NOUT*K*D*4;
    uint32_t* dA=nullptr;
    CK(cudaMalloc(&dA,a_bytes));
    fill_A<<<(unsigned)((a_bytes/4+255)/256),256>>>(dA,K,SEED);
    CK(cudaDeviceSynchronize());

    if(selftest){
        CK(cudaMemset(dT,0,(size_t)NOUT*D*8));
        commit_kernel<<<GRID,TPB>>>(dC,dA,dT,K,SEED,1);
        CK(cudaDeviceSynchronize());
        unsigned long long hT[NOUT*D]; CK(cudaMemcpy(hT,dT,sizeof(hT),cudaMemcpyDeviceToHost));
        printf("selftest t[0..15] (mod QP): ");
        for(int i=0;i<16;i++) printf("%u ",(uint32_t)(hT[i]%QP));
        printf("\n");
        // A-fly arm must agree with A-mem arm (same PRF)
        CK(cudaMemset(dT,0,(size_t)NOUT*D*8));
        commit_kernel<<<GRID,TPB>>>(dC,dA,dT,K,SEED,0);
        CK(cudaDeviceSynchronize());
        unsigned long long hT2[NOUT*D]; CK(cudaMemcpy(hT2,dT,sizeof(hT2),cudaMemcpyDeviceToHost));
        int mism=0; for(int i=0;i<NOUT*D;i++) if(hT[i]%QP != hT2[i]%QP) mism++;
        printf("selftest arms agree: %s\n", mism? "NO **FAIL**":"yes");
        return 0;
    }

    // warmup both arms
    for(int a=0;a<2;a++){ CK(cudaMemset(dT,0,(size_t)NOUT*D*8));
        commit_kernel<<<GRID,TPB>>>(dC,dA,dT,K,SEED,a); }
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const char* names[2] = {"A-fly (compute-bound)","A-mem (bandwidth-bound)"};
    for(int arm=0; arm<2; arm++){
        CK(cudaEventRecord(e0));
        for(int it=0; it<iters; ++it){
            CK(cudaMemset(dT,0,(size_t)NOUT*D*8));
            commit_kernel<<<GRID,TPB>>>(dC,dA,dT,K,SEED,arm==1?1:0);
        }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); ms/=iters;
        double gbs = arm==1 ? (a_bytes + (double)n_coeffs*8)/(ms*1e-3)/1e9 : 0.0;
        printf(" %-24s : %8.3f ms/commit", names[arm], ms);
        if(arm==1) printf("   (%.0f GB/s effective, A=%zu MB)", gbs, a_bytes>>20);
        printf("\n");
        double budget = 1000.0/555.0;  // profile-D deploy-pipeline nonce budget
        printf("   vs D nonce budget %.2f ms: %s (%.1f%% of budget)\n",
               budget, ms<budget? "FITS":"OVER", 100.0*ms/budget);
    }
    return 0;
}
