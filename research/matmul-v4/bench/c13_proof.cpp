// C-13 algorithm proof (CPU, private): the mod-q sketch combine Chat = P*Q mod q
// computed via limb-decomposed integer GEMMs (which on a GPU become s8->s32
// tensor-core GEMMs), byte-exact to the direct mod-q combine.
//
// P (m x n), Q (n x m) are exact int32 with |.| < 2^30 (entries of U*A, B*V).
// Bias to unsigned P' = P + B, Q' = Q + B (B = 2^30), split each into L base-128
// limbs in [0,127] (valid non-negative s8). Then
//     P'[a][k] = sum_i Pl_i[a][k] 128^i,   Q'[k][c] = sum_j Ql_j[k][c] 128^j
// so   sum_k P'[a][k] Q'[k][c] = sum_{i,j} 128^(i+j) * (sum_k Pl_i[a][k] Ql_j[k][c])
//                              = sum_{i,j} 128^(i+j) * G[i][j][a][c]
// where each G[i][j] = Pl_i * Ql_j is an exact integer GEMM (|G| <= n*127^2 < 2^31,
// so on a GPU it is a u8/s8 -> s32 tensor GEMM). Recover the signed product:
//     Chat[a][c] = ( P'Q'[a][c] - B*rowP'[a] - B*colQ'[c] + n*B^2 ) mod q.
// This moves the length-n reduction (the expensive part) onto tensor cores,
// leaving only O(m^2) shift/reduce/correct on the ALU.
//
// build: clang++ -O3 -std=c++17 c13_proof.cpp -o c13_proof   (runs anywhere)

#include <cstdint>
#include <cstdio>
#include <vector>

static constexpr uint64_t Q = ((uint64_t)1 << 61) - 1;
static inline uint64_t fqred(unsigned __int128 x){ uint64_t lo=(uint64_t)(x&Q),hi=(uint64_t)(x>>61);uint64_t s=lo+hi;s=(s&Q)+(s>>61);if(s>=Q)s-=Q;return s; }
static inline uint64_t fqadd(uint64_t a,uint64_t b){uint64_t s=a+b;if(s>=Q)s-=Q;return s;}
static inline uint64_t fqmul(uint64_t a,uint64_t b){return fqred((unsigned __int128)a*b);}
static inline uint64_t fqsub(uint64_t a,uint64_t b){return a>=b?a-b:a+Q-b;}
static inline uint64_t fqfromS(int64_t x){ if(x>=0) return fqred((unsigned __int128)(uint64_t)x); uint64_t r=fqred((unsigned __int128)(uint64_t)(-x)); return r?Q-r:0; }

int main(){
    const uint32_t m=32, n=256;                // representative (identity holds any size)
    const int32_t B = 1<<30;                    // bias
    const int L = 5;                            // base-128 limbs: 5*7=35 bits covers P' < 2^31
    // deterministic pseudo-random P,Q in (-2^30, 2^30)  (no rng deps)
    auto rnd=[&](uint64_t& s){ s^=s<<13; s^=s>>7; s^=s<<17; return s; };
    uint64_t s=0x9e3779b97f4a7c15ull;
    std::vector<int32_t> P((size_t)m*n), Qm((size_t)n*m);
    for(auto& x:P)  x = (int32_t)(rnd(s)%(2u<<30)) - B;   // [-B, B)
    for(auto& x:Qm) x = (int32_t)(rnd(s)%(2u<<30)) - B;

    // ---- direct reference: Chat[a][c] = sum_k P[a][k]*Q[k][c] mod q ----
    std::vector<uint64_t> direct((size_t)m*m,0);
    for(uint32_t a=0;a<m;a++) for(uint32_t c=0;c<m;c++){
        uint64_t acc=0;
        for(uint32_t k=0;k<n;k++) acc=fqadd(acc, fqmul(fqfromS(P[(size_t)a*n+k]), fqfromS(Qm[(size_t)k*m+c])));
        direct[(size_t)a*m+c]=acc;
    }

    // ---- C-13 limb path ----
    // bias + split into L base-128 limbs (u8 in [0,127])
    std::vector<std::vector<uint8_t>> Pl(L, std::vector<uint8_t>((size_t)m*n)), Ql(L, std::vector<uint8_t>((size_t)n*m));
    for(size_t idx=0;idx<(size_t)m*n;idx++){ uint32_t v=(uint32_t)(P[idx]+B);  for(int i=0;i<L;i++){ Pl[i][idx]=(uint8_t)((v>>(7*i))&0x7f);} }
    for(size_t idx=0;idx<(size_t)n*m;idx++){ uint32_t v=(uint32_t)(Qm[idx]+B); for(int j=0;j<L;j++){ Ql[j][idx]=(uint8_t)((v>>(7*j))&0x7f);} }
    // bias-correction sums: rowP'[a]=sum_k P'[a][k], colQ'[c]=sum_k Q'[k][c] (mod q)
    std::vector<uint64_t> rowP(m,0), colQ(m,0);
    for(uint32_t a=0;a<m;a++){ uint64_t acc=0; for(uint32_t k=0;k<n;k++) acc=fqadd(acc,fqfromS((int64_t)P[(size_t)a*n+k]+B)); rowP[a]=acc; }
    for(uint32_t c=0;c<m;c++){ uint64_t acc=0; for(uint32_t k=0;k<n;k++) acc=fqadd(acc,fqfromS((int64_t)Qm[(size_t)k*m+c]+B)); colQ[c]=acc; }
    // 128^s mod q table
    uint64_t p128[2*L]; p128[0]=1; for(int t=1;t<2*L;t++) p128[t]=fqmul(p128[t-1],128);
    const uint64_t Bq=fqfromS(B), nB2=fqmul(fqfromS((int64_t)n), fqmul(Bq,Bq));

    std::vector<uint64_t> limb((size_t)m*m);
    size_t mism=0; int32_t maxG=0;
    for(uint32_t a=0;a<m;a++) for(uint32_t c=0;c<m;c++){
        // P'Q'[a][c] = sum_{i,j} 128^(i+j) * G[i][j], G = sum_k Pl_i[a][k]*Ql_j[k][c] (exact int32)
        uint64_t ppq=0;
        for(int i=0;i<L;i++) for(int j=0;j<L;j++){
            int32_t g=0;                                  // this is the tensor-core GEMM entry on GPU
            for(uint32_t k=0;k<n;k++) g += (int32_t)Pl[i][(size_t)a*n+k]*(int32_t)Ql[j][(size_t)k*m+c];
            if(g>maxG)maxG=g;
            ppq = fqadd(ppq, fqmul(p128[i+j], fqfromS(g)));
        }
        // Chat = P'Q' - B*rowP'[a] - B*colQ'[c] + n*B^2   (mod q)
        uint64_t v = ppq;
        v = fqsub(v, fqmul(Bq, rowP[a]));
        v = fqsub(v, fqmul(Bq, colQ[c]));
        v = fqadd(v, nB2);
        limb[(size_t)a*m+c]=v;
        if(v!=direct[(size_t)a*m+c]) mism++;
    }
    printf("C-13 limb combine vs direct mod-q combine:  mismatches = %zu / %u   %s\n",
           mism, m*m, mism? "FAIL":"BYTE-IDENTICAL");
    printf("  L=%d limbs (base-128), %d GEMMs/output, max limb-GEMM entry = %d (< 2^31 = %s exact s32)\n",
           L, L*L, maxG, (maxG < (1<<30))? "yes":"check");
    printf("  => the length-n reduction is %d exact s8->s32 tensor GEMMs; only O(m^2) shift/correct on ALU.\n", L*L);
    return 0;
}
