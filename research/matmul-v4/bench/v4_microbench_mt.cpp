// MatMul v4 work-shape microbench -- MULTI-THREADED (full-chip Mac / any CPU).
// Same 3 stages as v4_microbench.cpp but operand-gen + GEMMs are parallelized
// across all hardware threads, to get a realistic full-chip nonce/s for the
// M4 Max. SHA-256 via CommonCrypto (hardware SHA on Apple silicon).
//
// build (mac):  clang++ -O3 -std=c++17 v4_microbench_mt.cpp -o v4_microbench_mt
// run:          ./v4_microbench_mt 4096         # n=4096, all hw threads
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point t0){return std::chrono::duration<double,std::milli>(clk::now()-t0).count();}
static constexpr uint8_t kReject=251; static constexpr int32_t kBal=125;
static inline int8_t SampleS8(const uint8_t seed[32],uint32_t index){
    for(uint32_t retry=0;retry<256;++retry){
        uint8_t buf[40]; memcpy(buf,seed,32);
        buf[32]=index;buf[33]=index>>8;buf[34]=index>>16;buf[35]=index>>24; size_t len=36;
        if(retry>0){buf[36]=retry;buf[37]=retry>>8;buf[38]=retry>>16;buf[39]=retry>>24;len=40;}
        uint8_t h[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(buf,(CC_LONG)len,h);
        if(h[0]<kReject) return (int8_t)((int32_t)h[0]-kBal);
    } return 0;
}
static constexpr uint64_t kQ=((uint64_t)1<<61)-1;
static inline uint64_t FqR(unsigned __int128 x){uint64_t lo=(uint64_t)(x&kQ),hi=(uint64_t)(x>>61);uint64_t s=lo+hi;s=(s&kQ)+(s>>61);if(s>=kQ)s-=kQ;return s;}
static inline uint64_t FqA(uint64_t a,uint64_t b){uint64_t s=a+b;if(s>=kQ)s-=kQ;return s;}
static inline uint64_t FqM(uint64_t a,uint64_t b){return FqR((unsigned __int128)a*b);}
static inline uint64_t FqF(int32_t x){if(x>=0)return FqR((unsigned __int128)(uint64_t)x);uint64_t r=FqR((unsigned __int128)(uint64_t)(-(int64_t)x));return r==0?0:kQ-r;}

static void par(size_t count, int nthr, const std::function<void(size_t,size_t)>& fn){
    std::vector<std::thread> ts; size_t chunk=(count+nthr-1)/nthr;
    for(int t=0;t<nthr;t++){size_t b=(size_t)t*chunk,e=std::min(count,b+chunk);if(b>=e)break;
        ts.emplace_back([=]{fn(b,e);});}
    for(auto&t:ts)t.join();
}

int main(int argc,char**argv){
    uint32_t n=argc>1?atoi(argv[1]):4096; int nthr=argc>2?atoi(argv[2]):(int)std::thread::hardware_concurrency();
    const uint32_t m=n/8;
    printf("=== v4 microbench (MT)  n=%u  m=%u  threads=%d ===\n",n,m,nthr);
    uint8_t sA[32],sB[32],sU[32],sV[32];
    for(int i=0;i<32;i++){sA[i]=1+i;sB[i]=100+i;sU[i]=50+i;sV[i]=200+i;}
    std::vector<int8_t> A((size_t)n*n),B((size_t)n*n),U((size_t)m*n),V((size_t)n*m);

    auto t0=clk::now();
    par((size_t)n*n,nthr,[&](size_t b,size_t e){for(size_t i=b;i<e;i++)A[i]=SampleS8(sA,(uint32_t)i);});
    par((size_t)n*n,nthr,[&](size_t b,size_t e){for(size_t i=b;i<e;i++)B[i]=SampleS8(sB,(uint32_t)i);});
    par((size_t)m*n,nthr,[&](size_t b,size_t e){for(size_t i=b;i<e;i++)U[i]=SampleS8(sU,(uint32_t)i);});
    par((size_t)n*m,nthr,[&](size_t b,size_t e){for(size_t i=b;i<e;i++)V[i]=SampleS8(sV,(uint32_t)i);});
    double te=ms(t0); uint64_t sha=2ull*n*n+2ull*m*n;

    std::vector<int32_t> P((size_t)m*n,0),Q((size_t)n*m,0);
    t0=clk::now();
    par(m,nthr,[&](size_t rb,size_t re){for(size_t a=rb;a<re;a++){const int8_t*ur=&U[a*n];int32_t*pr=&P[a*n];
        for(uint32_t i=0;i<n;i++){int32_t u=ur[i];if(!u)continue;const int8_t*ar=&A[(size_t)i*n];
            for(uint32_t k=0;k<n;k++)pr[k]+=u*(int32_t)ar[k];}}});
    par(n,nthr,[&](size_t rb,size_t re){for(size_t k=rb;k<re;k++){const int8_t*br=&B[k*n];int32_t*qr=&Q[k*m];
        for(uint32_t j=0;j<n;j++){int32_t bb=br[j];if(!bb)continue;const int8_t*vr=&V[(size_t)j*m];
            for(uint32_t c=0;c<m;c++)qr[c]+=bb*(int32_t)vr[c];}}});
    double tg=ms(t0); uint64_t macs=2ull*n*n*m;

    std::vector<uint64_t> Chat((size_t)m*m,0);
    t0=clk::now();
    par(m,nthr,[&](size_t rb,size_t re){for(size_t a=rb;a<re;a++){const int32_t*pr=&P[a*n];uint64_t*cr=&Chat[a*m];
        for(uint32_t k=0;k<n;k++){int32_t p=pr[k];if(!p)continue;uint64_t pf=FqF(p);const int32_t*qr=&Q[(size_t)k*m];
            for(uint32_t c=0;c<m;c++)cr[c]=FqA(cr[c],FqM(pf,FqF(qr[c])));}}});
    double tc=ms(t0);
    volatile uint64_t sink=Chat[0]^Chat.back(); (void)sink;

    double tot=te+tg+tc;
    printf(" 1 operand-gen(SHA) %8.2f ms  %5.1f%%   %.1f M SHA-256/nonce  (%.2f GH/s aggregate)\n",te,100*te/tot,sha/1e6,sha/(te/1000.0)/1e9);
    printf(" 2 INT8 GEMMs       %8.2f ms  %5.1f%%   %.2e MAC\n",tg,100*tg/tot,(double)macs);
    printf(" 3 Fq combine       %8.2f ms  %5.1f%%\n",tc,100*tc/tot);
    printf(" TOTAL              %8.2f ms  -> %.1f nonce/s (full chip, CPU)\n",tot,1000.0/tot);
    printf(" NOTE: M4 has no INT8 tensor path -> matmul on scalar ALU; per v4 spec M1-M4 are VERIFY-ONLY (M5+ for competitive mining).\n");
    return 0;
}
