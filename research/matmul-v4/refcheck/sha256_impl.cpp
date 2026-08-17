// Correct streaming SHA-256 behind the CSHA256 interface, with a global
// Finalize counter (g_sha_finalize). Used ONLY to (a) count how many SHA-256
// compressions the reference operand expansion performs and (b) produce correct
// digests for the byte-exactness cross-check. KAT-verified in refcheck.cpp.
#include <crypto/sha256.h>
#include <atomic>
#include <cstring>

std::atomic<uint64_t> g_sha_finalize{0};

namespace {
inline uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t rbe32(const unsigned char* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}
inline void wbe32(unsigned char* p, uint32_t x) { p[0]=x>>24; p[1]=x>>16; p[2]=x>>8; p[3]=x; }
inline void wbe64(unsigned char* p, uint64_t x) { for (int i=0;i<8;i++) p[i]=(unsigned char)(x>>(56-8*i)); }

const uint32_t K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

void Transform(uint32_t* s, const unsigned char* chunk, size_t blocks) {
    while (blocks--) {
        uint32_t w[64];
        for (int i=0;i<16;i++) w[i]=rbe32(chunk+i*4);
        for (int i=16;i<64;i++){
            uint32_t s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3);
            uint32_t s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
        for (int i=0;i<64;i++){
            uint32_t S1=ror(e,6)^ror(e,11)^ror(e,25);
            uint32_t ch=(e&f)^((~e)&g);
            uint32_t t1=h+S1+ch+K[i]+w[i];
            uint32_t S0=ror(a,2)^ror(a,13)^ror(a,22);
            uint32_t maj=(a&b)^(a&c)^(b&c);
            uint32_t t2=S0+maj;
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
        chunk += 64;
    }
}
} // namespace

CSHA256::CSHA256() { Reset(); }
CSHA256& CSHA256::Reset() {
    s[0]=0x6a09e667;s[1]=0xbb67ae85;s[2]=0x3c6ef372;s[3]=0xa54ff53a;
    s[4]=0x510e527f;s[5]=0x9b05688c;s[6]=0x1f83d9ab;s[7]=0x5be0cd19;
    bytes=0; return *this;
}
CSHA256& CSHA256::Write(const unsigned char* data, size_t len) {
    const unsigned char* end = data + len;
    size_t bufsize = bytes % 64;
    if (bufsize && bufsize + len >= 64) {
        std::memcpy(buf + bufsize, data, 64 - bufsize);
        bytes += 64 - bufsize;
        data += 64 - bufsize;
        Transform(s, buf, 1);
        bufsize = 0;
    }
    if (size_t(end - data) >= 64) {
        size_t blocks = (end - data) / 64;
        Transform(s, data, blocks);
        data += 64 * blocks;
        bytes += 64 * blocks;
    }
    if (end > data) {
        std::memcpy(buf + bufsize, data, end - data);
        bytes += end - data;
    }
    return *this;
}
void CSHA256::Finalize(unsigned char hash[OUTPUT_SIZE]) {
    g_sha_finalize.fetch_add(1, std::memory_order_relaxed);
    static const unsigned char pad[64] = {0x80};
    unsigned char sizedesc[8];
    wbe64(sizedesc, bytes << 3);
    Write(pad, 1 + ((119 - (bytes % 64)) % 64));
    Write(sizedesc, 8);
    wbe32(hash,    s[0]); wbe32(hash+4,  s[1]); wbe32(hash+8,  s[2]); wbe32(hash+12, s[3]);
    wbe32(hash+16, s[4]); wbe32(hash+20, s[5]); wbe32(hash+24, s[6]); wbe32(hash+28, s[7]);
}
