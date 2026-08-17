// matador MatExpand (v4.4-LT) host replica — OUR code, gated against the reference
// oracle's Ahat.bin/Bhat.bin (n=64 test header). Self-contained: our SHA256 + ChaCha20.
// build: clang++ -std=c++20 -O2 matexpand.cpp -o matexpand   (run from scratchpad)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <string>
using std::vector; using u8=uint8_t; using u32=uint32_t; using u64=uint64_t; using i32=int32_t; using i8=int8_t; using i64=int64_t; using u16=uint16_t;

// ---------------- SHA-256 (compact, KAT-gated) ----------------
struct SHA256 {
    u32 h[8]; u8 buf[64]; u64 len=0; size_t n=0;
    static u32 ror(u32 x,int r){return (x>>r)|(x<<(32-r));}
    SHA256(){ h[0]=0x6a09e667;h[1]=0xbb67ae85;h[2]=0x3c6ef372;h[3]=0xa54ff53a;h[4]=0x510e527f;h[5]=0x9b05688c;h[6]=0x1f83d9ab;h[7]=0x5be0cd19; }
    void blk(const u8* p){
        static const u32 K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        u32 w[64]; for(int i=0;i<16;i++) w[i]=(p[i*4]<<24)|(p[i*4+1]<<16)|(p[i*4+2]<<8)|p[i*4+3];
        for(int i=16;i<64;i++){u32 s0=ror(w[i-15],7)^ror(w[i-15],18)^(w[i-15]>>3),s1=ror(w[i-2],17)^ror(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}
        u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for(int i=0;i<64;i++){u32 S1=ror(e,6)^ror(e,11)^ror(e,25),ch=(e&f)^((~e)&g),t1=hh+S1+ch+K[i]+w[i],S0=ror(a,2)^ror(a,13)^ror(a,22),mj=(a&b)^(a&c)^(b&c),t2=S0+mj;hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    void write(const void* data,size_t l){const u8* p=(const u8*)data;len+=l;while(l){size_t t=64-n;if(t>l)t=l;memcpy(buf+n,p,t);n+=t;p+=t;l-=t;if(n==64){blk(buf);n=0;}}}
    void final(u8 out[32]){u64 bits=len*8;u8 pad=0x80;write(&pad,1);u8 z=0;while(n!=56)write(&z,1);u8 lb[8];for(int i=0;i<8;i++)lb[i]=bits>>(56-8*i);write(lb,8);for(int i=0;i<8;i++){out[i*4]=h[i]>>24;out[i*4+1]=h[i]>>16;out[i*4+2]=h[i]>>8;out[i*4+3]=h[i];}}
};
static void sha256(const void* d,size_t l,u8 out[32]){SHA256 s;s.write(d,l);s.final(out);}

// ---------------- ChaCha20 (RFC8439, bitcoin Nonce96 layout) ----------------
static inline u32 rotl(u32 x,int n){return (x<<n)|(x>>(32-n));}
static void chacha_block(const u32 key[8], u32 counter, u32 n0, u32 n1, u32 n2, u8 out[64]){
    u32 s[16]={0x61707865,0x3320646e,0x79622d32,0x6b206574,
               key[0],key[1],key[2],key[3],key[4],key[5],key[6],key[7],
               counter,n0,n1,n2};
    u32 x[16]; memcpy(x,s,sizeof x);
    auto QR=[&](int a,int b,int c,int d){x[a]+=x[b];x[d]=rotl(x[d]^x[a],16);x[c]+=x[d];x[b]=rotl(x[b]^x[c],12);x[a]+=x[b];x[d]=rotl(x[d]^x[a],8);x[c]+=x[d];x[b]=rotl(x[b]^x[c],7);};
    for(int i=0;i<10;i++){QR(0,4,8,12);QR(1,5,9,13);QR(2,6,10,14);QR(3,7,11,15);QR(0,5,10,15);QR(1,6,11,12);QR(2,7,8,13);QR(3,4,9,14);}
    for(int i=0;i<16;i++){u32 v=x[i]+s[i];out[i*4]=v;out[i*4+1]=v>>8;out[i*4+2]=v>>16;out[i*4+3]=v>>24;}
}

// ---------------- 32-byte hash value ----------------
struct H32{ u8 d[32]; };
static H32 read_bin(const char* path){H32 h{};FILE* f=fopen(path,"rb");if(!f){fprintf(stderr,"missing %s\n",path);exit(2);}fread(h.d,1,32,f);fclose(f);return h;}

// DeriveTaggedSeed: SHA256(tag || hash.data()[32]) -> 32 bytes
static H32 tagged(const char* tag, const H32& hash){SHA256 s;s.write(tag,strlen(tag));s.write(hash.d,32);H32 o;s.final(o.d);return o;}

// M11 nibble table
static const bool M_ACC[16]={1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1};
static const i8   M_VAL[16]={0,0,1,0,2,3,4,6,0,0,-1,0,-2,-3,-4,-6};

// ExpandMantissaStream(seed, count) -> M11 int8[count]  (SeedBytesLE reverse, domain 0x6D)
static vector<i8> mant_stream(const H32& seed, size_t count){
    u8 sb[32]; for(int i=0;i<32;i++) sb[i]=seed.d[31-i];   // SeedBytesLE
    vector<i8> out(count); size_t filled=0; u64 block=0;
    while(filled<count){
        SHA256 s; s.write(sb,32); u8 dom=0x6D; s.write(&dom,1);
        u8 blk[8]; for(int i=0;i<8;i++) blk[i]=block>>(8*i); s.write(blk,8);   // WriteLE64
        u8 hash[32]; s.final(hash);
        for(int i=0;i<32 && filled<count;i++){
            u8 nb[2]={(u8)(hash[i]&0xF),(u8)((hash[i]>>4)&0xF)};
            for(u8 nib:nb){ if(M_ACC[nib]){ out[filled++]=M_VAL[nib]; if(filled==count) break; } }
        }
        ++block;
    }
    return out;
}
// exact GEMMs (row-major)
static vector<i32> gemm_s8s8(const vector<i8>&L,const vector<i8>&R,u32 rows,u32 inner,u32 cols){
    vector<i32> o((size_t)rows*cols,0);
    for(u32 i=0;i<rows;i++)for(u32 k=0;k<inner;k++){i32 l=L[(size_t)i*inner+k];if(!l)continue;const i8* r=&R[(size_t)k*cols];i32* op=&o[(size_t)i*cols];for(u32 c=0;c<cols;c++)op[c]+=l*(i32)r[c];}
    return o;
}
static vector<i32> gemm_s32s8(const vector<i32>&L,const vector<i8>&R,u32 rows,u32 inner,u32 cols){
    vector<i32> o((size_t)rows*cols,0);
    for(u32 i=0;i<rows;i++)for(u32 k=0;k<inner;k++){i32 l=L[(size_t)i*inner+k];if(!l)continue;const i8* r=&R[(size_t)k*cols];i32* op=&o[(size_t)i*cols];for(u32 c=0;c<cols;c++)op[c]+=l*(i32)r[c];}
    return o;
}
// ExtractDequantMatExpand
static const u32 LANE_MANT=0x4D414E54u, LANE_SCALE=0x53434C45u;
static u64 prf_le64(const u32 key[8], i32 raw, u32 i, u32 j, u32 remix, u32 lane){
    u32 n0=(u32)raw ^ lane;              // nonce_first
    u64 ns=((u64)i<<32)|(u64)j;          // nonce_second
    u8 ks[64]; chacha_block(key, remix, n0, (u32)(ns&0xffffffff), (u32)(ns>>32), ks);
    u64 v=0; for(int k=0;k<8;k++) v|=(u64)ks[k]<<(8*k); return v; // LE64
}
static i8 extract(const u32 key[8], i32 raw, u32 i, u32 j){
    for(u32 remix=0;;++remix){
        u64 mixed=prf_le64(key,raw,i,j,remix,LANE_MANT);
        for(int shift=0;shift<64;shift+=4){ u8 nib=(mixed>>shift)&0xF; if(M_ACC[nib]){
            u64 sc=prf_le64(key,raw,i,j,remix,LANE_SCALE); u8 e=sc&0x3;
            return (i8)((i32)M_VAL[nib]*(1<<e)); } }
    }
}
static void keywords(const H32& seed_w, u32 key[8]){ for(int i=0;i<8;i++) key[i]=seed_w.d[i*4]|(seed_w.d[i*4+1]<<8)|(seed_w.d[i*4+2]<<16)|((u32)seed_w.d[i*4+3]<<24); }

// MatExpandCore(tmpl, seed_w) -> Bhat n*n
static vector<i8> matexpand(const H32& tmpl, const H32& seed_w, u32 n){
    const u32 w=128;
    H32 seed_g=tagged("BTX_MATEXPAND_G_V44LT", tmpl);
    H32 seed_h=tagged("BTX_MATEXPAND_H_V44LT", tmpl);
    vector<i8> G=mant_stream(seed_g,(size_t)n*n);
    vector<i8> H=mant_stream(seed_h,(size_t)w*n);
    vector<i8> W=mant_stream(seed_w,(size_t)n*w);
    vector<i32> Y=gemm_s8s8(G,W,n,n,w);      // n x w
    vector<i32> B32=gemm_s32s8(Y,H,n,w,n);   // n x n
    // prf_key = SHA256("BTX_MATEXPAND_PRF_V44LT" || seed_w)
    H32 prf; { SHA256 s; const char* t="BTX_MATEXPAND_PRF_V44LT"; s.write(t,strlen(t)); s.write(seed_w.d,32); s.final(prf.d); }
    u32 key[8]; keywords(prf,key);
    vector<i8> Bhat((size_t)n*n);
    for(u32 i=0;i<n;i++)for(u32 j=0;j<n;j++){ size_t idx=(size_t)i*n+j; Bhat[idx]=extract(key,B32[idx],i,j); }
    return Bhat;
}

// ---------------- back half (F_q = 2^61-1 combine + digest) ----------------
static const u64 Q61=(((u64)1)<<61)-1;
static inline u64 fqred(unsigned __int128 x){ u64 lo=(u64)(x&Q61),hi=(u64)(x>>61); u64 s=lo+hi; s=(s&Q61)+(s>>61); if(s>=Q61)s-=Q61; return s; }
static inline u64 fq_i32(i32 x){ return x>=0 ? (u64)x : Q61-(u64)(-(i64)x); }
static inline u64 fqmul(u64 a,u64 b){ return fqred((unsigned __int128)a*b); }
static inline u64 fqadd(u64 a,u64 b){ u64 s=a+b; if(s>=Q61)s-=Q61; return s; }
// P = U(m x n) * A(n x n) -> m x n ; Q = B(n x n) * V(n x m) -> n x m
static vector<i32> proj_left(const vector<i8>&U,const vector<i8>&A,u32 n,u32 m){
    vector<i32> P((size_t)m*n,0);
    for(u32 a=0;a<m;a++)for(u32 i=0;i<n;i++){i32 u=U[(size_t)a*n+i];if(!u)continue;const i8*ar=&A[(size_t)i*n];i32*pr=&P[(size_t)a*n];for(u32 k=0;k<n;k++)pr[k]+=u*(i32)ar[k];}
    return P;
}
static vector<i32> proj_right(const vector<i8>&B,const vector<i8>&V,u32 n,u32 m){
    vector<i32> Q((size_t)n*m,0);
    for(u32 k=0;k<n;k++)for(u32 j=0;j<n;j++){i32 b=B[(size_t)k*n+j];if(!b)continue;const i8*vr=&V[(size_t)j*m];i32*qr=&Q[(size_t)k*m];for(u32 c=0;c<m;c++)qr[c]+=b*(i32)vr[c];}
    return Q;
}
static vector<u64> combine(const vector<i32>&P,const vector<i32>&Q,u32 n,u32 m){
    vector<u64> C((size_t)m*m,0);
    for(u32 a=0;a<m;a++){const i32*pr=&P[(size_t)a*n];u64*cr=&C[(size_t)a*m];
        for(u32 k=0;k<n;k++){i32 p=pr[k];if(!p)continue;u64 pf=fq_i32(p);const i32*qr=&Q[(size_t)k*m];
            for(u32 c=0;c<m;c++) cr[c]=fqadd(cr[c],fqmul(pf,fq_i32(qr[c])));}}
    return C;
}

// ComputeMatMulHeaderHash: SHA256(verLE||prev||merkle||timeLE||bitsLE||nonce64LE||dimLE||seed_a||seed_b)
static H32 hdr_hash(u32 ver,const u8* prev,const u8* merk,u32 tm,u32 bits,u64 nonce64,u16 dim,const u8* sa,const u8* sb){
    SHA256 s; u8 t[8];
    t[0]=ver;t[1]=ver>>8;t[2]=ver>>16;t[3]=ver>>24; s.write(t,4);
    s.write(prev,32); s.write(merk,32);
    t[0]=tm;t[1]=tm>>8;t[2]=tm>>16;t[3]=tm>>24; s.write(t,4);
    t[0]=bits;t[1]=bits>>8;t[2]=bits>>16;t[3]=bits>>24; s.write(t,4);
    for(int k=0;k<8;k++)t[k]=nonce64>>(8*k); s.write(t,8);
    t[0]=dim;t[1]=dim>>8; s.write(t,2);
    s.write(sa,32); s.write(sb,32); H32 o; s.final(o.d); return o;
}
int main(){
    { u8 o[32]; sha256("abc",3,o); const u8 exp[4]={0xba,0x78,0x16,0xbf}; if(memcmp(o,exp,4)){printf("SHA256 KAT FAIL\n");return 2;} }
    const u32 n=64, m=n/2;   // LT deep tile b=2
    // MakeLTHeader(0xdeadbeef, 64) — all-uniform uint256s so byte order is irrelevant
    u8 prev[32],merk[32],sa[32],sb[32],zero[32]; memset(prev,0x51,32);memset(merk,0xa3,32);memset(sa,0x11,32);memset(sb,0x22,32);memset(zero,0,32);
    H32 hh=hdr_hash(0x20000004u,prev,merk,1770000000u,0x207fffffu,0xdeadbeefULL,64,sa,sb);          // header hash
    H32 th=hdr_hash(0x20000004u,prev,merk,1770000000u,0x207fffffu,0,64,zero,zero);                  // template: nonce64/seed_a/seed_b zeroed
    H32 sigma; sha256(hh.d,32,sigma.d);                                                             // sigma = SHA256(header_hash)
    // MatExpand operands
    vector<i8> Ahat=matexpand(th, tagged("BTX_MATEXPAND_WA_V44LT",th), n);
    vector<i8> Bhat=matexpand(th, tagged("BTX_MATEXPAND_W_V44LT",hh), n);
    // stage gate vs oracle
    auto cmp=[&](const char* nm,const vector<i8>&ours,const char* path){FILE*f=fopen(path,"rb");if(!f){printf("%s: (no oracle file, skip xcheck)\n",nm);return true;}vector<i8>ref(ours.size());fread(ref.data(),1,ref.size(),f);fclose(f);size_t mm=0;for(size_t k=0;k<ours.size();k++)if(ours[k]!=ref[k])mm++;printf("%s: %zu/%zu %s\n",nm,ours.size()-mm,ours.size(),mm?"MISMATCH":"OK");return mm==0;};
    bool ok=cmp("Ahat",Ahat,"lt-oracle/Ahat.bin") & cmp("Bhat",Bhat,"lt-oracle/Bhat.bin");
    // U,V projectors (m x n, n x m) from template-scoped seeds
    vector<i8> U=mant_stream(tagged("BTX_MATMUL_V44LT_SKETCH_U",th),(size_t)m*n);
    vector<i8> V=mant_stream(tagged("BTX_MATMUL_V44LT_SKETCH_V",th),(size_t)n*m);
    // P,Q,Chat
    vector<i32> P=proj_left(U,Ahat,n,m);
    vector<i32> Q=proj_right(Bhat,V,n,m);
    vector<u64> Chat=combine(P,Q,n,m);
    // digest = SHA256d("BTX_MATMUL_V4" || sigma[32] || LE64(Chat[..]))
    vector<u8> buf; const char* tag="BTX_MATMUL_V4";
    buf.insert(buf.end(),tag,tag+strlen(tag));
    buf.insert(buf.end(),sigma.d,sigma.d+32);
    for(u64 w:Chat){ for(int k=0;k<8;k++) buf.push_back((u8)(w>>(8*k))); }
    u8 d1[32],d2[32]; sha256(buf.data(),buf.size(),d1); sha256(d1,32,d2);
    char hex[65]; for(int k=0;k<32;k++) sprintf(hex+2*k,"%02x",d2[31-k]);   // GetHex = reversed
    const char* EXP="db1136f2974d45d9757262978ab074ef53ba54c368df9829f565ee2d26da0da9";
    printf("payload=%zu bytes\ndigest=%s\nEXPECT=%s\n%s\n", Chat.size()*8, hex, EXP,
           (strcmp(hex,EXP)==0 && ok)? "=> FULL LT DIGEST BYTE-EXACT (db1136f2) — matador CPU LT solver validated":"=> MISMATCH");
    return (strcmp(hex,EXP)==0 && ok)?0:1;
}
