#include <matmul/matmul_v4_lt.h>
#include <matmul/matmul_v4.h>
#include <primitives/block.h>
#include <uint256.h>
#include <cstdio>
#include <vector>
static uint256 PU(const char* h){ auto o=uint256::FromHex(h); return o.value(); }
int main(){
    CBlockHeader hdr;
    hdr.nVersion=0x20000004;
    hdr.hashPrevBlock=PU("5151515151515151515151515151515151515151515151515151515151515151");
    hdr.hashMerkleRoot=PU("a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3a3");
    hdr.nTime=1770000000u; hdr.nBits=0x207fffffu;
    hdr.nNonce64=0xdeadbeefULL; hdr.nNonce=0xdeadbeefu; hdr.matmul_dim=64;
    hdr.seed_a=PU("1111111111111111111111111111111111111111111111111111111111111111");
    hdr.seed_b=PU("2222222222222222222222222222222222222222222222222222222222222222");
    uint256 digest; std::vector<unsigned char> payload;
    bool ok = matmul::v4::lt::ComputeDigestBMX4CLT(hdr, 64, digest, payload);
    printf("ok=%d\ndigest=%s\nEXPECT=db1136f2974d45d9757262978ab074ef53ba54c368df9829f565ee2d26da0da9\npayload_bytes=%zu\n",
           ok, digest.GetHex().c_str(), payload.size());
    auto A=matmul::v4::lt::ExpandOperandAMatExpand(hdr,64);
    auto B=matmul::v4::lt::ExpandOperandBMatExpand(hdr,64);
    long sa=0,sb=0; int amin=127,amax=-128,bmin=127,bmax=-128;
    for(int v:A){sa+=v; if(v<amin)amin=v; if(v>amax)amax=v;}
    for(int v:B){sb+=v; if(v<bmin)bmin=v; if(v>bmax)bmax=v;}
    printf("Ahat: n=%zu sum=%ld range[%d,%d]\nBhat: n=%zu sum=%ld range[%d,%d]\n",
           A.size(),sa,amin,amax,B.size(),sb,bmin,bmax);
    return ok?0:1;
}
