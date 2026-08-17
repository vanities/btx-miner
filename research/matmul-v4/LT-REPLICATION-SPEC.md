# ENC-BMX4C-LT (v4.4-LT) byte-exact replication spec (reference @ pr89-latest / fda2690)

Gate golden: `ComputeDigestBMX4CLT(header, n=64)` = **`db1136f2974d45d9757262978ab074ef53ba54c368df9829f565ee2d26da0da9`**
Test header: prevblock `5151…`, merkle `a3a3…`, seed_a `1111…`, seed_b `2222…` (see src/test/matmul_v4_lt_tests.cpp:60-73).

## The chain (src/matmul/matmul_v4_lt.cpp: ComputeDigestBMX4CLT)
LT REUSES the entire BMX4-C/v4 back half; only operand-gen + tile change.

```
m = n/2                             // ValidateDimsBMX4(n, kTileBLT=2)  [BMX4-C used b=4 -> n/4]
sigma       = DeriveSigma(header)                         // UNCHANGED v4
seed_u,seed_v = DeriveTaggedSeed("BTX_MATMUL_V44LT_SKETCH_U"/"..._V", ComputeTemplateHash(header))
Ahat = ExpandOperandAMatExpand(header, n)                 // MatExpand, TEMPLATE-scoped
Bhat = ExpandOperandBMatExpand(header, n)                 // MatExpand, NONCE-fresh
U = ExpandProjectorBMX4C(seed_u, m, n)                    // REUSED (M11, scale-free)
V = ExpandProjectorBMX4C(seed_v, n, m)                    // REUSED
P = ComputeProjectedLeft(U, Ahat, n, m)                  // REUSED  P=U*Ahat  (m x n)
Q = ComputeProjectedRight(Bhat, V, n, m)                 // REUSED  Q=Bhat*V  (n x m)
Chat = ComputeCombineModQ(P, Q, n, m)                    // REUSED  mod q=2^61-1
payload = SerializeSketch(Chat); digest = ComputeSketchDigest(sigma, payload)  // REUSED
```
NOTE |Ahat|,|Bhat| <= 48 (not <=6 like BMX4-C mantissa), |U|,|V| <= 6. NO E8M0 block-scale
stream anywhere (scale baked into each int8 by the Extract). So the FP4-axis issue is absent.

## MatExpand (the only genuinely-new operand gen) — MatExpandCore
```
w = kMatExpandPanelW = 128
seed_g = DeriveTaggedSeed("BTX_MATEXPAND_G_V44LT", tmpl)   // tmpl = ComputeTemplateHash(header)
seed_h = DeriveTaggedSeed("BTX_MATEXPAND_H_V44LT", tmpl)
G = ExpandProjectorBMX4C(seed_g, n, n)   // n x n  M11
H = ExpandProjectorBMX4C(seed_h, w, n)   // w x n  M11
W = ExpandProjectorBMX4C(seed_w, n, w)   // n x w  M11   (seed_w per caller, below)
Y   = ExactGemmS8S8 (G, W, n, n, w)      // (n x w) exact s8xs8->s32,  |Y|<=36n
B32 = ExactGemmS32S8(Y, H, n, w, n)      // (n x n) exact s32xs8->s32
prf_key = DeriveMatExpandPrfKey(seed_w)  // = SHA256("BTX_MATEXPAND_PRF_V44LT" || seed_w)
Bhat[i*n+j] = ExtractDequantMatExpand(B32[i*n+j], i, j, prf_key)   // i,j in [0,n)
```
- ExpandOperandB seed_w = DeriveTaggedSeed("BTX_MATEXPAND_W_V44LT",  ComputeMatMulHeaderHash(header))  // nonce-fresh
- ExpandOperandA seed_w = DeriveTaggedSeed("BTX_MATEXPAND_WA_V44LT", ComputeTemplateHash(header))       // template
- ExactGemmS8S8 / S32S8: plain row-major triple loop, zero-MAC skip (byte-identical to naive), exact int32.

## ExtractDequantMatExpand(raw, i, j, prf_key)  — ChaCha20 PRF
```
remix = 0
loop:
  mant  = ChaCha20 keystream(prf_key, raw, i, j, remix, lane=MANT='MANT'=0x4D414E54); mixed = LE64(mant[0:8])
  for shift in 0,4,8,...,60:
     nib = (mixed >> shift) & 0xF; mu = SampleMantissaNibble(nib)  // M11 accept 11/16 (SAME table as BMX4-C)
     if accepted:
        scale = ChaCha20 keystream(prf_key, raw, i, j, remix, lane=SCALE='SCLE'=0x53434C45); e = LE64(scale[0:8]) & 0x3
        return (int8)( (int32)mu * (1 << e) )        // exact mul, |.|<=48
  ++remix
```
ChaCha20 keystream (RFC8439, crypto/chacha20.h): key = prf_key (32B); Nonce96 = {nonce_first, nonce_second};
  nonce_first  = (uint32)raw ^ lane
  nonce_second = ((uint64)i << 32) | (uint64)j     // FULL-WIDTH — never truncate i,j to 16b (consensus-split + reopens 32x shortcut)
  block counter = remix
quarter-round = standard ChaCha (rotl 16,12,8,7). Seek(nonce, counter) then Keystream(out<=64B).

## Seed/hash helpers (all REUSED from BMX4-C, already byte-exact in clean-stack)
DeriveTaggedSeed(tag,hash) = SHA256(tag || hash).  Sha256dPair for Merkle. ComputeTemplateHash /
ComputeMatMulHeaderHash / DeriveSigma / SampleMantissaNibble (M11 table) — identical to BMX4-C.

## Phase B "seal-as-PoW" (window of Q* in {64,128}) — DEFERRED, not needed for the base digest gate.
Window slots via DeriveWindowSlotId/Nonce, ComputeWindowMerkleRoot (Bitcoin odd-dup), SealWindowCommit.
Only relevant if fMatMulLTSealAsPoW; the base ComputeDigestBMX4CLT gate above does not touch it.

## Build target: matador LT solver
Reuse clean-stack BMX4-C back half (U/V expand, P/Q GEMMs, 4-base-64 combine, device digest); ADD MatExpand
(G/W/H = same M11 projector; two exact GEMMs; ChaCha Extract) + set m=n/2 + new tags. Gate = db1136f2… at n=64.
GPU note: ExactGemmS32S8 is int32xint8 (|Y| up to 36n, exceeds int8) — needs a custom int32xint8 GEMM or
decomposition on GPU (not a stock tensor op); CPU replica first, then solve the GPU S32S8 path.
