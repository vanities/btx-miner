# V3 coupled oracle — fill-in scaffold  [DONE 2026-07-23]

**FILLED: `bench/rc_coupled_solver_v3.cpp` is BYTE-EXACT vs the upstream medium-V3 golden
`a4bb0cc4…`, and the clean-stack consensus TU (`clean-stack/core/matmul/matmul_v4_rc_coupled.*`,
gated by `rc_coupled_probe`) is lifted from it.** Two scaffold rows were outdated by the v4.6
fold and are corrected in the implementation: (1) V3 uses its own independent `COUP_*_V3` tag
family (not the V2 tags); (2) material exchange FROZE upstream (XOR keystream + balanced lane
permutation, `exchange_rows=128`, production `exchange_rounds=4`; the rounds tag's single frozen
value is `BTX_RC_COUP_MAT_XCHG_ROUNDS_V3`). Remaining open item: the production-dims
`exchange_rounds=4` path has NO upstream CI golden — validate against a dumped ground truth
before trusting it at 1536-page scale. Historical delta-list below.

## Concrete V3 deltas from V1 (7a7ce106), as of PR head af988a8
| thing | V1 (our oracle, `7a7ce106`) | V3 |
|---|---|---|
| `rows_per_lobe` (M) | 1 (GEMV) | **128** (real M-row GEMM) |
| `bank_pages` | 8 (toy) | **1536** (production) |
| `pages_per_barrier_lobe` | 12 | **24** |
| `lobe_width` (K=N) | 32 (toy) | **8192** |
| domain tags | `..._V1` | **`..._V2`** (classifier caught `kRCCoupEpisodeTagV2` … `kRCCoupMaterialExchangeTagV2`) |
| transcript version | ENC_RC_V1 | **ENC_RC_V3** |
| material exchange | hashes exchange_rows into the mix domain | **REAL 4 GiB exchange — STILL OPEN on his side (Workstream G may leave X_exchange OPEN). VERIFY.** |
| toy golden | `7a7ce106…` | new — a V3 toy golden (a coupled 64-hex `a4bb0cc4…` appeared in the range; CONFIRM it's the V3 toy value and pin it) |

## The one structural change: M=1 GEMV -> M-row GEMM
V1 state is `lobes × W`; the lobe op is `1×W · W×W -> 1×W`. V3 state is `lobes × M × W`; the lobe
op is a real `M×W · W×W -> M×W` GEMM. Replace `lobe_gemv` with:
```cpp
// M×W row-block . W×W page -> M×W int64 (row-major). M=1 reduces to the V1 GEMV.
static void lobe_gemm(const i8* rows, const std::vector<i8>& page, u32 M, u32 W, i64* out){
    for (u32 r=0;r<M;++r) for (u32 c=0;c<W;++c){ i64 s=0;
        for (u32 k=0;k<W;++k) s += (i64)rows[(size_t)r*W+k]*(i64)page[(size_t)k*W+c];
        out[(size_t)r*W+c]=s; }
}
```
State init becomes the first M rows of the W×W lobe tile (VERIFY against ground-truth — could be M
distinct tiles or the top M rows of one). `state_bytes()` -> `lobes*rows_per_lobe*lobe_width`.
`acc`, `partial`, permutation length `n`, and Extract all scale by M automatically once `n` includes M.

## OVERFLOW (mandatory, his Workstream C): re-derive the int64 bound at V3
At W=8192, 24 accumulated pages, M=128: max |value| = mu(<=6)*2^scale(<=3) = 48, so |a_k*b_k| <= 2304; a single dot product
over W=8192 <= 1.89e7; 24-page accum <= 4.53e8 (fine alone); then permutation/mix (butterfly a±b) and 4 exchange rounds. Prove the running int64
bound through EVERY mix/exchange stage — his own note: "multiple unrestricted butterfly rounds can
overflow int64." If he specifies a ring/field, mirror it EXACTLY (CPU==GPU). Do NOT assume the V1
butterfly composes safely at V3 depth.

## Fill-in checklist (each gated by dumped ground-truth from his build, per REACT-RUNBOOK §2)
1. [ ] set params: M=128, bank_pages=1536, pages/slot=24, W=8192, barriers=8, lobes=8
2. [ ] switch domain tags V1 -> V2 (all nine `kRCCoup*Tag`)
3. [ ] lobe_gemv -> lobe_gemm (above); state init to M rows; n includes M
4. [ ] material exchange: implement the FROZEN V3 semantics (currently OPEN — do not guess)
5. [ ] re-derive + assert the int64 overflow bound at every stage
6. [ ] pin the V3 toy golden; validate byte-exact (dump-bisect if it misses)
7. [ ] then medium + production dims; resolve his known "medium-digest mismatch" if it reproduces
8. [ ] gate in clean-stack (`rc_coupled_probe`), re-measure economics at real dims

## Reused verbatim (do NOT touch)
SHA256/d, ChaCha20, M11, `expand_mx_dequant_i8`, `extract_mx_tile_i64`, `sha_tag_u32(u32)`, ShaXof,
`mix_ascending/descending`, `balanced_perm`, `bank_commitment` (double-SHA), `barrier_root`
(double-SHA), header->sigma/bank_root_seed recompute. These have been stable v4.4-LT -> V1 -> V3.
