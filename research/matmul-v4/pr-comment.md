Really interesting proposal, and the O(n^2) Freivalds sketch is clean. I built a cross-hardware benchmark against the reference implementation, and I want to flag where the datacenter-wins goal breaks, because it looks structural.

**On real hardware a consumer RTX 5090 beats an H100 at v4, and widening the operand XOF does not fix it.**

**1. The operand XOF makes mining SHA-bound.** A, B and the U, V projectors are expanded one matrix element per SHA-256 (`SampleBalancedS8FromOracle` in `src/matmul/int8_field.cpp`, keeping only `hash[0]`, looped by `ExpandOperand`). At n=4096 that is 37,748,736 SHA-256 per nonce against about 1.7e10 INT8 MACs. Instrumenting this PR's own code confirms 38,499,212 SHA-256 per nonce.

**2. A wide XOF just moves the wall to the mod-q combine, not the tensor cores.** The sketch enforces only `2*n^2*m` INT8 MACs (the optimal miner evaluates `(U*A)(B*V)` directly, per 0.7-(3)), which on tensor cores is about 0.06 ms at n=4096. Everything else is the XOF and the `Chat = P*Q mod q` combine, both on integer ALU. Measured INT8 share of per-nonce work, RTX 5090:

| XOF | operand-gen (SHA) | INT8 matmul | mod-q combine |
|---|---|---|---|
| per element | 62.9% | 1.6% | 35.5% |
| wide | 23.5% | 4.7% | 72.8% |

**Cross-hardware (H100 SXM vs RTX 5090, both bit-exact to the reference digest, which also confirms cross-architecture determinism across sm_90 and sm_120):**

| card | n=4096 wide XOF | nonce per rental-dollar |
|---|---|---|
| RTX 5090 (consumer) | 670 nonce/s | 8.0M |
| H100 SXM (datacenter) | 339 nonce/s | 0.70M |

The 5090 is about 2x faster per card and about 11x better per rental-dollar. I tuned the combine to a bandwidth-friendly tiled kernel first, which helped the H100 more than the 5090 (as its HBM would predict) and lifted the INT8 share to about 12 percent, and the H100 still loses. The reason is structural: under the sketch the enforced INT8 work is capped at `2*n^2*m`, so it tops out near 13 percent even on an H100, and 87 percent or more of each nonce runs on integer and memory paths where high-clock consumer cards win.

**What it means.** A wide XOF is still worth doing, since it removes a SHA-256 bottleneck that a cheap SHA ASIC would otherwise exploit. But on its own it does not make v4 compute-bound on INT8 or make datacenter parts win. Making the INT8 GEMM dominant looks like it needs the full-C profile (enforce the full `n^3` product), which the sketch was chosen to avoid for payload and verification cost. So there is a real tension between cheap Freivalds verification and being INT8-compute-bound, and I do not think the current sketch construction delivers the datacenter-favoring goal.

Full writeup, methodology, difficulty analysis, and a reproduction harness that instruments the reference's own code: https://github.com/vanities/btx/blob/matmul-v4-review/REVIEW-matmul-v4-operand-xof.md

Happy to share the benchmark.
