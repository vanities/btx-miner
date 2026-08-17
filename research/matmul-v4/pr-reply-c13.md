Follow-up on C-13, since it is the open item: I went ahead and prototyped the limb-decomposition combine and benched it at n=8192.

Built it byte-exact to the mod-q combine (verified across CPU, a 5090 and an H100), then ran wide-XOF + C-13 at n=8192, consumer RTX 5090 vs H100 SXM:

- RTX 5090: 161 nonce/s
- H100 SXM: 64 nonce/s

C-13 does what you want at the wall-time level: at n=8192 the combine is ~60% of the nonce and it is all on tensor cores now, so §K.2a-WT is satisfied. But it does not flip the hardware ordering. The 5090 still wins ~2.5x per card. The H100 is slower at every stage: operand-gen (still SHA + memory, ~22% of the nonce, where the 5090's clocks win), the tall-skinny U*A / B*V GEMMs, and the small m*m*n combine GEMMs (cuBLASLt is weak on those shapes).

Straight caveat: I have not done extensive optimization here. The combine is 25 sequential cuBLASLt calls, the operand-gen compaction is naive, and nothing is tuned per architecture. A batched or fused INT8 path would help the H100 and narrow this, so treat 2.5x as directional, not a final number. But the operand-gen floor is SHA/memory-bound and pulls toward the high-clock consumer card regardless of how good the GEMMs get, so I doubt the per-card ordering flips at n=8192.

Worth confirming on your own datacenter silicon with the #90 harness before setting the activation dimension. Happy to compare notes on it.
