# DRAFT — PR #89 comment: the BMX4-C native path cannot dispatch via cuBLASLt (MXFP4 vs NVFP4)
# Voice: Adam. No em-dashes, no AI attribution. Post after the measurement comment.
# PRIVATE NOTES (do not post): reveals nothing about matador kernels. The instruction
# probe is framed as derived from public SM120 reverse-engineering, which it is.

---

Heads up on the v4.2 native path before anyone burns B200 hours on M-t24: as committed, `RunMxf4Gemm` can never dispatch on any current NVIDIA part, and the reason is a format gap in cuBLASLt itself, not a bug in your code.

**The gap.** `matmul_v4_bmx4_accel.cu` requests `CUDA_R_4F_E2M1` operands with `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` scales, which is MXFP4. cuBLASLt has no kernel for that combination on any architecture or toolkit we can find. Per the cuBLAS docs, E2M1 operands are served only with `VEC16_UE4M3` scales (NVFP4), and `VEC32_UE8M0` scales are served only for FP8 operands (MXFP8). MXFP4 block-scaled GEMM is simply not in the library.

**Measured, not inferred.** We mirrored your exact recipe (E2M1 + VEC32_UE8M0 + COMPUTE_32F, TN, FP32 out, unit 0x7F scales) into a standalone probe and ran it on real silicon:

| device | cuBLASLt | MXFP4 (your recipe) | NVFP4 control | INT8 control |
|---|---|---|---|---|
| RTX 5090 (sm_120) | 13.5.1 (CUDA 13.3) | no algorithm | AVAILABLE | AVAILABLE |
| B200 (sm_100) | 12.8.4 | no algorithm | n/a | AVAILABLE |
| B200 (sm_100) | 13.0.0 | no algorithm | n/a | AVAILABLE |

`cublasLtMatmulAlgoGetHeuristic` returns zero algorithms for the MXFP4 recipe at every shape we tried (256/512/1024 squares plus a 32x32xK=1,864,128 rail), both `FAST_ACCUM` settings. The INT8 and NVFP4 controls through the identical harness dispatch fine, so this is the format combination, not toolkit age or a harness bug.

**What this means for the design.**
1. The native path fails closed to the 1-GEMM INT8 fallback on every NVIDIA card, exactly as your fallback ladder intends. Fail-closed works. But the tax inversion never engages on NVIDIA through cuBLASLt, on B200 same as on consumer cards.
2. M-t24 qualification through cuBLASLt would report ineligible for lack of a kernel, not measure the accumulator. The verdict would be about NVIDIA's library packaging, not the silicon.
3. Switching to NVFP4 is not an out: UE4M3 scales are the fractional scales the spec already rejects for determinism, and that rejection is correct.

**The road that does work.** The hardware itself is fine. The PTX instruction underneath, `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0`, is exposed on consumer Blackwell, and the public SM120 reverse-engineering (florianmattana.com's FP4 attention writeup, the devforum block-scale threads) documents the fragment layouts. We verified at the instruction level with a minimal standalone probe on a 5090:

- It compiles only with `-gencode arch=compute_120a,code=sm_120a`. Plain `-arch=sm_120` gets "instruction not supported" from ptxas, which is the trap in the llama.cpp #19662 and cutlass #3227 issues.
- For `kind::mxf8f6f4` the E2M1 element sits in bits 5..2 of an 8-bit container (nibble<<2), not bits 3..0.
- An all-(+3) rail with unit UE8M0 scales accumulates exactly: 58,254 chained MMAs land on 16,777,152 = 2^24 - 64 bit-perfect in the FP32 accumulator. Every partial sum in the BMX4-C regime (E_max*n = 48*4096 well under 2^24) is exactly representable, so summation order cannot matter. Straight caveat: a rail of even steps pins the accumulator to t>=19, not the full t=24. Your odd-target probe (16,777,145) is the right discriminator, and it is reachable by mixing one odd-product MMA into the rail. We have not run that variant.
- `kind::mxf4` (m16n8k64, true packed nibbles) is the denser variant a production kernel would want. Same UE8M0 scales.

So the M-t24 gate is answerable on consumer Blackwell today, but only through CUTLASS block-scaled MXF4 or a hand-written kernel, not through cuBLASLt. If the native path stays cuBLASLt-based, I would document that it is inert on NVIDIA until NVIDIA ships MXFP4 GEMM, and treat CUTLASS/tcgen05 as the actual integration target for sm_100/sm_120.

Happy to run the odd-target instruction probe on the 5090 if that unblocks the M-t24 ladder ahead of the CUTLASS work.
