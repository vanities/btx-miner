// Shadow primitives/block.h — matmul_v4.cpp only passes CBlockHeader by const-ref
// to the (stubbed) matmul::DeriveSigma / ComputeMatMulHeaderHash; no field access
// in the routines we compile. A minimal type suffices.
#ifndef BTX_SHADOW_BLOCK_H
#define BTX_SHADOW_BLOCK_H
class CBlockHeader {};
#endif
