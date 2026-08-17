// Shadow matmul/matmul_pow.h — only the two decls matmul_v4.cpp references,
// without the real header's arith_uint256 / noise / solver_runtime chain.
#ifndef BTX_SHADOW_MATMUL_POW_H
#define BTX_SHADOW_MATMUL_POW_H
#include <uint256.h>
class CBlockHeader;
namespace matmul {
uint256 ComputeMatMulHeaderHash(const CBlockHeader& header);
uint256 DeriveSigma(const CBlockHeader& header);
} // namespace matmul
#endif
