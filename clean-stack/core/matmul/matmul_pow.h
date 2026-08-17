// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BTX_MATMUL_MATMUL_POW_H
#define BTX_MATMUL_MATMUL_POW_H

// Header commitment for the matmul PoW. Both functions are pure SHA-256 over the
// header fields, and they survived the v4 cut unchanged: sigma is still the root
// of the ENC_RC episode chain, and the header hash is still what the coupled bank
// template keys off. Everything else that used to live here -- PowState/PowConfig,
// Solve/Verify, the low-rank Denoise recovery -- belonged to the v3 solver and
// went with it.

#include <uint256.h>

class CBlockHeader;

namespace matmul {

uint256 ComputeMatMulHeaderHash(const CBlockHeader& header);
// Consensus sigma derivation, defined from the full block header.
uint256 DeriveSigma(const CBlockHeader& header);

} // namespace matmul

#endif // BTX_MATMUL_MATMUL_POW_H
