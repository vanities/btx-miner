// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <matmul/matmul_pow.h>

#include <crypto/common.h>
#include <crypto/sha256.h>
#include <primitives/block.h>
#include <span.h>

#include <cstdint>

namespace matmul {

uint256 ComputeMatMulHeaderHash(const CBlockHeader& header)
{
    CSHA256 hasher;

    uint8_t version_le[4];
    uint8_t time_le[4];
    uint8_t bits_le[4];
    uint8_t nonce64_le[8];
    uint8_t dim_le[2];

    WriteLE32(version_le, static_cast<uint32_t>(header.nVersion));
    WriteLE32(time_le, header.nTime);
    WriteLE32(bits_le, header.nBits);
    WriteLE64(nonce64_le, header.nNonce64);
    WriteLE16(dim_le, header.matmul_dim);

    hasher.Write(version_le, sizeof(version_le));
    hasher.Write(header.hashPrevBlock.data(), uint256::size());
    hasher.Write(header.hashMerkleRoot.data(), uint256::size());
    hasher.Write(time_le, sizeof(time_le));
    hasher.Write(bits_le, sizeof(bits_le));
    hasher.Write(nonce64_le, sizeof(nonce64_le));
    hasher.Write(dim_le, sizeof(dim_le));
    hasher.Write(header.seed_a.data(), uint256::size());
    hasher.Write(header.seed_b.data(), uint256::size());

    uint8_t digest[CSHA256::OUTPUT_SIZE];
    hasher.Finalize(digest);
    return uint256{Span<const unsigned char>{digest, CSHA256::OUTPUT_SIZE}};
}

uint256 DeriveSigma(const CBlockHeader& header)
{
    const uint256 header_hash = ComputeMatMulHeaderHash(header);

    uint8_t sigma_bytes[CSHA256::OUTPUT_SIZE];
    CSHA256().Write(header_hash.data(), uint256::size()).Finalize(sigma_bytes);
    return uint256{Span<const unsigned char>{sigma_bytes, CSHA256::OUTPUT_SIZE}};
}

} // namespace matmul
