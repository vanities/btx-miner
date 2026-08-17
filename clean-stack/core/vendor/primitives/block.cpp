#include <primitives/block.h>
#include <hash.h>

// BYTE-EXACT: matches btx CBlockHeader::GetHash() = SHA256d of the serialized header.
uint256 CBlockHeader::GetHash() const
{
    return (HashWriter{} << *this).GetHash();
}
