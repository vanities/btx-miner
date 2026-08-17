// Shadow crypto/common.h — LE read/write helpers used by int8_field.cpp /
// matmul_v4.cpp (host is little-endian; asserted in int8_field.cpp).
#ifndef BTX_SHADOW_CRYPTO_COMMON_H
#define BTX_SHADOW_CRYPTO_COMMON_H
#include <cstdint>
#include <cstring>
inline uint32_t ReadLE32(const unsigned char* p) { uint32_t x; std::memcpy(&x, p, 4); return x; }
inline uint64_t ReadLE64(const unsigned char* p) { uint64_t x; std::memcpy(&x, p, 8); return x; }
inline void WriteLE32(unsigned char* p, uint32_t x) { std::memcpy(p, &x, 4); }
inline void WriteLE64(unsigned char* p, uint64_t x) { std::memcpy(p, &x, 8); }
#endif
