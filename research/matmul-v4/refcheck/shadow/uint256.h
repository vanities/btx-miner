// Minimal shadow of BTX uint256.h — just the surface int8_field.cpp /
// matmul_v4.cpp use (Span ctor, data(), static size(), ==). Avoids the real
// header's util/strencodings + util/string dependency chain.
#ifndef BTX_SHADOW_UINT256_H
#define BTX_SHADOW_UINT256_H
#include <span.h>
#include <cstddef>
#include <cstring>
class uint256 {
    unsigned char m_data[32];
public:
    uint256() { std::memset(m_data, 0, 32); }
    explicit uint256(Span<const unsigned char> s) { std::memcpy(m_data, s.data(), 32); }
    const unsigned char* data() const { return m_data; }
    unsigned char* data() { return m_data; }
    static constexpr std::size_t size() { return 32; }
    friend bool operator==(const uint256& a, const uint256& b) { return std::memcmp(a.m_data, b.m_data, 32) == 0; }
    friend bool operator!=(const uint256& a, const uint256& b) { return !(a == b); }
};
#endif
