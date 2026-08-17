// Shadow crypto/sha256.h — same CSHA256 surface int8_field.cpp / matmul_v4.cpp
// use, backed by our own correct streaming SHA-256 (sha256_impl.cpp) with a
// Finalize counter. Avoids the real sha256.cpp's bitcoin-build-config.h /
// compat/cpuid.h / arch-dispatch dependencies.
#ifndef BTX_SHADOW_CRYPTO_SHA256_H
#define BTX_SHADOW_CRYPTO_SHA256_H
#include <cstdint>
#include <cstdlib>
class CSHA256 {
private:
    uint32_t s[8];
    unsigned char buf[64];
    uint64_t bytes;
public:
    static const size_t OUTPUT_SIZE = 32;
    CSHA256();
    CSHA256& Write(const unsigned char* data, size_t len);
    void Finalize(unsigned char hash[OUTPUT_SIZE]);
    CSHA256& Reset();
};
#endif
