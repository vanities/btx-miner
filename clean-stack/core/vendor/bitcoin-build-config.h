#ifndef MATADOR_BITCOIN_BUILD_CONFIG_H
#define MATADOR_BITCOIN_BUILD_CONFIG_H
// Minimal stand-in for btx's generated build-config. Empty on purpose: every
// optional asm/SIMD SHA path is gated behind defines here, so leaving them unset
// selects the portable C++ implementation (byte-identical output, just slower CPU
// reference -- the GPU backends carry the perf).
#endif
