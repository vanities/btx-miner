#!/usr/bin/env bash
# Reference ENC-BMX4C-LT oracle: compiles numair's own ComputeDigestBMX4CLT to
# reproduce golden db1136f2… (n=64 test header) and dump stage intermediates.
# The byte-exact ground truth for validating our matador LT replica.
#   git -C ~/git/btx-stock worktree add -f /tmp/btx-pr89 pr89-latest
#   bash lt_oracle_build.sh   # writes ./lt-oracle/oracle ; run it
set -e
SRC=${BTX_SRC:-/tmp/btx-pr89/src}
mkdir -p lt-oracle/cfg
cat > lt-oracle/cfg/bitcoin-build-config.h <<'H'
#ifndef BITCOIN_BUILD_CONFIG_H
#define BITCOIN_BUILD_CONFIG_H
#define CLIENT_VERSION_MAJOR 0
#define CLIENT_VERSION_MINOR 0
#define CLIENT_VERSION_BUILD 0
#define CLIENT_VERSION_IS_RELEASE false
#define COPYRIGHT_YEAR 2026
#define CLIENT_NAME "oracle"
#define CLIENT_BUGREPORT ""
#define COPYRIGHT_HOLDERS ""
#define COPYRIGHT_HOLDERS_SUBSTITUTION ""
#endif
H
printf '#include <string>\nstd::string FormatFullVersion(){ return std::string(); }\n' > lt-oracle/stub.cpp
# driver.cpp: build MakeLTHeader(0xdeadbeef,64), call lt::ComputeDigestBMX4CLT, print digest + Ahat/Bhat.
clang++ -std=c++20 -O2 -Ilt-oracle/cfg -I"$SRC" lt-oracle/driver.cpp lt-oracle/stub.cpp \
  "$SRC"/matmul/matmul_v4_lt.cpp "$SRC"/matmul/matmul_v4.cpp "$SRC"/matmul/matmul_v4_bmx4.cpp "$SRC"/matmul/matmul_pow.cpp "$SRC"/matmul/int8_field.cpp \
  "$SRC"/matmul/field.cpp "$SRC"/matmul/matrix.cpp "$SRC"/matmul/noise.cpp "$SRC"/matmul/transcript.cpp "$SRC"/matmul/solver_runtime.cpp \
  "$SRC"/crypto/sha256.cpp "$SRC"/crypto/chacha20.cpp "$SRC"/crypto/siphash.cpp "$SRC"/crypto/hex_base.cpp \
  "$SRC"/uint256.cpp "$SRC"/arith_uint256.cpp "$SRC"/support/cleanse.cpp \
  "$SRC"/util/strencodings.cpp "$SRC"/util/time.cpp "$SRC"/util/threadnames.cpp "$SRC"/util/fs.cpp "$SRC"/util/syserror.cpp "$SRC"/util/check.cpp "$SRC"/logging.cpp \
  -framework Accelerate -o lt-oracle/oracle
echo "built lt-oracle/oracle — run it; expect digest db1136f2974d45d9757262978ab074ef53ba54c368df9829f565ee2d26da0da9"
