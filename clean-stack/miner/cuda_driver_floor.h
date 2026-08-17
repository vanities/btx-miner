// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.
//
// Pure version math for the CUDA driver-floor preflight (see CheckCudaDriverFloor in
// matador-miner.cpp). Standalone + header-only so harness/cuda_driver_floor_test.cpp can
// exercise it without CUDA, a GPU, or a driver: the failure mode this gate exists to catch
// (miner runs, GPU never engages, nonce/s stays 0) is exactly the one a false PASS reopens,
// and a false BLOCK refuses to mine on a machine that would have been fine.

#ifndef MATADOR_MINER_CUDA_DRIVER_FLOOR_H
#define MATADOR_MINER_CUDA_DRIVER_FLOOR_H

// CUDA packs a version as 1000*major + 10*minor: 13030 == 13.3, 12040 == 12.4, 12000 == 12.0.
// Both cuDriverGetVersion() and cudaDriverGetVersion() report the LATEST CUDA the installed
// driver supports -- the same number nvidia-smi prints in its "CUDA Version:" header field,
// and exactly what cudart compares against at init.
//
// Returns false (leaving major/minor untouched) when `packed` carries no usable version, which
// is how "no driver installed" arrives: the CUDA docs specify 0 for that case.
inline bool DecodeCudaVersion(int packed, int& major, int& minor)
{
    if (packed <= 0) {
        return false;
    }
    const int decoded_major = packed / 1000;
    if (decoded_major <= 0) {
        return false;
    }
    major = decoded_major;
    minor = (packed % 1000) / 10;
    return true;
}

// True when a driver advertising CUDA (drv_major.drv_minor) can initialize a runtime that links
// CUDA (req_major.req_minor).
//
// Deliberately conservative -- an unknown driver version (drv_major <= 0) PASSES. This gate
// aborts the miner, so a false block is worse than a missed catch: the fallback path (backend
// probe -> "running on CPU" warning) still catches a driver that really cannot init, whereas a
// false block refuses to start on a machine that would have mined fine.
inline bool CudaDriverMeetsFloor(int drv_major, int drv_minor, int req_major, int req_minor)
{
    if (drv_major <= 0) {
        return true;
    }
    if (drv_major != req_major) {
        return drv_major > req_major;
    }
    return drv_minor >= req_minor;
}

#endif // MATADOR_MINER_CUDA_DRIVER_FLOOR_H
