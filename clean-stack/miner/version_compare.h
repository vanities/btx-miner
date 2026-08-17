// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.
#ifndef MATADOR_MINER_VERSION_COMPARE_H
#define MATADOR_MINER_VERSION_COMPARE_H

#include <string>

// Numeric semver compare: returns true iff release `a` is strictly newer than running `b`.
// Parses the leading major.minor.patch (tolerates a 'v' prefix and ignores any -suffix like
// -obs/-dbg). Components are compared as INTEGERS, so 0.6.10 > 0.6.9 and 10.0.0 > 9.9.9.
//
// This replaces the old auto-update gate's `tag != cur` test, which adopted ANY differing release
// (so a higher local build downgraded to the latest published) and, under a lexical compare,
// sorted "0.6.10" BELOW "0.6.9". Equal numeric versions return false (no re-adopt / no downgrade);
// a -suffix is ignored, so bump the numeric version for a real release rather than relying on it.
//
// Tested by clean-stack/harness/version_compare_test.cpp.
inline bool VersionGreater(const std::string& a, const std::string& b)
{
    auto parse = [](const std::string& s, long out[3]) {
        out[0] = out[1] = out[2] = 0;
        size_t i = (!s.empty() && (s[0] == 'v' || s[0] == 'V')) ? 1 : 0;
        for (int comp = 0; comp < 3 && i < s.size(); ++comp) {
            long n = 0;
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') { n = n * 10 + (s[i] - '0'); ++i; }
            out[comp] = n;
            if (i < s.size() && s[i] == '.') ++i; else break;
        }
    };
    long va[3], vb[3];
    parse(a, va);
    parse(b, vb);
    for (int i = 0; i < 3; ++i) if (va[i] != vb[i]) return va[i] > vb[i];
    return false;  // equal numeric version -> not newer
}

#endif  // MATADOR_MINER_VERSION_COMPARE_H
