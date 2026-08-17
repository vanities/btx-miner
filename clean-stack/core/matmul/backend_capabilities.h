// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.

#ifndef BTX_MATMUL_BACKEND_CAPABILITIES_H
#define BTX_MATMUL_BACKEND_CAPABILITIES_H

// Which solver backend is compiled in, which is actually usable on this host, and
// which one we ended up on. Two kinds survive the v4 cut: CUDA (the ENC_RC episode
// backend) and CPU (the byte-exact oracle -- a determinism cross-check, ~200x too
// slow to mine with). METAL and HIP are gone: neither ever had an ENC_RC path, and
// leaving them selectable would only produce a miner that silently does nothing.

#include <string>
#include <utility>
#include <vector>

namespace matmul::backend {

enum class Kind {
    CPU,
    CUDA,
};

struct Capability {
    bool compiled{false};
    bool available{false};
    std::string reason;
};

struct Selection {
    std::string requested_input;
    bool requested_known{true};
    Kind requested{Kind::CPU};
    Kind active{Kind::CPU};
    std::string reason;
};

std::string ToString(Kind kind);
Capability CapabilityFor(Kind kind);
std::vector<std::pair<Kind, Capability>> AllCapabilities();
Selection ResolveRequestedBackend(const std::string& requested);

//! Resolve from BTX_MATMUL_BACKEND (empty -> cpu) and log the outcome once. A GPU
//! backend that was requested but is unavailable is reported at WARNING with the
//! probe reason, so a silent CPU fallback -- which on the v4 path means the rig
//! effectively stops mining -- can never hide.
Selection ResolveMiningBackendFromEnvironment();

} // namespace matmul::backend

#endif // BTX_MATMUL_BACKEND_CAPABILITIES_H
