// Copyright (c) 2026 The BTX developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/license/mit/.

#include <matmul/backend_capabilities.h>

#include <logging.h>
#include <matmul/matmul_v4_rc.h>   // RCEpisodeGpuAvailable

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <utility>

namespace matmul::backend {
namespace {

char ToLowerAscii(char c)
{
    if (c >= 'A' && c <= 'Z') {
        return static_cast<char>(c + ('a' - 'A'));
    }
    return c;
}

std::string ToLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](char c) {
        return ToLowerAscii(c);
    });
    return value;
}

Capability CpuCapability()
{
    return Capability{
        .compiled = true,
        .available = true,
        .reason = "always_available",
    };
}

Capability CudaCapability()
{
#if defined(MATADOR_ENABLE_CUDA)
    // Probe the backend we actually mine with: the ENC_RC episode kernel chain.
    // Asking anything else (a v3 digest probe, a bare driver-present check) can
    // report "available" for a build that cannot run an episode.
    const bool ok = matmul::v4::rc::RCEpisodeGpuAvailable();
    return Capability{
        .compiled = true,
        .available = ok,
        .reason = ok ? "rc_episode_backend_ready" : "no_cuda_device_visible",
    };
#else
    return Capability{
        .compiled = false,
        .available = false,
        .reason = "disabled_by_build",
    };
#endif
}

Kind ParseKind(const std::string& requested, bool& known)
{
    const std::string normalized = ToLower(requested);
    if (normalized == "cpu") {
        known = true;
        return Kind::CPU;
    }
    if (normalized == "cuda") {
        known = true;
        return Kind::CUDA;
    }

    known = false;
    return Kind::CPU;
}

} // namespace

std::string ToString(Kind kind)
{
    switch (kind) {
    case Kind::CPU:
        return "cpu";
    case Kind::CUDA:
        return "cuda";
    }

    return "cpu";
}

Capability CapabilityFor(Kind kind)
{
    switch (kind) {
    case Kind::CPU:
        return CpuCapability();
    case Kind::CUDA:
        return CudaCapability();
    }

    return CpuCapability();
}

std::vector<std::pair<Kind, Capability>> AllCapabilities()
{
    return {
        {Kind::CPU, CapabilityFor(Kind::CPU)},
        {Kind::CUDA, CapabilityFor(Kind::CUDA)},
    };
}

Selection ResolveRequestedBackend(const std::string& requested)
{
    Selection selection;
    selection.requested_input = requested;

    bool known{false};
    selection.requested = ParseKind(requested, known);
    selection.requested_known = known;

    if (!known) {
        selection.active = Kind::CPU;
        selection.reason = "unknown_backend_fallback_to_cpu";
        return selection;
    }

    const auto requested_capability = CapabilityFor(selection.requested);
    if (requested_capability.available) {
        selection.active = selection.requested;
        selection.reason = "requested_backend_available";
        return selection;
    }

    selection.active = Kind::CPU;
    selection.reason = ToString(selection.requested) + "_unavailable_fallback_to_cpu:" + requested_capability.reason;
    return selection;
}

Selection ResolveMiningBackendFromEnvironment()
{
    const char* const env_backend = std::getenv("BTX_MATMUL_BACKEND");
    const std::string requested = (env_backend != nullptr && env_backend[0] != '\0')
        ? std::string{env_backend}
        : std::string{"cpu"};
    const Selection selection = ResolveRequestedBackend(requested);

    // One unmissable line, the first time the backend resolves. A GPU backend that
    // was asked for but is unavailable is a WARNING carrying the concrete probe
    // reason, because on the v4 path a silent CPU fallback is indistinguishable
    // from a dead rig -- the episode oracle is ~200x too slow to land a share.
    static std::atomic_bool logged_resolved_backend{false};
    bool expected{false};
    if (logged_resolved_backend.compare_exchange_strong(expected, true)) {
        const std::string active_label = ToString(selection.active);
        const std::string requested_label = selection.requested_known
            ? ToString(selection.requested)
            : (selection.requested_input.empty() ? std::string{"<empty>"} : selection.requested_input);

        if (selection.active == selection.requested && selection.requested_known) {
            LogPrintf("MatMul mining backend: %s (requested=%s, %s)\n",
                      active_label, requested_label, selection.reason);
        } else {
            LogPrintf("MatMul mining backend: %s [WARNING: requested %s but it is "
                      "unavailable -> %s]\n",
                      active_label, requested_label, selection.reason);
        }
    }

    return selection;
}

} // namespace matmul::backend
