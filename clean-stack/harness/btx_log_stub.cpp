// No-op btx logging so digest_probe can link matador_core standalone.
// matador_core's solver sources call LogPrintf (-> LogInstance().LogPrintStr); the real impl
// lives in btx libbitcoin_util, which the miner links but the standalone byte-exact probe does
// not. The probe doesn't need logs, so these are no-ops. (#13)
#include <logging.h>
#include <util/time.h>

#include <chrono>

void BCLog::Logger::LogPrintStr(std::string_view, std::source_location&&, BCLog::LogFlags,
                                BCLog::Level, bool) {}

bool BCLog::Logger::WillLogCategoryLevel(BCLog::LogFlags, BCLog::Level) const { return false; }

BCLog::Logger& LogInstance()
{
    static BCLog::Logger logger;
    return logger;
}

// Real wall-clock seconds for probes that pull in pow_solve.cpp.o (the solver's
// header-time refresh calls GetTime). The miner links btx libbitcoin_util for this;
// standalone probes get the same semantics from the system clock.
int64_t GetTime()
{
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// pow_solve.cpp.o also carries the difficulty-adjustment entry points
// (GetNextWorkRequired & co.), which reference CBlockIndex chain traversal that
// probes never exercise (probes drive SolveMatMul with plain headers, no chain).
// Satisfy the linker with never-called stubs; the miner links the real btx impl.
#include <chain.h>

CBlockIndex* CBlockIndex::GetAncestor(int) { return nullptr; }
const CBlockIndex* CBlockIndex::GetAncestor(int) const { return nullptr; }
