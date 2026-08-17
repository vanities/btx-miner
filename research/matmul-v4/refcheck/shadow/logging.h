// Shadow logging.h — LogPrintf is only hit on the (unreachable) oracle fallback.
#ifndef BTX_SHADOW_LOGGING_H
#define BTX_SHADOW_LOGGING_H
template <typename... Args> inline void LogPrintf(Args&&...) {}
#endif
