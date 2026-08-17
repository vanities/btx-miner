// clean-stack/miner/log_tee.h
//
// Optional --log-file support. Mirrors EVERYTHING the miner writes to stderr
// (the mlog/std::clog lines, the raw fprintf telemetry, plus any CUDA/driver
// or crash output) into a file, while STILL printing to the original stderr
// (console / systemd journal). Implemented as an fd-level tee on STDERR_FILENO
// so it captures all stderr writers with zero per-call-site changes.
//
//   * Off the hot path: a detached drain thread does the file writes; the GPU
//     feed loop never touches the disk. Log volume is a few lines per window,
//     so this is microseconds and cannot gate mining.
//   * No fsync: plain write() into the page cache (durability is the OS's job).
//   * The console copy is byte-for-byte (color preserved); the FILE copy has
//     ANSI escapes stripped so the log stays greppable.
//   * Fail-soft: any setup error logs a warning and leaves stderr untouched
//     (the miner keeps mining -- ABM).
//   * Exec-safe: the miner re-exec's itself (auto-update, solo<->pool
//     fallback). Call ExecRestore() before any execv() so the replacement
//     process inherits the REAL stderr -- the drain thread does NOT survive
//     exec, and an un-restored pipe would block/SIGPIPE the new process. The
//     re-exec'd process re-installs the tee from its own args/env.
//
// POSIX-only (matches the rest of the miner: <unistd.h>/STDERR_FILENO).
#pragma once

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

namespace logtee {

// Saved real stderr (console / journal). Set on Install(); read by ExecRestore()
// to hand the real stderr back before the process re-exec's itself. -1 = no tee.
inline std::atomic<int>& ConsoleFd() { static std::atomic<int> fd{-1}; return fd; }

// write() that retries partial writes / EINTR; gives up this sink on hard error
// (e.g. the console pipe was closed) rather than spinning or throwing.
inline void WriteAll(int fd, const char* p, size_t n)
{
    while (n > 0) {
        ssize_t w = ::write(fd, p, n);
        if (w < 0) { if (errno == EINTR) continue; break; }
        p += static_cast<size_t>(w);
        n -= static_cast<size_t>(w);
    }
}

// Drains the pipe forever: raw bytes -> console, ANSI-stripped bytes -> file.
// Exits only on EOF (process teardown) or an unrecoverable read error. Owns no
// shared fds beyond the read end; the OS reclaims everything at process exit.
inline void DrainLoop(int read_fd, int console_fd, int file_fd)
{
    char buf[8192];
    char clean[8192];
    // Minimal CSI stripper: 0 = normal, 1 = just saw ESC, 2 = inside CSI params.
    // The miner only emits SGR color codes (ESC '[' ... 'm'), which this covers.
    int esc = 0;
    for (;;) {
        ssize_t n = ::read(read_fd, buf, sizeof(buf));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        WriteAll(console_fd, buf, static_cast<size_t>(n));  // console: exact bytes (keep color)
        size_t c = 0;
        for (ssize_t k = 0; k < n; ++k) {
            unsigned char ch = static_cast<unsigned char>(buf[k]);
            if (esc == 0) {
                if (ch == 0x1B) { esc = 1; continue; }          // ESC
                clean[c++] = static_cast<char>(ch);
            } else if (esc == 1) {
                esc = (ch == '[') ? 2 : 0;                      // CSI vs other 2-byte escape (drop)
            } else {  // esc == 2
                if (ch >= 0x40 && ch <= 0x7E) esc = 0;          // final byte ends the CSI
            }
        }
        if (c) WriteAll(file_fd, clean, c);
    }
}

// Installs the tee, mirroring stderr into `path` (opened for append). Returns
// true if the tee is now active. Fail-soft: returns false and leaves stderr
// untouched on any error. Callers guard re-install with the returned bool (a
// second successful call would stack another tee).
inline bool Install(const std::string& path)
{
    if (path.empty()) return false;

    const bool was_tty = ::isatty(STDERR_FILENO) != 0;

    int file_fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (file_fd < 0) {
        std::fprintf(stderr, "[log-tee] cannot open '%s': %s -- continuing without --log-file\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }
    int pipefd[2];
    if (::pipe(pipefd) != 0) {
        std::fprintf(stderr, "[log-tee] pipe() failed: %s -- continuing without --log-file\n",
                     std::strerror(errno));
        ::close(file_fd);
        return false;
    }
    // The drain thread's read end must not leak across a re-exec.
    ::fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);

    int console_fd = ::dup(STDERR_FILENO);
    if (console_fd < 0) {
        std::fprintf(stderr, "[log-tee] dup(stderr) failed: %s -- continuing without --log-file\n",
                     std::strerror(errno));
        ::close(pipefd[0]); ::close(pipefd[1]); ::close(file_fd);
        return false;
    }
    // Saved console is restored onto fd 2 by ExecRestore() right before execv,
    // so it too must be close-on-exec (the restored fd 2 is what the new process
    // inherits).
    ::fcntl(console_fd, F_SETFD, FD_CLOEXEC);

    if (::dup2(pipefd[1], STDERR_FILENO) < 0) {
        std::fprintf(stderr, "[log-tee] dup2 failed: %s -- continuing without --log-file\n",
                     std::strerror(errno));
        ::close(pipefd[0]); ::close(pipefd[1]); ::close(file_fd); ::close(console_fd);
        return false;
    }
    ::close(pipefd[1]);                       // STDERR_FILENO now owns the write end
    ::setvbuf(stderr, nullptr, _IONBF, 0);    // unbuffered (the default; reaffirm so lines flow promptly)

    ConsoleFd().store(console_fd, std::memory_order_release);
    std::thread(DrainLoop, pipefd[0], console_fd, file_fd).detach();

    // fd 2 is now a (non-TTY) pipe, which would auto-disable color. Keep color
    // on the console by forcing it on -- the file copy strips ANSI regardless,
    // so this only affects the console. Honor an explicit NO_COLOR/FORCE_COLOR.
    if (was_tty && std::getenv("NO_COLOR") == nullptr && std::getenv("FORCE_COLOR") == nullptr) {
        ::setenv("FORCE_COLOR", "1", 1);
    }
    return true;
}

// Hand the real stderr back to STDERR_FILENO before the process re-exec's. The
// drain thread does not survive execv(); without this the replacement process
// would write into a pipe nobody drains (block, then SIGPIPE). No-op if the tee
// was never installed. The re-exec'd process re-installs its own tee.
inline void ExecRestore()
{
    int c = ConsoleFd().load(std::memory_order_acquire);
    if (c < 0) return;
    ::fflush(nullptr);                  // push buffered stdio through the tee first
    ::dup2(c, STDERR_FILENO);           // restore real stderr for the new image
}

// Scan argv + env for an early log-file path (CLI wins over env) so the tee can
// be installed BEFORE the first log line. A config-file path is picked up later
// (post-parse) since the config file is not read yet at this point.
inline std::string EarlyPath(int argc, char* argv[])
{
    const std::string key = "--log-file";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(key + "=", 0) == 0) return a.substr(key.size() + 1);
        if (a == key && i + 1 < argc)   return argv[i + 1];
    }
    if (const char* e = std::getenv("MATADOR_LOG_FILE"); e && *e) return std::string(e);
    return std::string();
}

}  // namespace logtee
