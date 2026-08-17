// Unit test for the pure stratum wire-parsing helpers (clean-stack/miner/stratum_client.h).
// Everything a pool SENDS is untrusted: vendored UniValue's typed getters throw
// std::runtime_error on any type mismatch (getInt<> also throws on floats like 10.0), and
// before the hardening an uncaught throw in the reader thread std::terminate'd the whole
// miner -> systemd restart -> the pool replays the same line -> crash loop. This pins the
// tolerant helpers DispatchLine now uses (id extraction, error-message extraction), the
// emit-side JSON escaping, the buffered-recv line splitter, and the nvidia-smi "[N/A]"
// -> JSON null guard. Standalone: MATADOR_STRATUM_PARSE_HELPERS_ONLY compiles just the
// pure helper block (no Config/Stats/mlog deps); links only vendored univalue.
//   ./run-tests.sh   (exit 0 = pass)
// run-tests: -Icore/vendor/univalue/include core/vendor/univalue/lib/univalue.cpp core/vendor/univalue/lib/univalue_read.cpp core/vendor/univalue/lib/univalue_get.cpp core/vendor/univalue/lib/univalue_write.cpp
#define MATADOR_STRATUM_PARSE_HELPERS_ONLY
#include "../miner/stratum_client.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

static void ok(bool cond, const char* label)
{
    if (cond) std::printf("  ok   %s\n", label);
    else { std::printf("  FAIL %s\n", label); ++g_fail; }
}

static UniValue parse(const char* json)
{
    UniValue v;
    if (!v.read(json)) { std::printf("  FAIL test-json unparseable: %s\n", json); ++g_fail; }
    return v;
}

// ---- StratumIdToUint: tolerate every id shape a pool has been seen to send ----
static void id_accepts(const char* json, uint64_t want, const char* label)
{
    const UniValue v = parse(json);
    uint64_t out = 777;
    bool r = false;
    try { r = StratumIdToUint(v["id"], out); }
    catch (...) { std::printf("  FAIL %-44s -> THREW\n", label); ++g_fail; return; }
    if (r && out == want) std::printf("  ok   %-44s -> %llu\n", label, static_cast<unsigned long long>(out));
    else { std::printf("  FAIL %-44s -> ok=%d out=%llu (want %llu)\n", label, r ? 1 : 0,
                       static_cast<unsigned long long>(out), static_cast<unsigned long long>(want)); ++g_fail; }
}

static void id_rejects(const char* json, const char* label)
{
    const UniValue v = parse(json);
    uint64_t out = 777;
    bool r = true;
    try { r = StratumIdToUint(v["id"], out); }
    catch (...) { std::printf("  FAIL %-44s -> THREW (must never throw)\n", label); ++g_fail; return; }
    if (!r) std::printf("  ok   %-44s -> rejected, no throw\n", label);
    else { std::printf("  FAIL %-44s -> accepted out=%llu (want reject)\n", label,
                       static_cast<unsigned long long>(out)); ++g_fail; }
}

// ---- StratumErrorMessage: never throws, best-effort human text ----
static void err_msg(const char* json, const char* want, const char* label)
{
    const UniValue v = parse(json);
    std::string got;
    try { got = StratumErrorMessage(v["error"]); }
    catch (...) { std::printf("  FAIL %-44s -> THREW (must never throw)\n", label); ++g_fail; return; }
    if (got == want) std::printf("  ok   %-44s -> \"%s\"\n", label, got.c_str());
    else { std::printf("  FAIL %-44s -> \"%s\" (want \"%s\")\n", label, got.c_str(), want); ++g_fail; }
}

int main()
{
    std::printf("[stratum_dispatch_test]\n");

    // ---- response-id tolerance (the reader-thread crash class) ----
    id_accepts("{\"id\":3,\"result\":true}", 3, "id int");
    id_accepts("{\"id\":\"3\",\"result\":true}", 3, "id STRING \"3\" (login pools)");
    id_accepts("{\"id\":10.0,\"result\":true}", 10, "id FLOAT 10.0 (getInt<> throws on this)");
    id_accepts("{\"id\":12,\"error\":[21,\"x\"]}", 12, "id int with error");
    id_rejects("{\"id\":{},\"result\":true}", "id object");
    id_rejects("{\"id\":[1],\"result\":true}", "id array");
    id_rejects("{\"id\":\"abc\",\"result\":true}", "id non-numeric string");
    id_rejects("{\"id\":\"12x\",\"result\":true}", "id trailing garbage string");
    id_rejects("{\"id\":\"\",\"result\":true}", "id empty string");
    id_rejects("{\"id\":true,\"result\":true}", "id bool");

    // ---- error-message tolerance ----
    err_msg("{\"id\":4,\"error\":[21,\"low difficulty\"]}", "low difficulty", "error [code,\"msg\"]");
    err_msg("{\"id\":4,\"error\":{\"code\":-1,\"message\":\"stale share\"}}", "stale share", "error {message:str}");
    err_msg("{\"id\":4,\"error\":\"broke\"}", "broke", "error bare string");
    // non-string message: old code get_str()-THREW here -> reader death; now raw JSON
    err_msg("{\"id\":4,\"error\":{\"message\":42}}", "{\"message\":42}", "error {message:42} (non-string)");
    err_msg("{\"id\":4,\"error\":[21,42]}", "[21,42]", "error [code,42] (non-string)");
    err_msg("{\"id\":4,\"error\":7}", "7", "error bare number");

    // ---- AsUint: lenient num/hex/dec, but garbage still THROWS by design ----
    // (a garbage notify field must fail the WHOLE job parse -- keep mining the old
    //  job -- rather than silently become 0 and corrupt the digest into rejects)
    {
        const UniValue v = parse("{\"a\":42,\"b\":\"0x1f\",\"c\":\"123\",\"d\":\"abc\",\"e\":true}");
        ok(AsUint(v["a"]) == 42, "AsUint(42) == 42");
        ok(AsUint(v["b"]) == 31, "AsUint(\"0x1f\") == 31");
        ok(AsUint(v["c"]) == 123, "AsUint(\"123\") == 123");
        ok(AsUint(v["e"]) == 0, "AsUint(bool) == 0");
        bool threw = false;
        try { (void)AsUint(v["d"]); } catch (const std::exception&) { threw = true; }
        ok(threw, "AsUint(\"abc\") throws (callers catch; job dropped loudly)");
    }

    // ---- FeedRecvBytes: buffered recv line-splitting (must mirror the old 1-byte loop) ----
    {
        auto feed = [](std::string& carry, const char* data, std::vector<std::string>& out) {
            FeedRecvBytes(carry, data, std::strlen(data), [&](const std::string& line) { out.push_back(line); });
        };
        std::string carry;
        std::vector<std::string> lines;
        feed(carry, "{\"a\":1}\n{\"b\":2}\n", lines);
        ok(lines.size() == 2 && lines[0] == "{\"a\":1}" && lines[1] == "{\"b\":2}" && carry.empty(),
           "two complete lines in one chunk");

        carry.clear(); lines.clear();
        feed(carry, "{\"par", lines);
        ok(lines.empty() && carry == "{\"par", "partial line carried across recvs");
        feed(carry, "tial\":1}\nrest", lines);
        ok(lines.size() == 1 && lines[0] == "{\"partial\":1}" && carry == "rest",
           "carried line completed + next partial retained");

        carry.clear(); lines.clear();
        feed(carry, "x\r\ny\r\n", lines);
        ok(lines.size() == 2 && lines[0] == "x" && lines[1] == "y", "CRLF handled ('\\r' stripped)");

        carry.clear(); lines.clear();
        feed(carry, "a\rb\n", lines);
        ok(lines.size() == 1 && lines[0] == "ab", "lone '\\r' stripped mid-line (old-loop semantics)");

        carry.clear(); lines.clear();
        feed(carry, "\n\n\n", lines);
        ok(lines.empty() && carry.empty(), "blank lines skipped");

        carry.clear(); lines.clear();
        feed(carry, "one-byte-at-a-time", lines);
        for (const char* p = "\nnext\n"; *p != '\0'; ++p) {
            const char one[2] = {*p, '\0'};
            feed(carry, one, lines);
        }
        ok(lines.size() == 2 && lines[0] == "one-byte-at-a-time" && lines[1] == "next",
           "degenerate 1-byte chunks behave like the old loop");
    }

    // ---- CsvNumberOrNull: nvidia-smi "[N/A]" columns must emit JSON null ----
    {
        ok(CsvNumberOrNull("42") == "42", "CsvNumberOrNull(\"42\") == 42");
        ok(CsvNumberOrNull("42.5") == "42.5", "CsvNumberOrNull(\"42.5\") == 42.5");
        ok(CsvNumberOrNull("550.75") == "550.75", "CsvNumberOrNull power watts passthrough");
        ok(CsvNumberOrNull("[N/A]") == "null", "CsvNumberOrNull(\"[N/A]\") == null");
        ok(CsvNumberOrNull("N/A") == "null", "CsvNumberOrNull(\"N/A\") == null");
        ok(CsvNumberOrNull("") == "null", "CsvNumberOrNull(\"\") == null");
        ok(CsvNumberOrNull("12abc") == "null", "CsvNumberOrNull(\"12abc\") == null");
        ok(CsvNumberOrNull("nan") == "null", "CsvNumberOrNull(\"nan\") == null (non-finite)");
    }

    // ---- JsonEscapeMinimal: emit-side JSON hygiene (user/pass/worker) ----
    {
        ok(JsonEscapeMinimal("plain.worker") == "plain.worker", "escape passthrough");
        ok(JsonEscapeMinimal("pa\"ss") == "pa\\\"ss", "escape embedded quote");
        ok(JsonEscapeMinimal("a\\b") == "a\\\\b", "escape backslash");
        ok(JsonEscapeMinimal("\"") == "\\\"", "escape lone quote");
        // round-trip through the real JSON parser: an escaped hostile password must
        // land as ONE string value, not break out of the frame
        UniValue v;
        const std::string frame = "{\"pass\":\"" + JsonEscapeMinimal("x\",\"evil\":\"1") + "\"}";
        ok(v.read(frame) && v["pass"].isStr() && v["pass"].get_str() == "x\",\"evil\":\"1" && !v.exists("evil"),
           "hostile password cannot break out of its JSON string");
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
