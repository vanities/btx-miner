// Unit test for the /summary GPU JSON assembly (clean-stack/miner/status_api.h).
// /summary and the watchdog thermal tick now read util/power/temp through the
// fork-free NVML path (gpu_telemetry.h) first -- popen("nvidia-smi") forked the
// whole CUDA-mapped miner per poll -- with nvidia-smi kept as the fallback and
// the one-time gpu_uuid source. The hub/matlog parse this JSON, so this pins
// the EXACT shape both sources must produce: same keys in the same order,
// number-or-null semantics, string escaping, and the NVML->metric mapping edge
// cases (power/temp 0 -> null, mirroring nvidia-smi "[N/A]"; util 0 stays a
// real number -- an idle/gated GPU genuinely reads 0%). Standalone:
// MATADOR_STATUS_API_GPU_JSON_ONLY compiles just the pure helper block of
// status_api.h (no Config/Stats/mlog/popen deps), same idiom as
// MATADOR_CONFIG_PARSE_HELPERS_ONLY.
#define MATADOR_STATUS_API_GPU_JSON_ONLY
#include "../miner/status_api.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_fail = 0;

static void ok(bool cond, const char* label)
{
    if (cond) std::printf("  ok   %s\n", label);
    else { std::printf("  FAIL %s\n", label); ++g_fail; }
}

static void eq(const std::string& got, const std::string& want, const char* label)
{
    if (got == want) std::printf("  ok   %s\n", label);
    else {
        std::printf("  FAIL %s\n       got : %s\n       want: %s\n", label, got.c_str(), want.c_str());
        ++g_fail;
    }
}

int main()
{
    std::printf("[status_api_gpu_json_test]\n");

    // ---- GpuRuntimeJson: the pinned /summary gpu_runtime shape ----
    {
        GpuRuntimeMetric m;
        m.uuid = "GPU-abc123";
        m.vendor = "nvidia";
        m.util_pct = 98.0;   m.has_util = true;
        m.power_w = 552.34;  m.has_power = true;   // nvidia-smi fallback: fractional watts survive
        m.temp_c = 62.0;     m.has_temp = true;
        eq(GpuRuntimeJson({m}),
           "[{\"gpu_uuid\":\"GPU-abc123\",\"vendor\":\"nvidia\","
           "\"util_pct\":98,\"power_w\":552.34,\"temp_c\":62}]",
           "full row: keys, order, numbers");
    }
    {
        GpuRuntimeMetric m;   // defaults: every has_ flag false
        m.uuid = "GPU-x";
        m.vendor = "nvidia";
        eq(GpuRuntimeJson({m}),
           "[{\"gpu_uuid\":\"GPU-x\",\"vendor\":\"nvidia\","
           "\"util_pct\":null,\"power_w\":null,\"temp_c\":null}]",
           "absent sensors: null, never 0/NaN/[N/A]");
    }
    eq(GpuRuntimeJson({}), "[]", "no GPUs (non-NVIDIA/absent box): empty array");
    {
        GpuRuntimeMetric a, b;
        a.uuid = "GPU-0"; a.vendor = "nvidia"; a.util_pct = 1; a.has_util = true;
        b.uuid = "GPU-1"; b.vendor = "nvidia"; b.temp_c = 55; b.has_temp = true;
        eq(GpuRuntimeJson({a, b}),
           "[{\"gpu_uuid\":\"GPU-0\",\"vendor\":\"nvidia\",\"util_pct\":1,\"power_w\":null,\"temp_c\":null},"
           "{\"gpu_uuid\":\"GPU-1\",\"vendor\":\"nvidia\",\"util_pct\":null,\"power_w\":null,\"temp_c\":55}]",
           "two rows joined with a comma (multi-GPU fallback path)");
    }
    {
        GpuRuntimeMetric m;   // hostile uuid must be escaped, not break the JSON
        m.uuid = "GPU-\"quote\\slash";
        m.vendor = "nvidia";
        eq(GpuRuntimeJson({m}),
           "[{\"gpu_uuid\":\"GPU-\\\"quote\\\\slash\",\"vendor\":\"nvidia\","
           "\"util_pct\":null,\"power_w\":null,\"temp_c\":null}]",
           "uuid escaping via JsonString");
    }

    // ---- MetricFromNvmlTelemetry: NVML sample -> metric row mapping ----
    {
        GpuTelemetry t;   // healthy rig sample (NVML integers: mW->W truncates)
        t.ok = true; t.temp_c = 62; t.pow_w = 552; t.util_pct = 98;
        t.sm_mhz = 2900; t.mem_mhz = 14001; t.fan_pct = 60;   // carried by NVML, NOT emitted
        const GpuRuntimeMetric m = MetricFromNvmlTelemetry(t, "GPU-abc123");
        ok(m.has_util && m.has_power && m.has_temp, "healthy NVML sample: all fields present");
        eq(GpuRuntimeJson({m}),
           "[{\"gpu_uuid\":\"GPU-abc123\",\"vendor\":\"nvidia\","
           "\"util_pct\":98,\"power_w\":552,\"temp_c\":62}]",
           "NVML row: same keys/shape as the nvidia-smi row (clocks/fan NOT added)");
    }
    {
        GpuTelemetry t;   // sensors NVML cannot read stay 0 -> must map to null,
        t.ok = true;      // exactly like nvidia-smi printing [N/A]; util 0 is REAL
        const GpuRuntimeMetric m = MetricFromNvmlTelemetry(t, "GPU-idle");
        eq(GpuRuntimeJson({m}),
           "[{\"gpu_uuid\":\"GPU-idle\",\"vendor\":\"nvidia\","
           "\"util_pct\":0,\"power_w\":null,\"temp_c\":null}]",
           "NVML zeros: power/temp -> null, util 0 stays the number 0");
    }
    {
        GpuTelemetry t;   // !ok (dlopen/init/device failed): defensive all-null row --
        t.temp_c = 62;    // callers skip !ok and fall back to nvidia-smi, but the
        t.pow_w = 552;    // mapper must never invent numbers from a failed probe
        t.util_pct = 98;
        const GpuRuntimeMetric m = MetricFromNvmlTelemetry(t, "GPU-dead");
        ok(!m.has_util && !m.has_power && !m.has_temp, "!ok telemetry maps to all-null");
    }

    // ---- nvidia-smi fallback CSV bits (shared parse helpers) ----
    {
        const auto col = SplitCsvTrim("GPU-7509xyz, 98, 552.34, 62\n");
        ok(col.size() == 4 && col[0] == "GPU-7509xyz" && col[1] == "98" &&
           col[2] == "552.34" && col[3] == "62",
           "SplitCsvTrim: nvidia-smi csv row -> trimmed columns");
    }
    {
        double d = 0.0;
        ok(!ParseDoubleFinite("[N/A]", d), "ParseDoubleFinite rejects \"[N/A]\" (-> null)");
        ok(!ParseDoubleFinite("", d), "ParseDoubleFinite rejects empty (-> null)");
        ok(!ParseDoubleFinite("nan", d), "ParseDoubleFinite rejects non-finite (-> null)");
        ok(ParseDoubleFinite("552.34", d) && d == 552.34, "ParseDoubleFinite accepts \"552.34\"");
        ok(ParseDoubleFinite("62 ", d) && d == 62.0, "ParseDoubleFinite accepts trailing space");
    }
    eq(JsonNumberOrNull(false, 123.0), "null", "JsonNumberOrNull: absent -> null");
    eq(JsonNumberOrNull(true, 0.0), "0", "JsonNumberOrNull: present 0 -> 0");
    ok(ContainsLower("Average Graphics Package Power (W)", "power") &&
       !ContainsLower("GPU use (%)", "temp"),
       "ContainsLower: rocm-smi header matching");

    // ---- fallback safety smoke: the NVML probe must be crash-free anywhere ----
    // On a no-NVIDIA box (this Mac, CI) dlopen fails and the probe must return a
    // zeroed !ok struct -- the exact condition that routes QueryGpuRuntimeMetrics
    // to the popen fallback. On an NVIDIA rig ok=true and the check is vacuous.
    {
        const GpuTelemetry live = QueryGpuTelemetry();
        ok(live.ok || (live.temp_c == 0 && live.pow_w == 0 && live.util_pct == 0 &&
                       live.sm_mhz == 0 && live.mem_mhz == 0 && live.fan_pct == 0),
           "QueryGpuTelemetry: crash-free; !ok is a zeroed struct (fallback trigger)");
    }

    if (g_fail == 0) std::printf("ALL PASS\n");
    else std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
