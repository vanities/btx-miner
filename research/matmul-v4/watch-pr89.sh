#!/usr/bin/env bash
# PR#89 change classifier — turns "numair pushed something" into a prioritized "does it matter to
# us, and what do we do" report in one command. Runs the fetch+diff+classify on the pc btx checkout
# (the only box with it). Pin the last-seen SHA in .pr89-last-seen so re-runs only report deltas.
#
#   research/matmul-v4/watch-pr89.sh            # diff since last-seen, classify, DON'T advance pin
#   research/matmul-v4/watch-pr89.sh --accept   # ...and advance the pin to the new head
#
# Priority signals (what we react to), highest first:
#   ACTIVATION  a nMatMul*Height went finite  -> RC/coupled GOING LIVE -> build competitive solver NOW
#   GOLDEN      a 64-hex golden changed/added -> byte-exactness moved   -> update/build the oracle
#   PROFILE     V3 dims (1536/M=128/24/packed) changed -> RE-MEASURE economics
#   ARBITER     gkr_arbiter flipped true       -> proof now gates consensus
#   ECONOMICS   code/doc touches batch/residency/measurement -> may answer our finding
#   GKR-ONLY    only gkr/fri/logup churn       -> crypto hardening, keep monitoring
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIN="$HERE/.pr89-last-seen"
ACCEPT=0; [ "${1:-}" = "--accept" ] && ACCEPT=1
LAST="$(cat "$PIN" 2>/dev/null || echo '')"
SSH="ssh -o ControlMaster=no -o ControlPath=none pc"

NOW="$($SSH 'cd ~/btx-pr89 && git fetch -q origin "refs/pull/89/head:refs/remotes/pr89-head" --force 2>/dev/null; git rev-parse --short refs/remotes/pr89-head' 2>/dev/null)"
[ -z "$NOW" ] && { echo "FAIL: could not fetch PR#89 head (pc reachable? ~/btx-pr89 present?)"; exit 1; }
echo "PR#89 head: $NOW   (last-seen: ${LAST:-<none>})"
if [ "$NOW" = "$LAST" ]; then echo "  no change since last-seen."; exit 0; fi

# everything below runs against the fetched head on pc
$SSH "cd ~/btx-pr89 && bash -s -- '${LAST:-}' '$NOW'" <<'REMOTE'
LAST="$1"; NOW="$2"; R=refs/remotes/pr89-head
RANGE=""; [ -n "$LAST" ] && RANGE="$LAST..$R"
n(){ git rev-list --count $RANGE 2>/dev/null || echo '?'; }
echo "  commits in range: $(n)"
echo
CHANGED="$( [ -n "$LAST" ] && git diff --name-only $RANGE 2>/dev/null || echo '(no baseline; full-tree scan)')"

flag(){ printf '  [%-9s] %s\n' "$1" "$2"; }

# --- ACTIVATION (critical) -------------------------------------------------------------------
# The real signal is the MAINNET/TESTNET guard asserts, NOT any finite height (regtest sets finite
# heights on purpose for local testing -- those are noise). Inert == the asserts that force
# nMatMulRCHeight/nMatMulRCCoupledHeight == INT32_MAX on public nets are still present.
GUARD=$(git show $R:src/kernel/chainparams.cpp 2>/dev/null | grep -cE 'nMatMulRC(Coupled)?Height == std::numeric_limits<int32_t>::max')
if [ "${GUARD:-0}" -ge 2 ]; then flag "ok" "mainnet/testnet guard asserts intact -> RC + coupled still INERT on public nets"
else flag "ACTIVATION" "!!! the INT32_MAX guard assert for RC/coupled heights is WEAKENED/GONE (found $GUARD) -> RC MAY BE GOING LIVE ON A PUBLIC NET. VERIFY chainparams NOW."; fi

# v4.6 single-switch surface: the deploy recipe (chainparams comment) says the owner flips, in ONE
# release: nMatMulV4Height + nMatMulRCHeight=kRCDatacenterActivationHeight + ASERT 16422/1027 + the
# no-inversion ratification gate. Watch each leg -- any ONE moving is the earliest tripwire.
CP=$(git show $R:src/kernel/chainparams.cpp 2>/dev/null)
echo "$CP" | grep -q 'BTX_MATMUL_NO_INVERSION_GATE_RATIFIED' || \
  flag "ACTIVATION" "!!! no-inversion RATIFICATION GATE symbol vanished from chainparams -> recheck the fail-closed construction path"
if echo "$CP" | grep -E 'nMatMulRCHeight *= *kRCDatacenterActivationHeight' >/dev/null; then
  flag "ACTIVATION" "!!! mainnet nMatMulRCHeight ASSIGNED the datacenter activation height -> RC IS GOING LIVE"
fi
# baseline 1 = the REGTEST profile-2 block (line ~1785 at a4bdefb); a SECOND assignment site
# means a public-net chainparams block picked it up.
MAINNET_RESCALE=$(echo "$CP" | grep -cE 'nMatMulRCAsertRescaleNum *= *(16422|kRCDatacenterAsertRescaleNum)')
[ "${MAINNET_RESCALE:-0}" -gt 1 ] && \
  flag "ACTIVATION" "!!! ASERT rescale 16422/1027 assigned at $MAINNET_RESCALE sites (baseline 1 = regtest) -> datacenter cutover staged on a public net"

# --- GOLDENS (high) --------------------------------------------------------------------------
# our tracked goldens; report if any is MISSING (=changed) or if NEW 64-hex goldens appear.
for g in 5b1bff3c835b1c8e 7a7ce1065c7881aa; do
  if git grep -q "$g" $R -- 'src/matmul/*.h' 'src/test/matmul_v4_rc*tests.cpp' 2>/dev/null; then :; else
    flag "GOLDEN" "!!! tracked golden ${g}... NO LONGER PRESENT -> re-pinned; our oracle may be stale"; fi
done
if [ -n "$LAST" ]; then
  NEWG=$(git diff $RANGE -- 'src/test/matmul_v4_rc*tests.cpp' 2>/dev/null | grep -E '^\+' | grep -oE '"[0-9a-f]{64}"' | sort -u | head -6)
  [ -n "$NEWG" ] && { flag "GOLDEN" "new/changed 64-hex golden(s) in coupled/rc tests -> byte-exactness moved:"; echo "$NEWG" | sed 's/^/        /'; }
fi

# --- PROFILE (high) --------------------------------------------------------------------------
if [ -n "$LAST" ] && echo "$CHANGED" | grep -qE 'matmul_v4_rc_datacenter|matmul_v4_rc_coupled\.(h|cpp)|accel_policy'; then
  PDIFF=$(git diff $RANGE -- src/matmul/matmul_v4_rc_datacenter.h src/matmul/matmul_v4_rc_coupled.h 2>/dev/null | grep -E '^[-+].*(bank_pages|rows_per_lobe|pages_per_barrier|PackedBankTarget|lobe_width|barriers|exchange_rows|1536|kRCCoup)' | grep -vE '^\+\+\+|^---' | head -10)
  [ -n "$PDIFF" ] && { flag "PROFILE" "V3 profile/dims changed -> RE-MEASURE economics:"; echo "$PDIFF" | sed 's/^/        /'; }
fi

# --- ARBITER (high) --------------------------------------------------------------------------
ARB=$(git show $R:src/matmul/matmul_v4_rc_datacenter.h 2>/dev/null | grep -E 'gkr_arbiter\{' | head -1)
echo "$ARB" | grep -q 'false' && flag "ok" "gkr_arbiter still false (hard-disabled)" || flag "ARBITER" "!!! gkr_arbiter no longer false: $ARB"

# --- ECONOMICS (medium) — did they touch our lane / respond to the finding? ------------------
if [ -n "$LAST" ]; then
  ECON=$(git diff --name-only $RANGE 2>/dev/null | grep -iE 'adversar|batch|econ|5090|b200|measurement|residency|packed-bank' | head -6)
  [ -n "$ECON" ] && { flag "ECONOMICS" "touched files in our lane (batch/residency/measurement):"; echo "$ECON" | sed 's/^/        /'; }
fi

# --- classify the bulk -----------------------------------------------------------------------
if [ -n "$LAST" ]; then
  TOTAL=$(echo "$CHANGED" | grep -c .); GKR=$(echo "$CHANGED" | grep -cE 'gkr|fri|logup|air|sumcheck|Fp[23]')
  if [ "$TOTAL" -gt 0 ] && [ "$GKR" -ge $((TOTAL*3/4)) ]; then
    flag "GKR-ONLY" "$GKR/$TOTAL changed files are GKR/proof-system -> crypto hardening, LOW priority for us"
  fi
fi
echo
echo "  files changed: $(echo "$CHANGED" | grep -c . 2>/dev/null || echo '?')"
REMOTE

echo
if [ "$ACCEPT" = 1 ]; then echo "$NOW" > "$PIN"; echo "pinned last-seen -> $NOW"; else
  echo "(pin NOT advanced; re-run with --accept once you've reviewed)"; fi
