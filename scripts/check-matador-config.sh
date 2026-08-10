#!/usr/bin/env bash
# check-matador-config.sh — fast tests for standalone matador-miner config support.
#
# This stays Docker/GPU/network-free. The proprietary matador-miner source lives
# under ignored private/, so CI/public clones skip the source-specific assertions;
# local development runs them automatically when the source is present.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

note() { printf '%s\n' "$*" >&2; }
now_ms() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
  else
    date +%s000
  fi
}

start_ms=$(now_ms)
note "[check-matador-config] validating config fixtures"
python3 - <<'PY'
import json
from pathlib import Path
# CUDA is the only backend since ENC_RC (v4) activated: the AMD sidecar and the
# Metal engine both solved the retired v3 algorithm.
expected_backends = {
    'docs/config.example.nvidia.json': 'cuda',
}
example_payout = 'btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv'
shibs_pool_url = 'stratum+tcp://stratum.minebtx.com:3333'
required = {'mode', 'pools', 'worker', 'payoutaddress', 'chain', 'backend'}
for path, backend in expected_backends.items():
    p = Path(path)
    data = json.loads(p.read_text())
    missing = sorted(required - set(data))
    assert not missing, f"{p} missing keys: {missing}"
    assert data['mode'] in {'solo', 'pool'}, p
    assert data['backend'] == backend, f"{p} backend={data['backend']} expected {backend}"
    assert data['gpus'] == [0], f"{p}: expected editable single-GPU default"
    assert data['payoutaddress'] == example_payout, f"{p}: expected shared example payout address"
    assert isinstance(data['pools'], list) and data['pools'], f'{p}: pools must be a non-empty list'
    assert data['pools'][0].get('label') == 'minebtx', f'{p}: first pool should be labeled minebtx'
    assert data['pools'][0].get('url') == shibs_pool_url, f'{p}: first pool should be the minebtx stratum'
    for i, pool in enumerate(data['pools']):
        assert isinstance(pool, (str, dict)), f'{p}: pools[{i}] must be string or object'
        if isinstance(pool, dict):
            assert 'url' in pool or ('host' in pool and 'port' in pool), f'{p}: pools[{i}] needs url or host+port'
    assert isinstance(data['overlap'], bool), p
    assert isinstance(data['update_check'], bool), p
    assert isinstance(data['auto_update'], bool), p
    assert isinstance(data['api'], dict), f'{p}: api must be an object'
    assert data['api']['listen'] == '127.0.0.1', p
    assert isinstance(data['api']['port'], int) and data['api']['port'] > 0, p
    assert isinstance(data['watchdog'], dict), f'{p}: watchdog must be an object'
    assert data['watchdog']['enabled'] is True, p
    assert data['watchdog']['check_s'] > 0, p
    assert data['watchdog']['reject_streak'] >= 0, p
    assert data['watchdog']['no_share_s'] >= 0, p
    assert isinstance(data['thermal'], dict), f'{p}: thermal must be an object'
    assert data['thermal']['enabled'] is True, p
    assert data['thermal']['warn_temp_c'] >= 0, p
    assert data['thermal']['critical_temp_c'] >= 0, p
    assert data['thermal']['warn_power_w'] >= 0, p
PY

elapsed=$(( $(now_ms) - start_ms ))
note "[check-matador-config] OK in ${elapsed}ms"
