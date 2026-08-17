#!/usr/bin/env bash
# Install the git pre-commit hook that runs the clean-stack unit tests (run-tests.sh).
# Run once per clone:  ./clean-stack/install-hooks.sh
# Idempotent; backs up any existing hook to pre-commit.bak. Bypass a single commit with `git commit -n`.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOK="$ROOT/.git/hooks/pre-commit"

if [ -e "$HOOK" ] && ! grep -q "clean-stack/run-tests.sh" "$HOOK" 2>/dev/null; then
    cp "$HOOK" "$HOOK.bak"
    echo "[install-hooks] backed up existing hook -> pre-commit.bak"
fi

cat > "$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
# matador-src pre-commit: run the standalone clean-stack unit tests when clean-stack/ is touched.
# Auto-installed by clean-stack/install-hooks.sh. Bypass once with `git commit -n`.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
TESTS="$ROOT/clean-stack/run-tests.sh"

# Only gate commits that actually touch clean-stack (keeps doc/config commits fast).
if ! git diff --cached --name-only | grep -q '^clean-stack/'; then
    exit 0
fi
[ -x "$TESTS" ] || { echo "[pre-commit] $TESTS missing/not executable -- skipping" >&2; exit 0; }
if ! command -v "${CXX:-c++}" >/dev/null 2>&1; then
    echo "[pre-commit] no C++ compiler (${CXX:-c++}) -- skipping unit tests" >&2
    exit 0
fi

echo "[pre-commit] clean-stack changed -> running unit tests" >&2
"$TESTS"
HOOK_EOF

chmod +x "$HOOK"
echo "[install-hooks] installed $HOOK"
echo "[install-hooks] it runs clean-stack/run-tests.sh on commits touching clean-stack/ (bypass: git commit -n)"
