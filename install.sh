#!/usr/bin/env bash
# install.sh - download, verify (sha256), smoke-test, and install matador-miner.
#
#   curl -fsSL https://raw.githubusercontent.com/vanities/matador-miner/main/install.sh | bash
#
# Env options:
#   VERSION=v0.3.0          install a specific tag (default: newest release incl. prereleases)
#   PREFIX=$HOME/.local/bin install dir (default: /usr/local/bin via sudo, else ~/.local/bin)
#   REPO=owner/name         override the source repo (default: vanities/matador-miner)
set -euo pipefail

REPO="${REPO:-vanities/matador-miner}"
log(){ printf '[install] %s\n' "$*" >&2; }
die(){ printf '[install] ERROR: %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null || die "curl is required"
if   command -v sha256sum >/dev/null; then SHACHK="sha256sum"
elif command -v shasum    >/dev/null; then SHACHK="shasum -a 256"
else die "need sha256sum or shasum to verify the download"; fi

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) asset_pattern='linux-x86_64' ;;
  Darwin-*) die "macOS cannot mine BTX. ENC_RC needs NVIDIA tensor cores (Ampere or newer)." ;;
  *) die "unsupported platform $(uname -s)-$(uname -m); expected Linux x86_64" ;;
esac

# GPU-arch gate. ENC_RC (v4) is a tensor-core workload, so compute capability >= 8.0
# (Ampere or newer) is a hard requirement, not a preference: an older card cannot mine it at
# any speed. Stop here with a useful message instead of installing a binary that will never
# find a share. SKIP_GPU_CHECK=1 bypasses (e.g. installing on a box with no GPU attached yet).
if [ "${SKIP_GPU_CHECK:-}" != 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
  maxcc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -g | tail -1)"
  if [ -n "$maxcc" ] && awk "BEGIN{exit !($maxcc < 8.0)}" 2>/dev/null; then
    die "GPU compute capability $maxcc is below 8.0 (Pascal/Volta/Turing). BTX needs Ampere or
     newer tensor cores; this card cannot mine the chain at any speed."
  fi
fi

if [ -n "${VERSION:-}" ]; then
  api="https://api.github.com/repos/$REPO/releases/tags/$VERSION"
else
  # /releases is newest-first and includes prereleases. /releases/latest skips prereleases.
  api="https://api.github.com/repos/$REPO/releases"
fi

log "resolving release: $api"
json="$(curl -fsSL "$api")" || die "cannot reach the GitHub API (is a release published yet?)"
# `|| true`: no match must fall through to the helpful die, not exit silently via set -e/pipefail.
url="$(printf '%s' "$json" | grep -oE '"browser_download_url": *"[^"]+'"$asset_pattern"'"' | cut -d'"' -f4 | head -1 || true)"
[ -n "$url" ] || die "no $asset_pattern binary asset found in that release"
asset="$(basename "$url")"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
log "downloading $asset"
curl -fsSL "$url"        -o "$tmp/$asset"
curl -fsSL "$url.sha256" -o "$tmp/$asset.sha256" || die "release is missing the .sha256 checksum asset"

log "verifying checksum"
( cd "$tmp" && $SHACHK -c "$asset.sha256" >/dev/null ) || die "CHECKSUM MISMATCH - refusing to install"
log "checksum OK"
chmod +x "$tmp/$asset"

log "smoke-testing binary"
"$tmp/$asset" --help >/dev/null || die "downloaded binary failed --help smoke test"
if [ "$asset_pattern" = linux-x86_64 ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/[install] nvidia: /' >&2 || true
  else
    log "nvidia-smi not found; skipping NVIDIA driver visibility check"
  fi
fi

# Pick an install dir: explicit PREFIX, else /usr/local/bin (sudo), else ~/.local/bin.
if [ -n "${PREFIX:-}" ]; then
  mkdir -p "$PREFIX"; mv "$tmp/$asset" "$PREFIX/matador-miner"; dst="$PREFIX"
elif [ -w /usr/local/bin ]; then
  mv "$tmp/$asset" /usr/local/bin/matador-miner; dst="/usr/local/bin"
elif command -v sudo >/dev/null; then
  log "installing to /usr/local/bin (sudo)"
  sudo mv "$tmp/$asset" /usr/local/bin/matador-miner; dst="/usr/local/bin"
else
  mkdir -p "$HOME/.local/bin"; mv "$tmp/$asset" "$HOME/.local/bin/matador-miner"; dst="$HOME/.local/bin"
fi

log "installed -> $dst/matador-miner"
case ":$PATH:" in *":$dst:"*) ;; *) log "NOTE: add $dst to your PATH";; esac
log "next: matador-miner --help"
log "pool example: matador-miner --mode pool --pool stratum+tcp://stratum.minebtx.com:3333 --pool stratum+tcp://stratum.btxbyronbay.com:3335 --worker rig1 --payoutaddress btx1zcf4z36asua8ylchysphgwfgyfr8267vvznth826epden7lar4fnqvy9gzv --api --api-port 4060"
