#!/usr/bin/env bash
# build-hiveos-pkg.sh -- assemble the HiveOS custom miner package for a release.
#
#   VER=v0.8.25 bash build-hiveos-pkg.sh
#
# Wraps the ALREADY BUILT release binary with the HiveOS
# integration scripts from clean-stack/hiveos/ into:
#
#   dist/matador-miner-<ver>.tar.gz        (+ .sha256)
#
# containing  matador-miner/{h-manifest.conf,h-config.sh,h-run.sh,h-stats.sh,
#                            matador-miner,README.md,...}
#
# The binary is taken from dist/matador-miner-$VER-linux-x86_64 when
# present, otherwise downloaded from the GitHub release. Each is verified
# against its .sha256 before packaging.
#
# NOTE the asset name matador-miner-<ver-without-v>.tar.gz is deliberate:
#  - HiveOS derives the miner folder name by stripping "-<version>" from the
#    archive name, so it must resolve to exactly "matador-miner".
#  - It must NOT contain "linux-x86_64" anywhere,
#    or deployed auto-updaters would consider it a binary asset.
set -euo pipefail

S="${S:-$(cd "$(dirname "$0")" && pwd)}"
VER="${VER:?set VER, e.g. VER=v0.8.25}"
REPO="${REPO:-vanities/matador-miner}"
PKGVER="${VER#v}"

HIVE_SRC="$S/hiveos"
EXAMPLE_SRC="${EXAMPLE_SRC:-$S/../hiveos}"   # repo-root hiveos/: importable flight sheets
DIST="$S/dist"
STAGE="$S/build-hiveos/matador-miner"
OUT="$DIST/matador-miner-$PKGVER.tar.gz"

case "$OUT" in
    *linux-x86_64*)
        echo "FATAL: package name '$OUT' collides with the updater's binary asset matching" >&2
        exit 1;;
esac

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# Fetch (if needed) and verify one release binary. $1 = asset name, $2 = staged name.
fetch_verify() {
    local asset="$1" dest="$2"
    if [[ ! -f "$DIST/$asset" || ! -f "$DIST/$asset.sha256" ]]; then
        echo "[hiveos-pkg] $asset not in dist/; downloading from release $VER"
        gh release download "$VER" --repo "$REPO" --pattern "$asset" --pattern "$asset.sha256" --dir "$DIST" --clobber
    fi
    local want got
    want=$(awk '{print $1}' < "$DIST/$asset.sha256")
    got=$(sha256 "$DIST/$asset")
    if [[ "$want" != "$got" ]]; then
        echo "FATAL: sha256 mismatch for $asset (want $want got $got)" >&2
        exit 1
    fi
    cp "$DIST/$asset" "$STAGE/$dest"
    chmod 755 "$STAGE/$dest"
    echo "[hiveos-pkg] verified $asset -> $dest ($got)"
}

rm -rf "$S/build-hiveos"
mkdir -p "$STAGE" "$DIST"

for f in h-config.sh h-run.sh h-stats.sh; do
    cp "$HIVE_SRC/$f" "$STAGE/$f"
    chmod 755 "$STAGE/$f"
done
sed "s/^CUSTOM_MINER_VER=.*/CUSTOM_MINER_VER=$PKGVER/" "$HIVE_SRC/h-manifest.conf" > "$STAGE/h-manifest.conf"
grep -q "CUSTOM_MINER_VER=$PKGVER" "$STAGE/h-manifest.conf"
cp "$HIVE_SRC/README.md" "$STAGE/README.md"
# Importable flight sheets live at the repo root (hiveos/), not next to the h-*.sh
# integration scripts: they are user-facing examples, not part of the miner package's
# runtime. Ship every one of them.
for f in "$EXAMPLE_SRC"/flight-sheet.*.json; do
    [[ -e "$f" ]] || continue
    cp "$f" "$STAGE/$(basename "$f")"
done
test -e "$STAGE/flight-sheet.example.json"   # the default sheet must exist in the package

# --- bundled glibc runtime -----------------------------------------------------
# The binary needs glibc >= 2.34; HiveOS on an Ubuntu 20.04 base has 2.31 and the
# loader rejects it. h-run.sh launches through this runtime ONLY when the host's
# glibc is older, so rigs on a current image are unaffected.
#
# Extracted from the BUILD IMAGE, not from this packaging host -- it has to be the
# glibc the binary was compiled against, and the packaging host (Arch) is a
# different one entirely. libnss_*/libresolv are included deliberately: glibc
# dlopens them for name resolution, and a mismatch there breaks DNS, which for a
# miner means never reaching the pool.
BUILD_IMG="${BUILD_IMG:-matador-build:pathb-deps-cm4}"
if command -v docker >/dev/null 2>&1 && docker image inspect "$BUILD_IMG" >/dev/null 2>&1; then
    mkdir -p "$STAGE/runtime"
    docker run --rm -v "$STAGE/runtime":/out "$BUILD_IMG" bash -lc '
        L=/lib/x86_64-linux-gnu
        cp -L $L/ld-linux-x86-64.so.2 $L/libc.so.6 $L/libm.so.6 $L/libdl.so.2 \
              $L/libpthread.so.0 $L/librt.so.1 $L/libresolv.so.2 \
              $L/libnss_dns.so.2 $L/libnss_files.so.2 /out/ 2>/dev/null
        chmod 755 /out/*' >/dev/null 2>&1
    if [[ -x "$STAGE/runtime/ld-linux-x86-64.so.2" ]]; then
        echo "[hiveos-pkg] bundled glibc runtime ($(du -sh "$STAGE/runtime" | cut -f1)) for pre-2.34 hosts"
    else
        rm -rf "$STAGE/runtime"
        echo "[hiveos-pkg] WARNING: glibc runtime extraction failed; rigs on glibc < 2.34 will not start" >&2
    fi
else
    echo "[hiveos-pkg] WARNING: no docker/$BUILD_IMG -- shipping WITHOUT the glibc runtime;" >&2
    echo "[hiveos-pkg]          rigs on glibc < 2.34 (Ubuntu 20.04 HiveOS) will not start" >&2
fi

fetch_verify "matador-miner-$VER-linux-x86_64" "matador-miner"
# One binary. Pre-Ampere cards cannot mine ENC_RC at all, so there is no second lane;
# h-run.sh stops a pre-Ampere-only rig with a clear message.

tar -czf "$OUT" -C "$S/build-hiveos" matador-miner
( cd "$DIST" && { command -v sha256sum >/dev/null 2>&1 && sha256sum "$(basename "$OUT")" || shasum -a 256 "$(basename "$OUT")"; } > "$(basename "$OUT").sha256" )

echo
echo "[hiveos-pkg] built: $OUT"
tar -tzf "$OUT"
echo
echo "[hiveos-pkg] upload with:"
echo "  gh release upload $VER --repo $REPO $OUT $OUT.sha256"
