#!/usr/bin/env bash
#
# Build a coherent, version-matched Intel GPU userspace stack for torch.xpu on
# Arc Battlemage (B580) WITHOUT touching the system driver. Downloads the matched
# Level-Zero loader + NEO + IGC + gmmlib .debs, extracts the .so files, and
# consolidates them into $BOLTZ_XPU_HOME/gpu-stack/lib, then writes xpu-env.sh.
#
# No sudo, no install — a pure userspace LD_LIBRARY_PATH shadow. The stock system
# driver is left in place. See docs/FINDINGS.md for why these exact versions.
#
set -euo pipefail

# --- version-matched component set (docs/FINDINGS.md §3) ---
NEO=26.05.37020.3     # libze-intel-gpu1 (compute-runtime / NEO)
GMM=22.9.0            # libigdgmm12
IGC=v2.28.4           # intel-igc-core-2 / intel-igc-opencl-2
LZ=v1.29.0            # libze1 (Level-Zero loader)
UBU=u24.04            # ubuntu .deb flavor to pull (works on Debian 13 too)

DEST="${BOLTZ_XPU_HOME:-$HOME/boltz-xpu}"
WORK="$DEST/.driver-debs"
LIB="$DEST/gpu-stack/lib"
mkdir -p "$WORK" "$LIB"

# Resolve a release asset's download URL via the GitHub API (build suffixes vary).
asset_url() { # repo  tag  name_regex
  curl -fsSL "https://api.github.com/repos/$1/releases/tags/$2" \
    | grep -oE '"browser_download_url": "[^"]+"' | sed 's/.*: "//; s/"$//' \
    | grep -E "$3" | grep -v dbgsym | head -1
}
fetch() { local u; u="$1"; [ -n "$u" ] || { echo "!! asset not found"; exit 1; }; echo "  $u"; ( cd "$WORK" && curl -fsSL -O "$u" ); }

echo ">> downloading version-matched component .debs into $WORK"
fetch "$(asset_url intel/compute-runtime "$NEO" "libze-intel-gpu1_${NEO}-0_amd64\.deb")"
fetch "$(asset_url intel/compute-runtime "$NEO" "libigdgmm12_${GMM}_amd64\.deb")"
fetch "$(asset_url intel/intel-graphics-compiler "$IGC" "intel-igc-core-2_")"
fetch "$(asset_url intel/intel-graphics-compiler "$IGC" "intel-igc-opencl-2_")"
fetch "$(asset_url oneapi-src/level-zero "$LZ" "libze1_.*${UBU}_amd64\.deb")"

echo ">> extracting .so files"
rm -rf "$WORK/ex"; mkdir -p "$WORK/ex"
for d in "$WORK"/*.deb; do dpkg-deb -x "$d" "$WORK/ex"; done

echo ">> consolidating into $LIB"
rm -f "$LIB"/*.so* 2>/dev/null || true
cp -Pf "$WORK"/ex/usr/lib/x86_64-linux-gnu/*.so* "$LIB"/ 2>/dev/null || true   # loader, NEO, gmm
cp -Pf "$WORK"/ex/usr/local/lib/*.so*            "$LIB"/ 2>/dev/null || true   # IGC

echo ">> writing $DEST/xpu-env.sh"
cat > "$DEST/xpu-env.sh" <<EOF
#!/usr/bin/env bash
# Coherent Intel GPU userspace stack for torch.xpu on Arc Battlemage.
# loader $LZ + NEO $NEO + IGC $IGC + gmmlib $GMM. Shadows the system driver
# per-process only (system is untouched). Source before any torch.xpu run.
export LD_LIBRARY_PATH="$LIB\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
chmod +x "$DEST/xpu-env.sh"

echo
echo "Done. Consolidated GPU stack:"
ls "$LIB" | grep -E 'libze_loader|libze_intel_gpu|libigdgmm|libigc\.|libigdfcl|libiga64' | sed 's/^/  /'
echo
echo "Next: ./scripts/install_boltz_xpu.sh   then   source $DEST/xpu-env.sh"
