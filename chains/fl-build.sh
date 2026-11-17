#!/bin/bash
# On 03 (setsid): fl-build.sh SHA NAME -- rx/rx-build.sh (tp4-build.sh's image, zig and `zig build native`, an empty zig
# global cache of its own, tf-work/tp4/zc-NAME, which must not be there; <a local note> §1), then
# `zig build compile-cuda-tests` in the same tree and cache for tf-cuda-test (glm-index-check), as sbench-build.sh
# builds it. For the 0091 PR's head with the takesDecode floor (<the local ledger> 11-17 09:0x). Log tf-work/tp4/build-NAME.log:
# "== tp4-build NAME finished" or "== tp4-build NAME stopped".
set -u
P=$HOME/LLLLM/models/glm53-nvfp4/tf-work/tp4
IMG=tf3-build:20261014 ZIG=$HOME/LLLLM/fp4l-work/zig
WANT=$1
NAME=$2
LOG=$P/build-$NAME.log
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
halt() { log "STOP: $*"; log "== tp4-build $NAME stopped"; exit 1; }
ZC=$P/zc-$NAME
log "== tp4-build $NAME begin ($WANT)"
[ "$(hostname)" = host03 ] || halt "not 03"
[ ! -e "$ZC" ] || halt "$ZC is there: the global cache must start empty"
[ -f "$P/src-$NAME/build.zig" ] || halt "no source in $P/src-$NAME"
[ "$(cat "$P/src-$NAME/.sha" 2>/dev/null)" = "$WANT" ] || halt "src-$NAME is not $WANT"
mkdir -p "$ZC" "$P/home"
run() {  # STEP LOGFILE: zig build STEP in the build image
  docker run --rm --name "tp4-build-$NAME" --user "$(id -u):$(id -g)" -e HOME=/w/home -e ZIG_GLOBAL_CACHE_DIR=/zc \
    -v "$P:/w" -v "$ZC:/zc" -v "$ZIG:/opt/zig:ro" -w "/w/src-$NAME" "$IMG" \
    bash -c "nice -n 5 /opt/zig/zig build $1 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121" > "$2" 2>&1
}
run native "$P/build-native-$NAME.log" || halt "native failed: $(grep -m3 'error' "$P/build-native-$NAME.log" | tr '\n' ' ' | cut -c1-300)"
mkdir -p "$P/bin-$NAME" && cp "$P/src-$NAME/zig-out/native/bin/tensorfold-native" "$P/bin-$NAME/" || halt "no binary"
log "  binary: $(sha256sum "$P/bin-$NAME/tensorfold-native" | cut -c1-16) $(stat -c %s "$P/bin-$NAME/tensorfold-native") bytes"
touch "$P/bin-$NAME/stamp"
run compile-cuda-tests "$P/build-cuda-tests-$NAME.log" || halt "compile-cuda-tests failed: $(grep -m3 'error' "$P/build-cuda-tests-$NAME.log" | tr '\n' ' ' | cut -c1-300)"
exe=$(find "$P/src-$NAME/.zig-cache/o" "$ZC/o" -type f -name tf-cuda-test -newer "$P/bin-$NAME/stamp" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2)
[ -n "$exe" ] || halt "no tf-cuda-test newer than the native build"
cp "$exe" "$P/bin-$NAME/tf-cuda-test" || halt "copy tf-cuda-test"
log "  tf-cuda-test: $(sha256sum "$P/bin-$NAME/tf-cuda-test" | cut -c1-16) $(stat -c %s "$P/bin-$NAME/tf-cuda-test") bytes"
for f in "$P/build-native-$NAME.log" "$P/build-cuda-tests-$NAME.log"; do
  log "  ${f##*/}: disabled $(grep -c -i 'disabled' "$f"), fallback $(grep -c -i 'fallback' "$f"), compile failed $(grep -c -i 'compile failed' "$f")"
done
log "== tp4-build $NAME finished"
