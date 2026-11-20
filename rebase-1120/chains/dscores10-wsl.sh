#!/usr/bin/env bash
# WSL, 11-20 07:3x: #3689 rebased onto #3776 (GLM decode kernels as dependent launches) by this session (<a local session>, <the local ledger> 11-20 07:2x, local choices). R = the rebased head 50c6634539 (the four
# code commits on main b0b17e8962; the launches #3689 adds go through cuda_pdl.launch). N = that main: lean's list.
# On R: lean against N, zig fmt, ast-check, dscores9-wsl.sh's steps, #3776's test-glm-pdl-scan (a CI host step) and
# its SASS checker's fixtures (tools/test_pdl_early_loads.py, a CI step), then CI's pytest from the tree root
# (ops/tf-rules/README.md; skips to be read against main's CI log). check-pdl-loads needs nvcc, which WSL does not
# have: it runs on a GB10 host with the receipts' build. Tars from git -c core.autocrlf=false archive on the PC in $1.
# Every zig build takes the zt-heavy lock itself (the shim): a step that ends 75 (the lock not free in 90 minutes),
# 2 (the lock file not opened) or 124 (timeout) stops the script. Log ~/zt-tp4/dscores10.log; the last line is
# "== dscores10-wsl finished" or "== dscores10-wsl stopped".
set -u
Z=$HOME/.local/share/zig/0.17.0/zig
export ZIG_GLOBAL_CACHE_DIR=$HOME/.cache/zig
export ZT_LABEL=tp4-dscores10
S=$1
R=$HOME/zt-tp4
L=$R/dscores10.log
STEPS="test-environment-host test-glm-index-host test-glm-index-emulate test-glm-all-host test-glm-contract-host test-kernel-contract check-native-cuda-host compile-cuda-tests test-glm-pdl-scan"
stop() { echo "$(date '+%F %T') $*"; echo "== dscores10-wsl stopped"; exit 1; }
held() { case $1 in 75|2|124) stop "$2 rc=$1 (the zig lock or the timeout)";; esac; }
unpack() {  # unpack TAG TAR: a fresh tree in $R/dscores10-TAG
  local h=$R/dscores10-$1
  rm -rf "${h:?}"; mkdir -p "$h"; tar xf "$S/$2" -C "$h" || stop "tar $2 failed"
  echo "tree $1: $(cd "$h" && find . -type f | wc -l) files"
}
throwaway() {  # the GLM module with no filter, as a local step
  python3 - "$1" <<'PY' || stop "throwaway $1"
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "zig/build/glm_family_host.zig"
anchor = "    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);\n"
s = p.read_text()
assert s.count(anchor) == 1, s.count(anchor)
p.write_text(s.replace(anchor, anchor + '    b.step("test-glm-all-host", "throwaway: the GLM module unfiltered").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm })).step);\n'))
print("throwaway step test-glm-all-host")
PY
}
step() {  # step TAG DIR NAME
  echo "== $1 $3 $(date '+%T')"
  (cd "$2" && timeout 3600 "$Z" build "$3" -j4 --summary all > "$R/dscores10-$1.$3.log" 2>&1)
  local rc=$?
  echo "rc=$rc $(grep -E 'Build Summary' "$R/dscores10-$1.$3.log" | tail -n 1)"
  grep -E "^error: '|error: .*[.]zig:[0-9]+:[0-9]+:|FAIL" "$R/dscores10-$1.$3.log" | cut -c1-220 | head -8
  held "$rc" "step $1 $3"
}
lean() {  # lean TAG: lean_check's list of the tree
  (cd "$R/dscores10-$1" && python3 tools/zig/lean_check.py > "$R/dscores10-$1.lean.txt" 2>&1; echo "lean $1 rc=$?: $(tail -n 1 "$R/dscores10-$1.lean.txt" | cut -c1-200)")
}
{
  echo "$(date '+%F %T') begin"
  zt-heavy status 2>&1 | head -n 3
  echo "#### N: main b0b17e8962"
  unpack main dscores10-main.tar
  lean main
  echo "#### R: the rebased head 50c6634539"
  unpack head dscores10-head.tar
  h=$R/dscores10-head
  lean head
  (cd "$h" && python3 tools/zig/lean_new.py "$R/dscores10-main.lean.txt" "$R/dscores10-head.lean.txt"; echo "lean_new head rc=$?")
  # shellcheck disable=SC2046
  (cd "$h" && "$Z" fmt --check $(cat "$S/dscores10-files.txt") && echo "fmt changed files ok" || echo "fmt changed files rc=$?")
  (cd "$h" && "$Z" fmt --check build.zig zig > "$R/dscores10-head.fmt-all.txt" 2>&1; echo "fmt all rc=$? ($(wc -l < "$R/dscores10-head.fmt-all.txt") lines)")
  for f in $(cat "$S/dscores10-files.txt"); do (cd "$h" && "$Z" ast-check "$f" > /dev/null 2>&1; echo "ast-check $f rc=$?"); done
  throwaway "$h"
  for s in $STEPS; do step head "$h" "$s"; done
  echo "== head test_pdl_early_loads $(date '+%T')"
  (cd "$h" && timeout 600 python3 -B tools/test_pdl_early_loads.py -v > "$R/dscores10-head.pdl-fixtures.txt" 2>&1; echo "test_pdl_early_loads rc=$?: $(grep -E '^(Ran|OK|FAILED)' "$R/dscores10-head.pdl-fixtures.txt" | tr '\n' ' ')")
  echo "== head pytest $(date '+%T')"
  (cd "$h" && PATH=$HOME/.local/lib/zt-ci/bin:$HOME/.local/share/zig/0.17.0:$PATH timeout 3600 "$HOME/.local/lib/zt-ci/venv/bin/python" -m pytest -q -rs -p no:cacheprovider > "$R/dscores10-head.pytest.txt" 2>&1)
  rc=$?
  echo "pytest head rc=$rc: $(tail -n 1 "$R/dscores10-head.pytest.txt" | cut -c1-200)"
  grep -E '^FAILED|^ERROR' "$R/dscores10-head.pytest.txt" | cut -c1-220 | head -8
  echo "skip reasons head: $(grep -c '^SKIPPED' "$R/dscores10-head.pytest.txt") lines"
  held "$rc" "pytest head"
  echo "$(date '+%F %T') end"
  echo "== dscores10-wsl finished"
} > "$L" 2>&1
tail -n 5 "$L"
