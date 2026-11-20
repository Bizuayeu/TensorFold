#!/bin/bash
# (11-20: made from <the local records>/ by <the local records>, the names rc2)
# (11-16 23:5x: copied by <the local records> for tonight's binaries, me86a and hcc45)
# rc2: MIA0091 stage 4's F gate at TP=2 (<a local plan> §3; <a local session>, 11-16), made from
# ds/rchook.sh for tp2-arm.sh (rank 0 on 04, rank 1 on 05) and fbench.sh.
# On 04, the rc2 arms' HOOK while a server answers at 127.0.0.1:8095: rc2hook.sh LABEL -- rank 0's look first (a look
# other than the arm's writes rc2/stop-LABEL and runs nothing else), then fbench.sh LABEL (MODE fbench). The arm's want
# is rc2/want-LABEL, one line "MODE HOW BIN":
#   HOW timed  rank 0's "glm: windows (timed): P; experts W"; "P; experts W" is saved as rc2/own-BIN
#   HOW kept   "glm: windows (kept): " and rc2/own-BIN's words: the start read the table its binary's timed start saved
# Every arm: no TF_GLM_DSCORES line of either kind (no arm sets it, and main has none), no "TF_GLM_DSEL=0:" line, and
# rank 0's serving line's drafts word neither empty nor "off". rc2/form.log gets "LABEL serving: drafts: WORDS" and one
# look line "LABEL want MODE HOW BIN: LOOK; dscores N, not loaded N, dsel N -> ok|STOP". rc2/look.log gets "LABEL loaded
# Ns; startup K+K" (each rank's disabled, fallback or compile failed lines outside NCCL INFO; 05 over sx4.sh), recorded,
# not judged. Every "glm: " line of rank 0's log goes to rc2/glm-LABEL.txt.
set -u
L=${1:?label}
T=$HOME/LLLLM/glm-work/tp4
F=$T/rc2
R0=$T/logs/serve-r0-$L.log
ZLINE="TF_GLM_DSCORES=0: a decode window's index scores keep tf_glm_scores"
NLINE="TF_GLM_DSCORES: glm_index is not loaded"
DLINE="TF_GLM_DSEL=0:"
read -r mode how bin < "$F/want-$L" || { echo "no want for $L" > "$F/stop-$L"; exit 1; }
[ -n "${bin:-}" ] || { echo "no binary in the want for $L" > "$F/stop-$L"; exit 1; }
[ -f "$R0" ] || { echo "no rank 0 log $R0" > "$F/stop-$L"; exit 1; }
grep -a -o 'glm: .*$' "$R0" > "$F/glm-$L.txt"
sl=$(grep -a -m1 '\] serving ' "$R0")
dw=$(echo "$sl" | sed -n 's/^.*; drafts: \([^;]*\);.*$/\1/p')
ld=$(echo "$sl" | sed -n 's/^.*loaded in \([0-9.]*\)s.*$/\1/p')
st=$(grep -a -E 'disabled|fallback|compile failed' "$R0" | grep -a -v 'NCCL INFO' | grep -c .)
for p in "host05 1"; do
  h=${p% *} r=${p#* }
  k=$(bash "$T/sx4.sh" "$h" "f=\$HOME/LLLLM/glm-work/tp4/logs/serve-r$r-$L.log; if [ -f \$f ]; then grep -a -E 'disabled|fallback|compile failed' \$f | grep -a -v 'NCCL INFO' | grep -c .; else echo missing; fi" < /dev/null 2>/dev/null | tail -n 1)
  st="$st+${k:-?}"
done
echo "$(date '+%F %T') $L loaded ${ld:-?}s; startup $st" >> "$F/look.log"
echo "$(date '+%F %T') $L serving: drafts: ${dw:-none found}" >> "$F/form.log"
line=$(grep -a -m1 'glm: windows (' "$R0" | sed 's/^.*glm: windows (/glm: windows (/')
nz=$(grep -a -c -F "$ZLINE" "$R0")
nn=$(grep -a -c -F "$NLINE" "$R0")
nd=$(grep -a -c -F "$DLINE" "$R0")
ok=0
case $how in
  timed)
    case $line in
      "glm: windows (timed): "*"; experts "*) echo "${line#glm: windows (timed): }" > "$F/own-$bin" && ok=1 ;;
    esac ;;
  kept)
    own=$(cat "$F/own-$bin" 2>/dev/null)
    [ -n "$own" ] && [ "$line" = "glm: windows (kept): $own" ] && ok=1 ;;
esac
[ "$nz" = 0 ] && [ "$nn" = 0 ] && [ "$nd" = 0 ] || ok=0
case "$dw" in "" | off) ok=0 ;; esac
look="${line:-no windows line}; dscores $nz, not loaded $nn, dsel $nd"
echo "$(date '+%F %T') $L want $mode $how $bin: $look -> $([ $ok = 1 ] && echo ok || echo STOP)" >> "$F/form.log"
[ $ok = 1 ] || { echo "$look; drafts: ${dw:-none found}" > "$F/stop-$L"; exit 1; }
case $mode in
  fbench) exec bash "$T/fbench.sh" "$L" ;;
esac
echo "mode $mode" > "$F/stop-$L"
exit 1
