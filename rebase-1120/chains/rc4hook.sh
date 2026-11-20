#!/bin/bash
# (11-20: made from <the local records>/ by <the local records>, the names rc4)
# rx: the 0091 PR's TP=4 receipt (<the local records>; <a local session>, 11-16), made by mk-rx.py from
# 20261110-dscores/ds/rchook.sh with M2's drafted reps (20261116-scores/m2/m2hook.sh).
# On 04, the rx arms' HOOK while a server answers at 127.0.0.1:8095: rc4hook.sh LABEL -- rank 0's look first (a look
# other than the arm's writes rc4/stop-LABEL and runs nothing else), then bench4.sh LABEL (MODE bench) or grid4.sh LABEL
# and, once the grid's log ends "grid LABEL end", m2draft.py's three drafted streams on the kept 1,040,384 prefix into
# rc4/draft-LABEL.json (MODE grid). The arm's want is rc4/want-LABEL, one line "MODE HOW BIN":
#   HOW timed  rank 0's "glm: windows (timed): P; experts W"; "P; experts W" is saved as rc4/own-BIN
#   HOW kept   "glm: windows (kept): " and rc4/own-BIN's words: the start read the table its binary's timed start saved,
#              so every counted start of a binary prices its windows and picks its expert form as that binary's first did
# Every arm: no TF_GLM_DSCORES line of either kind (no arm sets it, and main has none), no "TF_GLM_DSEL=0:" line, and
# rank 0's serving line's drafts word neither empty nor "off". rc4/form.log gets "LABEL serving: drafts: WORDS" and one
# look line "LABEL want MODE HOW BIN: LOOK; dscores N, not loaded N, dsel N -> ok|STOP". rc4/look.log gets "LABEL loaded
# Ns; startup K+K+K+K" (each rank's disabled, fallback or compile failed lines outside NCCL INFO; 05, 07 and 06 over
# sx4.sh), recorded, not judged. Every "glm: " line of rank 0's log goes to rc4/glm-LABEL.txt.
set -u
L=${1:?label}
T=$HOME/LLLLM/glm-work/tp4
F=$T/rc4
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
for p in "host05 1" "host07 2" "host06 3"; do
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
  grid)
    bash "$T/grid4.sh" "$L"
    grep -a -q -x "grid $L end" "$T/bench/$L.log" 2>/dev/null || exit 1
    echo "$(date '+%F %T') == rc4hook $L drafted: m2draft $(sha256sum "$T/m2draft.py" | cut -c1-16), 3 reps on the kept 1040384" >> "$T/bench/$L.log"
    timeout 400 python3 "$T/m2draft.py" http://127.0.0.1:8095 "$T/bench/grid-ids.json" 1040384 3 > "$F/draft-$L.json" 2>&1
    echo "$(date '+%F %T') == rc4hook $L drafted rc=$?" >> "$T/bench/$L.log"
    exit 0 ;;
  bench) exec bash "$T/bench4.sh" "$L" ;;
esac
echo "mode $mode" > "$F/stop-$L"
exit 1
