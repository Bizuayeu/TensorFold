# Receipts: GLM-5.3-Flash on CUDA, a decode window's DSA select on many CTAs

The raw results and the comparison tools behind the tables of the pull request whose head is `a23c1fe6d` (on main
`1b16a6e8d`). Binaries, by sha256 (first 16): main's native `7a1b270cfa91da1d` and this head's `9b51d5a655f84042`,
each built for these runs with CUDA 13.3; `d8`'s `7a0a87cf53aa3379`. Every run was on GB10s (sm_121), each start fresh.

## Layout

- `tp2/`, `tp3/`: main against this head at TP=2 and TP=3, on the release's settings (drafts on, MTP and copies). The
  arms in the order they ran: `fw2` (this head, not counted), `fm2a`, `fh2a`, `fm2b`, `fh2b`, `fm2c` (`fm` main, `fh`
  this head; `fw3` … `fm3c` at TP=3).
  - `fbench/ARM-*.json`: the reports of upstream's `tools/bench_concurrent.py`, `tools/bench_openai.py` and
    `tools/bench_cache.py`, as `tools/fbench.sh` ran them; `fbench/ARM.log`: its lines.
  - `fbench/f2-fsum.json`, `f2-fsum.txt`: `tools/fsum.py` over the five counted arms (`f3-` at TP=3).
  - `logs/serve-r0-ARM.log`: rank 0's `[tensorfold]` lines, the startup lines `fsum.py` reads among them. The rest of
    each log (NCCL's and the launcher's lines) names the hosts' addresses and is left out.
  - `ftable-tp2.md`, `ftable-tp3.md`: the pull request's tables, from `tools/ftable.py`.
- `tp4l/`: long contexts at TP=4 (four GB10s on a ring, `TF_GLM_PREFILL_REDUCE=gather`, `NCCL_CUMEM_ENABLE=1`,
  `TF_GLM_PROMPT_TAIL=1`), main against this head: `lw4` (this head, not counted), `lm4a` (main), `lh4` (this head),
  `lm4b` (main). `ARM-grid.json` is `tools/grid.py`'s report (the decode round in ms after a shared prompt of 4,096,
  65,536, 262,144 and 1,040,384 tokens, at 1, 4 and 8 streams, `"draft": false`, greedy, every stream's `token_sha`),
  `ARM.log` its lines; `lw4-*` are the uncounted start's `tools/bench4.sh` reports; `logs/` as above; `l-gsum.txt` is
  `tools/gsum.py` over the three.
- `tp4k/`: TP=4's 4K cells taken again, as `tp4l/` ran them but at 4,096 tokens only, 6 reps a cell: `kw4` (this head,
  not counted), `km4a`, `kh4a`, `km4b`, `kh4b`, `km4c` (`km` main, `kh` this head); the files as in `tp4l/`,
  `k-gsum.txt` over the five.
- `d8/`: the switch's runs on `d8` (this change's kernel code and dispatch on an earlier main, `b503008ce`): at TP=4
  `dz4a` (`TF_GLM_DSEL=0`), `dd4` (default), `dz4b` (`=0`); at TP=2 `dz2a`, `dd2`, `dz2b`; TP=2's short cells again,
  `ez2a`, `ed2a`, `ez2b`, `ed2b`, `ez2c` (`ez` `=0`, `ed` default).
- `tools/`: the comparison tools, and the clients the chains ran on the hosts beside the servers.
- `chains/`: the chains and arm scripts as they ran, to read. They reach the hosts through `sx4.sh` and start the ranks
  with `serve4.sh` and a `rank.env`, which hold the hosts' addresses and are left out. In the three arm scripts the
  boxes' names read `host01` to `host07`, and one comment in `chains/tp2-arm.sh` that named the pair's link address
  reads `<the pair's link address>`; nothing else differs. The files as they ran, by sha256: `tp2-arm.sh`
  `14246af9a461e0d0db057112375a69d32e0dcdd692eefb17955c2987e3a802af`, `tp3-arm.sh`
  `6f87303cd17282055db10f086f9b89876725a621314f747d10995e50529e0c9c`, `tp4-arm.sh`
  `11ced8842eae0e62ca65c2c9f5a55c62a920a8afecf09af8ccfa5d1cb4400b21`.
- `SHA256SUMS`: every file's sha256.

## Rerun the tables

Python 3, standard library only, from this directory:

```sh
python3 tools/fsum.py tp2/fbench tp2/logs fm2a,fm2b,fm2c fh2a,fh2b --json /tmp/f2-fsum.json  # tp2/fbench/f2-fsum.json
python3 tools/ftable.py 2 tp2/fbench/f2-fsum.json                                         # tp2/ftable-tp2.md
python3 tools/ccold.py tp2/fbench fw2,fm2a,fh2a,fm2b,fh2b,fm2c   # each cold prompt's greedy reply, one hash over the arms
python3 tools/cwarm.py tp2/fbench fw2,fm2a,fh2a,fm2b,fh2b,fm2c   # each resumed request against its cold one
python3 tools/gsum.py tp4l lm4a lh4 lm4b                         # long contexts: rounds, spread, replies against lm4a
python3 tools/gverdict.py tp4l lm4a,lm4b lh4                     # the TP=4 table's last column, under fsum.py's rule
python3 tools/gsum.py tp4k km4a kh4a km4b kh4b km4c              # the 4K cells taken again
python3 tools/gverdict.py tp4k km4a,km4b,km4c kh4a,kh4b          # the same, main's three against this head's two
python3 tools/gsum.py d8 dz4a dd4 dz4b
```

At TP=3, the same with `tp3`, `fm3a,fm3b,fm3c fh3a,fh3b` and `fw3,…`. `fsum.py`'s rule: a row is slower when this
head's mean loses more than main's range or 0.25%, whichever is larger, decided before the runs.
