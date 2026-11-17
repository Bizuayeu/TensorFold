# Receipts: GLM-5.3-Flash on CUDA, a decode window's DSA index scores on glm_index_scores

The raw results and the comparison tools behind the tables of the pull request whose last commit with code is
`a4da5a4eb2` (on main `e86a0bed1d`; the commits after it change `CHANGELOG.md` only). Binaries, by sha256 (first 16), each built for these runs with CUDA 13.3 (nvcc V13.3.73, sm_121) in an
empty Zig global cache: main's native `1cfdb8a0c81eb7c0`, this head's `3bfb718d39a304e4` (TP=2, TP=3), and the third
commit `cc457382c9`'s `529d728d99655ee9` (TP=4). Every run was on GB10s (sm_121), each start fresh.

## Layout

- `tp4/`: main against the branch at TP=4 (four GB10s on a ring, `TF_GLM_PREFILL_REDUCE=gather`, `NCCL_CUMEM_ENABLE=1`,
  `TF_GLM_PROMPT_TAIL=1`), every counted start kept on main's window and draft prices.
  - `rx2/`: `rxma` (main), `rxh` (the branch), `rxmb` (main). `ARM-grid.json` is `tools/grid.py`'s report (the decode
    round in ms after a shared prompt of 4,096, 65,536, 262,144 and 1,040,384 tokens, at 1, 4 and 8 streams,
    `"draft": false`, greedy, every stream's `token_sha`), `ARM.log` its lines; `draft-ARM.json` the three drafted reps
    after the kept 1,040,384-token prompt (512 tokens, greedy, drafts on), from `tools/m2draft.py`; `gsum.txt` is
    `tools/gsum.py` over the three.
  - `rx3/`: 256K and 1016K again, five runs: `rx3ma`, `rx3ha`, `rx3mb`, `rx3hb`, `rx3mc` (`m` main, `h` the branch).
  - `logs/serve-r0-ARM.log`: rank 0's `[tensorfold]` lines, the start-up lines among them. The rest of each log (NCCL's
    and the launcher's lines) names the hosts' addresses and is left out.
- `tp2/`, `tp3/`: main against this head at TP=2 and TP=3, on the release's settings (drafts on, MTP and copies), every
  counted start kept on main's window and draft prices.
  - TP=2's arms in the order they ran: `rf2cwh` (this head, not counted: it times its prices once), then `rf2dma`,
    `rf2dha`, `rf2dmb`, `rf2dhb`, `rf2dmc` (`m` main, `h` this head). TP=3's: `rf3dwh` (not counted), `rf3dma`, then
    `rf3eha`, `rf3emb`, `rf3ehb`, `rf3emc`. Each set's chain stopped once on its own check of the price records and
    went on from there under a new name (`rf2d`, `rf3e`): at TP=2 after the uncounted start, before any counted run
    (16 minutes before `rf2dma`); at TP=3 after main's first counted run, when this head's next start (`rf3dha`) kept
    its own prices, so it served for about a minute and stopped before its benchmarks; it is left out, and `rf3eha`
    began 14 minutes after `rf3dma` ended.
  - `fbench/ARM-*.json`: the reports of upstream's `tools/bench_concurrent.py`, `tools/bench_openai.py` and
    `tools/bench_cache.py`, as `tools/fbench.sh` ran them; `fbench/ARM.log`: its lines.
  - `fbench/rf2d-fsum.json`, `rf2d-fsum.txt`: `tools/fsum.py` over the five counted arms (`rf3e-` at TP=3).
  - `logs/serve-r0-ARM.log`: as in `tp4/`.
  - `ftable-tp2.md`, `ftable-tp3.md`: the pull request's tables, from `tools/ftable.py`.
- `tp3-retake/`: TP=3's `bench_openai` code row at t=0 (`fibonacci-raw`) taken again, once, with the arms, their order
  and the rule fixed before it ran. Nine fresh starts in this order: `rf3gma` (main), `rf3gha` (this head), `rf3gza`
  (this head with `TF_GLM_DSCORES=0` in every rank's environment, not counted), `rf3gmb`, `rf3ghb`, `rf3gmc`, `rf3gzb`,
  `rf3ghc`, `rf3gmd`. Each ran fbench's cold prompts and then `bench_openai` (`chains/obench.sh`: `tools/fbench.sh` up to
  that step, unchanged), on the price table that `tp3/`'s set ended with, put back before the first start and the same
  after every start (`tables/`: the table's files by sha256, as put back and after each start); every start kept
  main's window and draft prices (`form.log`, each start's words).
  - `fbench/ARM-*.json`, `fbench/ARM.log`: as in `tp3/`. `fbench/rf3g-fsum.*`: `tools/fsum.py` over main's four and this
    head's three; `fbench/rf3g-zsum.*`: over main's four and the two `TF_GLM_DSCORES=0` starts. The set stops after
    `bench_openai`, so it has no `bench_concurrent` cells: fsum lists each start's decode and edit cells as exactness
    failures, "no report" (14 lines in `rf3g-fsum.txt`, 12 in `rf3g-zsum.txt`), and those are its only ones.
  - `logs/serve-r0-ARM.log`: rank 0's `[tensorfold]` lines, its price words (`info: glm: windows`) and its
    `TF_GLM_DSCORES=0` warning. `dscores-lines.txt`: that warning's count in each rank's log, by start.
- `tp3-first/`: the first three commits (`cc457382c9`) at TP=3, three sets of five runs (`rf3`, `rf3r` on each
  binary's own prices, `rf3c` on main's): fbench's reports and fsum's text.
- `sass/`: main's and this head's builds compared: the source trees and Zig caches that built the TP=2 and TP=3
  binaries (this head's tree also built `tf-cuda-test` after its binary). For each fatbin name found in both trees'
  caches (115, `both.list`, each cache's newest file of the name), `cuobjdump -sass` of both, compared function by
  function (`sass-0091.sh`, `sass-0091.py`); `summary.txt`: 1,176 functions, every one the same in both.
- `bench/`: the one-GPU bench outputs, `tf-cuda-test glm-score-bench` from a local branch (11 key caches of the
  context's length read in turn as one pass, the median of 15 reps by GPU events): `s1-bench.txt`, the short contexts
  behind the fourth commit; `c4-bench.txt` to `c6-bench.txt`, the sweep behind `decodeLoop`, compared in `c456-cmp.txt`
  by `tools/bcmp2.py`.
- `tools/`: the comparison tools, and the clients the chains ran on the hosts beside the servers. `t0rounds.py` and
  `t0reps.py` read the `bench_openai` rows at t=0 from rank 0's done lines: each run's three reps after the warm-up
  (tok/s, rounds, ms a round), and the reps' mean by binary.
- `chains/`: the chains, hooks and arm scripts as they ran, to read. They reach the hosts through `sx4.sh` and `sx.sh`
  and start the ranks with `serve4.sh`, the release image's `serve.sh` and a rank file, which hold the hosts' addresses
  and are left out. In `chains/` and `tools/` the boxes' names read `host01` to `host07`, one comment's address of the
  TP=2 pair's link reads `<the pair's link address>`, and comments naming local notes, local paths, local choices
  and the working sessions read `<a local plan>`, `<the local ledger>`, `<a local note>`, `<the local records>`,
  `a local decision`, `<a local session>` or `<another local session>`; nothing else differs. The files so changed, as they ran, by sha256:
  `fl-build.sh` `df8556675b896c906b99f52bea9d1e0ada5963e1dc29e8b35a849838e74a158c`, `fl-gpu01.sh` `170f3919c5b37e38b9f16a4832a4c35898696d2da52c3ba7b97b196a2deb0a26`, `grid4l.sh` `e8e87cd45cedb0e8fa89702ee531b00d881a957ea7d55b62f32ce5f37eca3f1c`, `obench.sh` `1eee00a03cf56790a33ad8797febec54d8f633d8b2158423234cd26509799b5a`, `rf2c-chain.sh` `b711bb068790475a9af589459e0c57cdfbcfd7c4da0e0b8e9c19af1f95814f75`, `rf2chook.sh` `90e91fc156d0af7e5373810fc07ec3a99bb910d061d1b0bbdc49ba6b7bacb627`, `rf2d-chain.sh` `c28444018d5d8ec18f3ab85659dad3da5f9ad0dc565502d0e24946b139806e8d`, `rf2dhook.sh` `ab4bbe4fd15ffdd05270e922d1505fbff2ffc57f2b0fbbea22ec6c7da763c47d`, `rf3d-chain.sh` `9bd8bf63c16d1ab28391f3ec3861da9694b7d1fd0c92220921a6c481506742d0`, `rf3dhook.sh` `faf9112826d61603cb3fc04e16f8e411ebea935ce9c686ca6ef613151a0efe8a`, `rf3e-chain.sh` `81bfc954d97548fe591aaf27cf984113c5df37b2c467618b0b515bcf76dc4028`, `rf3ehook.sh` `a9bda93e33648e32eb2f4463d8f38fac0f37bb6953243973153a44f5b6ad7dc8`, `rf3g-chain.sh` `bbc2ed1764226972f4bb9d3609d390046e1d77216c979f20144403042034c94b`, `rf3ghook.sh` `6a747b41338f40392ab256bd3cef68f7e8c4664cd3e933f0832ad550ee8235ca`, `rx2-chain.sh` `77e591a9c5a5173dd2e0b5d000346cebae0b0f2d5c979beb81730eb122ad3912`, `rx3-chain.sh` `262871f6c05315db374b6e377be7cde072d020ef824fa6e090980af9294dff7e`, `rx3hook.sh` `183ec6f13760e2a582dbe2c88ab95f410943ac7aa909bbaa4e17c2422beb08ea`, `rxhook.sh` `ad352f508b0dfe967b0d41c0555efb11bf413a50ecc85bb55307de823baf4e42`, `tp2-arm.sh` `14246af9a461e0d0db057112375a69d32e0dcdd692eefb17955c2987e3a802af`, `tp3-arm-g.sh` `92038dccb4213ca8ea39d2c29e9c395f0e1c7094b4aea4315d0958a007adb6db`, `tp3-arm.sh` `0a2181c597a3f8bb5acfcfca053f191f098470135d7421de175905e0a6e13355`, `tp4-arm.sh` `b8ed84656bc56c046818ac0829f9134886d8f3fbb55cc96f3894ddf5db55f9a8`, `sass-0091.sh` `8b19c7a073e047e1c3d1c4a1990ce5f843bf87241f5007e79cd3184cfc6e0373`, `fbench.sh` `3497f5c77c54a8d187e60a3648842227294c613a7652373fc49a9447a6b154a9`, `flwords.py` `80810024518d9b90ecb6798cc27fceea970cd23f7061989271427cfa6a743d2a`, `fsum.py` `db74650034aa894d2116fe09ad0d584d93002cf61874e5ab008f094ebe33f7e7`, `grid.py` `ac167c3342d4f43f13cead4c650579a49ba5ae3819ac897e4476ccea2cee932a`, `kzt.py` `413eb166da8f8e7292eefcbdb8da8a9080c7de06c67659d1d8f70db423a15a09`, `m2draft.py` `6be92c482124f5ccffb09621c0aa273151f711c135256be2f5b4cbb9eb9ff304`, `mkprompts_f.py` `3c5951437fb81854f16924f1833ba470baabde70f07fc16fa959fe1bd9018bb1`, `x4pool.py` `c631df20cbc65c5e562b7bfb732428958900bd12c6e7ff298d3219d619347d31`.
- `SHA256SUMS`: every file's sha256.

## Rerun the tables

Python 3, standard library only, from this directory:

```sh
python3 tools/fsum.py tp2/fbench tp2/logs rf2dma,rf2dmb,rf2dmc rf2dha,rf2dhb --json /tmp/rf2d-fsum.json  # tp2/fbench/rf2d-fsum.json
python3 tools/ftable.py 2 tp2/fbench/rf2d-fsum.json rf2dma,rf2dha,rf2dmb,rf2dhb,rf2dmc         # tp2/ftable-tp2.md
python3 tools/ccold.py tp2/fbench rf2cwh,rf2dma,rf2dha,rf2dmb,rf2dhb,rf2dmc   # each cold prompt's greedy reply, one hash over the arms
python3 tools/cwarm.py tp2/fbench rf2cwh,rf2dma,rf2dha,rf2dmb,rf2dhb,rf2dmc   # each resumed request against its cold one
python3 tools/x4pool.py tp2/fbench rf2dma,rf2dmb,rf2dmc rf2dha,rf2dhb         # the decode cells' tokens a round and round
python3 tools/gsum.py tp4/rx2 rxma rxh rxmb                                    # long contexts: rounds, spread, replies against rxma
python3 tools/gverdict.py tp4/rx2 rxma,rxmb rxh                                # under fsum.py's rule
python3 tools/gverdict.py tp4/rx3 rx3ma,rx3mb,rx3mc rx3ha,rx3hb
```

At TP=3, the same with `tp3`, `rf3dma,rf3emb,rf3emc rf3eha,rf3ehb` and `rf3dwh,rf3dma,rf3eha,rf3emb,rf3ehb,rf3emc`.
The retake:

```sh
python3 tools/fsum.py tp3-retake/fbench tp3-retake/logs rf3gma,rf3gmb,rf3gmc,rf3gmd rf3gha,rf3ghb,rf3ghc --json /tmp/rf3g-fsum.json
python3 tools/fsum.py tp3-retake/fbench tp3-retake/logs rf3gma,rf3gmb,rf3gmc,rf3gmd rf3gza,rf3gzb --json /tmp/rf3g-zsum.json
python3 tools/t0rounds.py tp3-retake/logs rf3gma rf3gha rf3gza rf3gmb rf3ghb rf3gmc rf3gzb rf3ghc rf3gmd
python3 tools/t0reps.py tp3-retake/logs code rf3gma,rf3gmb,rf3gmc,rf3gmd rf3gha,rf3ghb,rf3ghc
python3 tools/t0reps.py tp3-retake/logs code rf3gma,rf3gmb,rf3gmc,rf3gmd rf3gza,rf3gzb
python3 tools/ccold.py tp3-retake/fbench rf3gma,rf3gha,rf3gza,rf3gmb,rf3ghb,rf3gmc,rf3gzb,rf3ghc,rf3gmd
python3 tools/cwarm.py tp3-retake/fbench rf3gma,rf3gha,rf3gza,rf3gmb,rf3ghb,rf3gmc,rf3gzb,rf3ghc,rf3gmd
python3 tools/t0rounds.py tp3/logs rf3dma rf3eha rf3emb rf3ehb rf3emc                  # the first TP=3 set's rounds
```
`fsum.py`'s rule: a row is slower when this head's mean loses more than main's range or 0.25%, whichever is larger,
decided before the runs.
