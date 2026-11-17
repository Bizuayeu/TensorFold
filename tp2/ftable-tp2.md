| TP=2, mean of 3 runs of main and 2 of this head | main | this head | this head against main (+ slower, − faster) |
|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 streams | 48.4 / 17.6 | 48.4 / 17.6 | +0.00% / +0.00% |
| decode, all streams' tok/s, 4 streams | 68.0 | 68.3 | −0.56% |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 6.977 / 26.226 | 6.953 / 26.301 | −0.34% / +0.29% |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.354 / 5.745 / 22.594 / 44.832 | 1.346 / 5.781 / 22.519 / 44.618 | −0.58% / +0.62% / −0.33% / −0.48% |
| `bench_openai`, t=0, code / chat (tok/s) | 43.4 / 37.2 | 43.0 / 36.8 | +0.91% / +1.14% |
| `bench_openai`, t=1, code / chat (tok/s) | 41.4 / 36.0 | 41.7 / 36.3 | −0.76% / −0.72% |

| TP=2, in run order | main | this head | main | this head | main |
|---|---|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 streams | 48.4 / 17.6 | 49.0 / 17.6 | 48.1 / 17.6 | 47.8 / 17.6 | 48.7 / 17.6 |
| decode, all streams' tok/s, 4 streams | 68.3 | 68.2 | 67.9 | 68.5 | 67.7 |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 6.976 / 26.070 | 7.040 / 26.324 | 7.053 / 26.273 | 6.866 / 26.278 | 6.902 / 26.336 |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.370 / 5.724 / 22.739 / 44.711 | 1.353 / 5.827 / 22.626 / 44.451 | 1.348 / 5.767 / 22.663 / 44.944 | 1.339 / 5.736 / 22.413 / 44.786 | 1.344 / 5.745 / 22.381 / 44.842 |
| `bench_openai`, t=0, code / chat (tok/s) | 43.8 / 37.0 | 44.2 / 35.9 | 43.0 / 37.3 | 41.9 / 37.7 | 43.6 / 37.3 |
| `bench_openai`, t=1, code / chat (tok/s) | 41.6 / 35.8 | 41.6 / 36.2 | 42.5 / 35.9 | 41.8 / 36.3 | 40.0 / 36.4 |

  cold 2048 s                              loss  -0.58% main's range  1.93% within
  cold 8192 s                              loss  +0.62% main's range  0.75% open
* cold 9600 s (mean of two)                loss  -0.34% main's range  2.17% within
  cold 32768 s                             loss  -0.33% main's range  1.58% within
* cold 38400 s (mean of two)               loss  +0.29% main's range  1.01% within
  cold 65536 s                             loss  -0.48% main's range  0.52% faster
  openai fibonacci-raw t=1.0 tok/s         loss  -0.76% main's range  5.93% within
  openai gpu-chat-no-think t=1.0 tok/s     loss  -0.72% main's range  1.72% within
  openai fibonacci-raw t=0.0 tok/s         loss  +0.91% main's range  1.84% within
  openai gpu-chat-no-think t=0.0 tok/s     loss  +1.14% main's range  0.93% slower
* decode x1 per stream tok/s               loss  +0.00% main's range  1.24% within
* decode x4 per stream tok/s               loss  +0.00% main's range  0.00% within
* decode x4 aggregate tok/s                loss  -0.56% main's range  0.88% faster
  edit x1 per stream tok/s                 loss  +1.04% main's range  2.51% open
  edit x4 per stream tok/s                 loss  -1.03% main's range  0.44% faster
  edit x4 aggregate tok/s                  loss  -1.02% main's range  1.56% faster
gate figures: 5, main's widest range 2.17% (cold 9600 s (mean of two)); verdicts {'faster': 4, 'within': 9, 'open': 2, 'slower': 1}; exactness failures 0
