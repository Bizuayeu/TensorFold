| TP=2, mean of 3 runs of main and 2 of this head | main | this head | this head against main (+ slower, − faster) |
|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 streams | 48.5 / 17.6 | 48.5 / 17.7 | +0.10% / −0.76% |
| decode, all streams' tok/s, 4 streams | 68.2 | 68.2 | +0.02% |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 6.851 / 26.344 | 6.898 / 26.297 | +0.69% / −0.18% |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.345 / 5.759 / 22.645 / 45.091 | 1.343 / 5.725 / 22.531 / 44.934 | −0.10% / −0.59% / −0.50% / −0.35% |
| `bench_openai`, t=0, code / chat (tok/s) | 45.0 / 36.7 | 45.0 / 36.8 | −0.11% / −0.36% |
| `bench_openai`, t=1, code / chat (tok/s) | 42.7 / 35.8 | 42.4 / 35.7 | +0.56% / +0.39% |

| TP=2, in run order | main | this head | main | this head | main |
|---|---|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 streams | 48.5 / 17.4 | 48.4 / 17.6 | 48.6 / 17.7 | 48.5 / 17.8 | 48.4 / 17.6 |
| decode, all streams' tok/s, 4 streams | 67.9 | 67.9 | 68.4 | 68.4 | 68.2 |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 7.031 / 26.386 | 6.994 / 26.406 | 6.791 / 26.460 | 6.801 / 26.187 | 6.730 / 26.186 |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.334 / 5.725 / 22.898 / 45.548 | 1.335 / 5.728 / 22.675 / 45.062 | 1.345 / 5.739 / 22.466 / 44.743 | 1.351 / 5.723 / 22.387 / 44.807 | 1.354 / 5.813 / 22.571 / 44.982 |
| `bench_openai`, t=0, code / chat (tok/s) | 44.9 / 36.6 | 45.0 / 36.8 | 45.0 / 36.7 | 45.1 / 36.8 | 45.0 / 36.7 |
| `bench_openai`, t=1, code / chat (tok/s) | 42.4 / 35.7 | 42.4 / 35.8 | 43.2 / 36.0 | 42.4 / 35.5 | 42.3 / 35.7 |

  cold 2048 s                              loss  -0.10% main's range  1.49% within
  cold 8192 s                              loss  -0.59% main's range  1.53% faster
* cold 9600 s (mean of two)                loss  +0.69% main's range  4.39% within
  cold 32768 s                             loss  -0.50% main's range  1.91% within
* cold 38400 s (mean of two)               loss  -0.18% main's range  1.04% within
  cold 65536 s                             loss  -0.35% main's range  1.79% within
  openai fibonacci-raw t=1.0 tok/s         loss  +0.56% main's range  2.01% within
  openai gpu-chat-no-think t=1.0 tok/s     loss  +0.39% main's range  0.73% open
  openai fibonacci-raw t=0.0 tok/s         loss  -0.11% main's range  0.18% faster
  openai gpu-chat-no-think t=0.0 tok/s     loss  -0.36% main's range  0.15% faster
* decode x1 per stream tok/s               loss  +0.10% main's range  0.41% within
* decode x4 per stream tok/s               loss  -0.76% main's range  1.71% faster
* decode x4 aggregate tok/s                loss  +0.02% main's range  0.73% within
  edit x1 per stream tok/s                 loss  +2.15% main's range  0.13% slower
  edit x4 per stream tok/s                 loss  +0.29% main's range  0.88% within
  edit x4 aggregate tok/s                  loss  -0.09% main's range  3.77% within
gate figures: 5, main's widest range 4.39% (cold 9600 s (mean of two)); verdicts {'faster': 4, 'within': 10, 'open': 1, 'slower': 1}; exactness failures 0
