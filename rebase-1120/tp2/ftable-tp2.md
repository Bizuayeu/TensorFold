| TP=2, mean of 4 runs of main and 3 of this head | main | this head | this head against main (+ slower, − faster) |
|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 streams | 48.9 / 17.9 | 48.6 / 17.8 | +0.60% / +0.74% |
| decode, all streams' tok/s, 4 streams | 68.9 | 68.9 | +0.04% |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 6.961 / 26.174 | 6.859 / 26.131 | −1.46% / −0.16% |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.342 / 5.721 / 22.564 / 44.754 | 1.339 / 5.706 / 22.455 / 44.662 | −0.23% / −0.26% / −0.48% / −0.21% |
| `bench_openai`, t=0, code / chat (tok/s) | 43.4 / 37.7 | 45.2 / 37.8 | −4.17% / −0.38% |
| `bench_openai`, t=1, code / chat (tok/s) | 42.7 / 36.5 | 42.6 / 36.2 | +0.18% / +0.83% |

| TP=2, in run order | main | this head | main | this head | main | this head | main |
|---|---|---|---|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 streams | 49.3 / 18.0 | 49.4 / 17.8 | 48.6 / 17.8 | 48.3 / 17.9 | 48.6 / 17.9 | 48.2 / 17.6 | 49.2 / 17.9 |
| decode, all streams' tok/s, 4 streams | 69.6 | 69.0 | 69.0 | 69.7 | 68.3 | 68.0 | 68.8 |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 6.947 / 26.157 | 6.718 / 26.065 | 7.072 / 26.132 | 6.897 / 26.236 | 6.863 / 26.268 | 6.962 / 26.092 | 6.962 / 26.140 |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.351 / 5.725 / 22.407 / 44.691 | 1.343 / 5.711 / 22.209 / 44.440 | 1.336 / 5.723 / 22.600 / 44.915 | 1.340 / 5.699 / 22.575 / 44.760 | 1.341 / 5.708 / 22.694 / 44.597 | 1.335 / 5.709 / 22.582 / 44.786 | 1.340 / 5.728 / 22.554 / 44.812 |
| `bench_openai`, t=0, code / chat (tok/s) | 43.4 / 37.8 | 45.9 / 38.0 | 41.7 / 37.2 | 43.7 / 37.6 | 44.2 / 37.4 | 46.0 / 37.9 | 44.1 / 38.2 |
| `bench_openai`, t=1, code / chat (tok/s) | 43.1 / 36.2 | 43.0 / 36.6 | 42.0 / 35.9 | 42.3 / 35.2 | 42.4 / 37.2 | 42.5 / 36.7 | 43.2 / 36.6 |

  cold 2048 s                              loss  -0.23% main's range  1.11% within
  cold 8192 s                              loss  -0.26% main's range  0.34% faster
* cold 9600 s (mean of two)                loss  -1.46% main's range  3.00% faster
  cold 32768 s                             loss  -0.48% main's range  1.27% within
* cold 38400 s (mean of two)               loss  -0.16% main's range  0.52% faster
  cold 65536 s                             loss  -0.21% main's range  0.71% within
  openai fibonacci-raw t=1.0 tok/s         loss  +0.18% main's range  2.70% within
  openai gpu-chat-no-think t=1.0 tok/s     loss  +0.83% main's range  3.53% within
  openai fibonacci-raw t=0.0 tok/s         loss  -4.17% main's range  5.65% faster
  openai gpu-chat-no-think t=0.0 tok/s     loss  -0.38% main's range  2.59% within
* decode x1 per stream tok/s               loss  +0.60% main's range  1.43% within
* decode x4 per stream tok/s               loss  +0.74% main's range  1.12% open
* decode x4 aggregate tok/s                loss  +0.04% main's range  1.89% within
  edit x1 per stream tok/s                 loss  +1.19% main's range  7.68% within
  edit x4 per stream tok/s                 loss  +0.14% main's range  1.74% within
  edit x4 aggregate tok/s                  loss  +0.37% main's range  3.46% within
gate figures: 5, main's widest range 3.00% (cold 9600 s (mean of two)); verdicts {'faster': 4, 'within': 11, 'open': 1, 'slower': 0}; exactness failures 0
