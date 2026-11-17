| TP=3, mean of 3 runs of main and 2 of this head | main | this head | this head against main (+ slower, − faster) |
|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 / 8 streams | 69.5 / 24.4 / 14.8 | 68.6 / 24.2 / 14.9 | +1.29% / +0.61% / −0.68% |
| decode, all streams' tok/s, 4 / 8 streams | 90.6 / 99.1 | 90.7 / 99.5 | −0.09% / −0.49% |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 5.341 / 20.664 | 5.343 / 20.645 | +0.04% / −0.09% |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.090 / 4.599 / 17.663 / 35.227 | 1.096 / 4.598 / 17.630 / 35.190 | +0.59% / −0.04% / −0.18% / −0.11% |
| `bench_openai`, t=0, code / chat (tok/s) | 63.4 / 51.6 | 61.4 / 51.7 | +3.14% / −0.28% |
| `bench_openai`, t=1, code / chat (tok/s) | 51.6 / 46.9 | 51.8 / 46.5 | −0.48% / +0.81% |

| TP=3, in run order | main | this head | main | this head | main |
|---|---|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 / 8 streams | 70.0 / 24.4 / 14.6 | 68.2 / 24.1 / 14.9 | 69.5 / 24.4 / 14.9 | 69.0 / 24.4 / 14.9 | 69.0 / 24.4 / 14.9 |
| decode, all streams' tok/s, 4 / 8 streams | 90.5 / 97.8 | 90.3 / 99.6 | 90.3 / 99.5 | 91.0 / 99.5 | 90.9 / 99.9 |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 5.347 / 20.708 | 5.339 / 20.632 | 5.353 / 20.668 | 5.348 / 20.658 | 5.325 / 20.616 |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.095 / 4.597 / 17.706 / 35.344 | 1.099 / 4.596 / 17.605 / 35.161 | 1.099 / 4.602 / 17.641 / 35.231 | 1.094 / 4.599 / 17.655 / 35.219 | 1.076 / 4.599 / 17.641 / 35.107 |
| `bench_openai`, t=0, code / chat (tok/s) | 63.9 / 51.3 | 61.1 / 52.1 | 62.9 / 51.5 | 61.7 / 51.3 | 63.4 / 51.8 |
| `bench_openai`, t=1, code / chat (tok/s) | 51.9 / 48.0 | 51.6 / 46.5 | 50.9 / 46.3 | 52.1 / 46.5 | 51.9 / 46.4 |

  cold 2048 s                              loss  +0.59% main's range  2.07% within
  cold 8192 s                              loss  -0.04% main's range  0.11% within
* cold 9600 s (mean of two)                loss  +0.04% main's range  0.54% within
  cold 32768 s                             loss  -0.18% main's range  0.37% faster
* cold 38400 s (mean of two)               loss  -0.09% main's range  0.44% within
  cold 65536 s                             loss  -0.11% main's range  0.67% within
  openai fibonacci-raw t=1.0 tok/s         loss  -0.48% main's range  2.07% within
  openai gpu-chat-no-think t=1.0 tok/s     loss  +0.81% main's range  3.66% within
  openai fibonacci-raw t=0.0 tok/s         loss  +3.14% main's range  1.65% slower
  openai gpu-chat-no-think t=0.0 tok/s     loss  -0.28% main's range  1.05% within
* decode x1 per stream tok/s               loss  +1.29% main's range  1.44% open
* decode x4 per stream tok/s               loss  +0.61% main's range  0.00% slower
* decode x4 aggregate tok/s                loss  -0.09% main's range  0.66% within
* decode x8 per stream tok/s               loss  -0.68% main's range  2.03% within
* decode x8 aggregate tok/s                loss  -0.49% main's range  2.12% within
  edit x1 per stream tok/s                 loss  -1.80% main's range  6.69% within
  edit x4 per stream tok/s                 loss  -1.22% main's range  2.33% within
  edit x4 aggregate tok/s                  loss  -0.06% main's range  2.63% within
gate figures: 7, main's widest range 2.12% (decode x8 aggregate tok/s); verdicts {'faster': 1, 'within': 14, 'open': 1, 'slower': 2}; exactness failures 0
