| TP=3, mean of 4 runs of main and 3 of this head | main | this head | this head against main (+ slower, − faster) |
|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 / 8 streams | 69.1 / 24.7 / 15.0 | 69.7 / 24.8 / 14.9 | −0.88% / −0.17% / +0.50% |
| decode, all streams' tok/s, 4 / 8 streams | 91.6 / 99.8 | 91.8 / 100.1 | −0.28% / −0.24% |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 5.343 / 20.675 | 5.342 / 20.652 | −0.02% / −0.11% |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.089 / 4.592 / 17.664 / 35.220 | 1.094 / 4.590 / 17.643 / 35.183 | +0.45% / −0.06% / −0.12% / −0.11% |
| `bench_openai`, t=0, code / chat (tok/s) | 62.8 / 51.7 | 63.1 / 52.0 | −0.41% / −0.55% |
| `bench_openai`, t=1, code / chat (tok/s) | 51.5 / 46.7 | 51.1 / 47.3 | +0.84% / −1.10% |

| TP=3, in run order | main | this head | main | this head | main | this head | main |
|---|---|---|---|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 / 8 streams | 68.5 / 24.6 / 14.9 | 70.1 / 24.7 / 14.6 | 69.1 / 24.8 / 15.0 | 69.9 / 24.8 / 15.1 | 69.0 / 24.6 / 14.9 | 69.2 / 24.8 / 15.0 | 69.9 / 24.9 / 15.1 |
| decode, all streams' tok/s, 4 / 8 streams | 92.3 / 98.5 | 92.0 / 99.4 | 91.2 / 100.3 | 91.2 / 100.5 | 90.6 / 100.2 | 92.3 / 100.3 | 92.2 / 100.3 |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 5.327 / 20.587 | 5.346 / 20.657 | 5.364 / 20.747 | 5.343 / 20.680 | 5.339 / 20.692 | 5.335 / 20.618 | 5.341 / 20.676 |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.087 / 4.586 / 17.574 / 35.077 | 1.089 / 4.584 / 17.631 / 35.198 | 1.099 / 4.596 / 17.713 / 35.344 | 1.095 / 4.599 / 17.678 / 35.239 | 1.086 / 4.589 / 17.674 / 35.222 | 1.098 / 4.586 / 17.619 / 35.111 | 1.085 / 4.599 / 17.695 / 35.238 |
| `bench_openai`, t=0, code / chat (tok/s) | 62.3 / 51.5 | 62.2 / 52.2 | 62.3 / 51.5 | 63.8 / 51.7 | 62.7 / 52.3 | 63.3 / 52.2 | 64.2 / 51.5 |
| `bench_openai`, t=1, code / chat (tok/s) | 50.8 / 46.2 | 52.1 / 47.0 | 52.1 / 46.5 | 50.3 / 47.0 | 51.3 / 46.8 | 51.0 / 47.8 | 52.0 / 47.4 |

  cold 2048 s                              loss  +0.45% main's range  1.27% within
  cold 8192 s                              loss  -0.06% main's range  0.29% within
* cold 9600 s (mean of two)                loss  -0.02% main's range  0.69% within
  cold 32768 s                             loss  -0.12% main's range  0.79% within
* cold 38400 s (mean of two)               loss  -0.11% main's range  0.77% within
  cold 65536 s                             loss  -0.11% main's range  0.76% within
  openai fibonacci-raw t=1.0 tok/s         loss  +0.84% main's range  2.50% within
  openai gpu-chat-no-think t=1.0 tok/s     loss  -1.10% main's range  2.39% within
  openai fibonacci-raw t=0.0 tok/s         loss  -0.41% main's range  3.06% within
  openai gpu-chat-no-think t=0.0 tok/s     loss  -0.55% main's range  1.56% within
* decode x1 per stream tok/s               loss  -0.88% main's range  2.03% within
* decode x4 per stream tok/s               loss  -0.17% main's range  1.21% within
* decode x4 aggregate tok/s                loss  -0.28% main's range  1.86% within
* decode x8 per stream tok/s               loss  +0.50% main's range  1.34% within
* decode x8 aggregate tok/s                loss  -0.24% main's range  1.80% within
  edit x1 per stream tok/s                 loss  -0.58% main's range  1.46% within
  edit x4 per stream tok/s                 loss  +0.32% main's range  1.94% within
  edit x4 aggregate tok/s                  loss  -0.19% main's range  2.59% within
gate figures: 7, main's widest range 2.03% (decode x1 per stream tok/s); verdicts {'faster': 0, 'within': 18, 'open': 0, 'slower': 0}; exactness failures 0
