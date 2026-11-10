| TP=3, mean of 3 runs of main and 2 of this head | main | this head | this head against main (+ slower, − faster) |
|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 / 8 streams | 67.3 / 24.2 / 14.7 | 68.0 / 24.1 / 14.7 | −0.92% / +0.62% / −0.23% |
| decode, all streams' tok/s, 4 / 8 streams | 89.5 / 97.6 | 88.8 / 97.9 | +0.76% / −0.27% |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 5.336 / 20.695 | 5.336 / 20.654 | +0.01% / −0.20% |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.089 / 4.592 / 17.754 / 35.400 | 1.085 / 4.584 / 17.652 / 35.307 | −0.35% / −0.19% / −0.57% / −0.26% |
| `bench_openai`, t=0, code / chat (tok/s) | 64.2 / 51.0 | 63.9 / 51.2 | +0.58% / −0.33% |
| `bench_openai`, t=1, code / chat (tok/s) | 51.9 / 46.0 | 51.9 / 46.2 | −0.03% / −0.32% |

| TP=3, in run order | main | this head | main | this head | main |
|---|---|---|---|---|---|
| decode, a stream's tok/s, 1 / 4 / 8 streams | 65.9 / 23.9 / 14.4 | 67.9 / 23.8 / 14.7 | 68.2 / 24.3 / 14.9 | 68.0 / 24.3 / 14.7 | 67.9 / 24.4 / 14.7 |
| decode, all streams' tok/s, 4 / 8 streams | 88.3 / 96.5 | 88.2 / 97.7 | 90.2 / 98.7 | 89.5 / 98.1 | 90.1 / 97.7 |
| cold 9,600 / 38,400, `ttft_s` (two prompts' mean) | 5.331 / 20.744 | 5.325 / 20.599 | 5.314 / 20.598 | 5.348 / 20.710 | 5.364 / 20.745 |
| cold 2K / 8K / 32K / 64K, `ttft_s` | 1.098 / 4.593 / 17.961 / 35.567 | 1.080 / 4.581 / 17.609 / 35.223 | 1.081 / 4.581 / 17.596 / 35.171 | 1.089 / 4.587 / 17.695 / 35.392 | 1.087 / 4.603 / 17.704 / 35.461 |
| `bench_openai`, t=0, code / chat (tok/s) | 64.2 / 51.0 | 64.0 / 51.0 | 64.4 / 51.4 | 63.7 / 51.3 | 64.1 / 50.6 |
| `bench_openai`, t=1, code / chat (tok/s) | 51.9 / 45.8 | 51.8 / 46.0 | 52.1 / 46.2 | 52.0 / 46.3 | 51.7 / 46.0 |

  cold 2048 s                              loss  -0.35% main's range  1.57% within
  cold 8192 s                              loss  -0.19% main's range  0.49% within
* cold 9600 s (mean of two)                loss  +0.01% main's range  0.93% within
  cold 32768 s                             loss  -0.57% main's range  2.05% within
* cold 38400 s (mean of two)               loss  -0.20% main's range  0.71% within
  cold 65536 s                             loss  -0.26% main's range  1.12% within
  openai fibonacci-raw t=1.0 tok/s         loss  -0.03% main's range  0.65% within
  openai gpu-chat-no-think t=1.0 tok/s     loss  -0.32% main's range  0.91% within
  openai fibonacci-raw t=0.0 tok/s         loss  +0.58% main's range  0.44% slower
  openai gpu-chat-no-think t=0.0 tok/s     loss  -0.33% main's range  1.67% within
* decode x1 per stream tok/s               loss  -0.92% main's range  3.42% within
* decode x4 per stream tok/s               loss  +0.62% main's range  2.07% within
* decode x4 aggregate tok/s                loss  +0.76% main's range  2.12% within
* decode x8 per stream tok/s               loss  -0.23% main's range  3.41% within
* decode x8 aggregate tok/s                loss  -0.27% main's range  2.25% within
  edit x1 per stream tok/s                 loss  -1.88% main's range  7.71% within
  edit x4 per stream tok/s                 loss  +1.43% main's range  1.65% open
  edit x4 aggregate tok/s                  loss  +2.56% main's range  2.92% open
gate figures: 7, main's widest range 3.42% (decode x1 per stream tok/s); verdicts {'faster': 0, 'within': 15, 'open': 2, 'slower': 1}; exactness failures 0
