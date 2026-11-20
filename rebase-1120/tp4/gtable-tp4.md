| TP=4 (ring, gather), drafted decode after a 1,040,384-token prompt, tok/s | main | this head | main | this head | main | this head | main |
|---|---|---|---|---|---|---|---|
| 3 reps of 512 tokens | 151.07 / 151.34 / 151.24 | 171.63 / 171.55 / 172.17 | 152.81 / 151.51 / 151.12 | 172.12 / 171.49 / 171.55 | 151.75 / 151.11 / 151.20 | 171.62 / 171.12 / 170.84 | 141.85 / 141.53 / 150.65 |

| TP=4 (ring, gather), decode round in ms, mean of 4 runs of main and 3 of this head | main | this head | this head against main (+ longer, − shorter) |
|---|---|---|---|
| 4K, 1 / 4 / 8 streams | 26.101 / 47.377 / 70.891 | 26.046 / 47.073 / 70.845 | −0.21% / −0.64% / −0.06% |
| 64K, 1 / 4 / 8 streams | 26.616 / 48.009 / 71.230 | 26.629 / 48.438 / 71.135 | +0.05% / +0.89% / −0.13% |
| 256K, 1 / 4 / 8 streams | 27.309 / 49.300 / 72.352 | 27.246 / 49.305 / 72.470 | −0.23% / +0.01% / +0.16% |
| 1016K, 1 / 4 / 8 streams | 30.227 / 62.117 / 98.728 | 30.195 / 61.862 / 98.644 | −0.10% / −0.41% / −0.08% |

| TP=4 (ring, gather), decode round in ms, in run order | main | this head | main | this head | main | this head | main |
|---|---|---|---|---|---|---|---|
| 4K, 1 / 4 / 8 streams | 26.126 / 47.063 / 70.867 | 26.015 / 47.049 / 70.778 | 26.098 / 47.077 / 70.850 | 26.070 / 47.083 / 70.866 | 26.083 / 47.732 / 70.929 | 26.051 / 47.087 / 70.891 | 26.095 / 47.637 / 70.917 |
| 64K, 1 / 4 / 8 streams | 26.673 / 48.055 / 71.168 | 26.637 / 48.925 / 71.076 | 26.642 / 48.031 / 71.298 | 26.623 / 48.321 / 71.037 | 26.629 / 47.920 / 71.103 | 26.628 / 48.066 / 71.293 | 26.519 / 48.031 / 71.353 |
| 256K, 1 / 4 / 8 streams | 27.361 / 49.292 / 72.594 | 27.178 / 49.362 / 72.603 | 27.297 / 49.336 / 72.253 | 27.249 / 49.345 / 72.309 | 27.212 / 49.277 / 72.272 | 27.312 / 49.207 / 72.498 | 27.364 / 49.293 / 72.287 |
| 1016K, 1 / 4 / 8 streams | 30.215 / 61.960 / 98.916 | 30.132 / 61.833 / 98.635 | 30.242 / 62.662 / 98.643 | 30.309 / 61.733 / 98.531 | 30.211 / 61.940 / 98.686 | 30.145 / 62.020 / 98.767 | 30.240 / 61.907 / 98.667 |

verdicts:
  grid 4K x1: -0.21%, main's range 0.17%, faster
  grid 4K x4: -0.64%, main's range 1.41%, within
  grid 4K x8: -0.06%, main's range 0.11%, faster
  grid 64K x1: +0.05%, main's range 0.58%, within
  grid 64K x4: +0.89%, main's range 0.28%, slower
  grid 64K x8: -0.13%, main's range 0.35%, within
  grid 256K x1: -0.23%, main's range 0.56%, within
  grid 256K x4: +0.01%, main's range 0.12%, within
  grid 256K x8: +0.16%, main's range 0.47%, within
  grid 1016K x1: -0.10%, main's range 0.10%, faster
  grid 1016K x4: -0.41%, main's range 1.21%, faster
  grid 1016K x8: -0.08%, main's range 0.28%, within
  grid geometric means over the cells each arm counted: rc4ma 47.851 ms, rc4ha 47.838 ms, rc4mb 47.861 ms, rc4hb 47.806 ms, rc4mc 47.831 ms, rc4hc 47.814 ms, rc4md 47.856 ms
  drafted rc4ma    main                tok/s 151.07 151.34 151.24  rounds 32,32,32  drafted 508,508,508  accepted 480,480,480  token_sha d50ec7cb16f5
  drafted rc4ha    head                tok/s 171.63 171.55 172.17  rounds 32,32,32  drafted 508,508,508  accepted 480,480,480  token_sha d50ec7cb16f5
  drafted rc4mb    main                tok/s 152.81 151.51 151.12  rounds 32,32,32  drafted 507,508,508  accepted 479,480,480  token_sha d50ec7cb16f5
  drafted rc4hb    head                tok/s 172.12 171.49 171.55  rounds 32,32,32  drafted 508,508,508  accepted 480,480,480  token_sha d50ec7cb16f5
  drafted rc4mc    main                tok/s 151.75 151.11 151.20  rounds 32,32,32  drafted 508,508,508  accepted 480,480,480  token_sha d50ec7cb16f5
  drafted rc4hc    head                tok/s 171.62 171.12 170.84  rounds 32,32,32  drafted 508,508,508  accepted 480,480,480  token_sha d50ec7cb16f5
  drafted rc4md    main                tok/s 141.85 141.53 150.65  rounds 32,32,32  drafted 508,508,508  accepted 480,480,480  token_sha d50ec7cb16f5
  drafted tokens: one token_sha over the 21 reps of 7 arms
  drafted STOP drafts differ between the counted arms (the speeds do not compare):
  drafted   rep 1: rc4ma 32/508/480, rc4ha 32/508/480, rc4mb 32/507/479, rc4hb 32/508/480, rc4mc 32/508/480, rc4hc 32/508/480, rc4md 32/508/480
  drafted   rep 2: rc4ma 32/508/480, rc4ha 32/508/480, rc4mb 32/508/480, rc4hb 32/508/480, rc4mc 32/508/480, rc4hc 32/508/480, rc4md 32/508/480
  drafted   rep 3: rc4ma 32/508/480, rc4ha 32/508/480, rc4mb 32/508/480, rc4hb 32/508/480, rc4mc 32/508/480, rc4hc 32/508/480, rc4md 32/508/480
verdicts: STOP: words needed
