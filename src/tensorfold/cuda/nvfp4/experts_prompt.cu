// Grouped NVFP4 experts, prompt form: items of 64 pairs x 128 columns staged through shared memory, each weight
// exact in bf16 (an e2m1 code times its e4m3 block scale), each 32-input block's mma sum added in fp32 in block order,
// the (expert, matrix) scale last.
// A pair's bits depend on its row and its expert alone (not on the item, the chunk or the other pairs), and are not
// decode's (experts.cu scales each 16-input product in fp32): a prompt row is only ever compared with prompt rows.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <torch/extension.h>

#include "../experts.cuh"

namespace {

constexpr int BLOCK4 = 36;           // uint4 a (32 columns, 32 inputs) block: 32 lanes' code words, then 4 of scales

// bf16x2 of the e2m1 nibbles at bits [s, s + 4) and [16 + s, 20 + s): fields into bf16's, times 2^126 (exact).
__device__ __forceinline__ uint32_t fp4pair(uint32_t w, int s) {
  const uint32_t v = w >> s;
  const uint32_t t = ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7E807E80u), "r"(0x80008000u));
  return r;
}

// bf16x2 (s, s) of one e4m3 byte: its fields into a bf16, times 2^120 (exact).
__device__ __forceinline__ uint32_t e4m3x2(uint32_t b) {
  const uint32_t x = b * 0x10001u;
  const uint32_t t = ((x & 0x007F007Fu) << 4) | ((x & 0x00800080u) << 8);
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x7B807B80u), "r"(0x80008000u));
  return r;
}

__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}

template <int M, int RT, int WM, int WN, int STAGES_>
struct Pro {
  static constexpr int THREADS = WM * WN * 32, BM = 16 * RT * WM, SG = 2, XC = 8, WB = M * BLOCK4;
  static constexpr int XU = BM * XC, WU = WN * SG * WB, SU = XU + WU;   // uint4 a stage: X rows, weight blocks
  static constexpr int XPT = (XU + THREADS - 1) / THREADS;
  static constexpr int STAGES = STAGES_;
  static constexpr int SMEM = STAGES * SU * 16;
  // a row's 8 chunks of 8 inputs, swizzled so the four rows a half-warp reads (32 bytes each) take all 32 banks
  static __device__ __forceinline__ int xslot(int r, int c) { return r * XC + (c ^ ((r & 3) << 1)); }
};

// Pair p reads X row p / slots when slots > 0 (X holds tokens), else X row p; items of expert ``skip`` are left alone.
template <int M, int EPI, int RT, int WM, int WN, int STAGES_>
__global__ void __launch_bounds__(WM * WN * 32, 2)
    nvfp4_prompt_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  using P = Pro<M, RT, WM, WN, STAGES_>;
  constexpr int STAGES = P::STAGES;
  extern __shared__ uint4 sm[];
  const int nbt = (NB + WN - 1) / WN, it = blockIdx.x / nbt, cbt = blockIdx.x - it * nbt;
  if (it >= __ldg(counts)) return;
  const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
  if (e == skip) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp - wm * WN;
  const int t = lane & 3, gq = lane >> 2, cb = cbt * WN + wn, cbs = min(WN, NB - cbt * WN);
  const __nv_bfloat16* xsrc[P::XPT];
  int xdst[P::XPT];
#pragma unroll
  for (int i = 0; i < P::XPT; ++i) {
    const int q = tid + i * P::THREADS, r = q / P::XC, c = q - r * P::XC;
    xdst[i] = P::xslot(r, c);
    xsrc[i] = nullptr;
    if (q < P::XU && r < cnt) {
      const int p = __ldg(members + first + r);
      xsrc[i] = X + (size_t)(slots ? p / slots : p) * x_stride + 8 * c;
    }
  }
  const uint4* wsrc = W + ((size_t)e * NB + cbt * WN) * (size_t)KG * P::WB;
  auto stage = [&](int s, int g0) {                  // blocks g0 .. g0 + ng - 1 (ng < SG only at an odd end)
    const int ng = min(P::SG, KG - g0);
    uint4* xs = sm + s * P::SU;
#pragma unroll
    for (int i = 0; i < P::XPT; ++i)
      if (xsrc[i] && ((tid + i * P::THREADS) % P::XC) < ng * 4) cp16(xs + xdst[i], xsrc[i] + (size_t)g0 * 32);
    uint4* ws = xs + P::XU;
    for (int q = tid; q < cbs * P::SG * P::WB; q += P::THREADS) {
      const int j = q / (P::SG * P::WB), o = q - j * P::SG * P::WB;
      if (o < ng * P::WB) cp16(ws + q, wsrc + ((size_t)j * KG + g0) * P::WB + o);
    }
  };
  const int base = 16 * RT * wm;
  const int nt = max(0, min(RT, (cnt - base + 15) >> 4));
  const bool active = nt > 0 && cb < NB;
  float acc[M][RT][NTW][4];
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j) acc[m][r][j][0] = acc[m][r][j][1] = acc[m][r][j][2] = acc[m][r][j][3] = 0.f;
  const int steps = (KG + P::SG - 1) / P::SG;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) stage(s, s * P::SG);
    cp_commit();
  }
  // the B lane's column in each n8 tile is gq: its block scales are quad gq / 2's, byte (h, tile, gq & 1)
  const int sq = gq >> 1, sc = 8 * (gq & 1);
  for (int k = 0; k < steps; ++k) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (k + STAGES - 1 < steps) stage((k + STAGES - 1) % STAGES, (k + STAGES - 1) * P::SG);
    cp_commit();
    if (!active) continue;
    const uint4* xs = sm + (k % STAGES) * P::SU;
#pragma unroll
    for (int gi = 0; gi < P::SG; ++gi) {
      if (k * P::SG + gi >= KG) break;
      const uint4* ws = xs + P::XU + (wn * P::SG + gi) * P::WB;
      uint4 wv[M];
      uint32_t sv[M][2][NTW];                         // (h, tile): the column's 16-input scale as bf16x2
#pragma unroll
      for (int m = 0; m < M; ++m) {
        wv[m] = ws[m * BLOCK4 + lane];
        const uint4 s4 = ws[m * BLOCK4 + 32 + sq];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int j = 0; j < NTW; ++j)
            sv[m][h][j] = e4m3x2((comp(s4, 2 * h + (j >> 1)) >> (16 * (j & 1) + sc)) & 0xFFu);
      }
      uint2 xa[2][RT], xb[2][RT];                      // rows gq and gq + 8: inputs 16h + 4t .. 16h + 4t + 3
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int r = 0; r < RT; ++r) {
          if (r >= nt) break;
          const int r0 = base + 16 * r + gq, c = 4 * gi + 2 * h + (t >> 1);
          xa[h][r] = reinterpret_cast<const uint2*>(xs + P::xslot(r0, c))[t & 1];
          xb[h][r] = reinterpret_cast<const uint2*>(xs + P::xslot(r0 + 8, c))[t & 1];
        }
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) {
          const uint32_t word = comp(wv[m], j);
          uint32_t b[2][2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            b[h][0] = mul2(fp4pair(word, 8 * h), sv[m][h][j]);
            b[h][1] = mul2(fp4pair(word, 8 * h + 4), sv[m][h][j]);
          }
#pragma unroll
          for (int r = 0; r < RT; ++r) {
            if (r >= nt) break;
            // a block's 32 inputs in the mma's own sum, then into the row's fp32 chain: long chains inside the mma
            // lose bits that the decode kernel's per-block fp32 adds keep
            float p[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int h = 0; h < 2; ++h) mma(p, xa[h][r].x, xb[h][r].x, xa[h][r].y, xb[h][r].y, b[h][0], b[h][1]);
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][r][j][q] += p[q];
          }
        }
    }
  }
  if (!active) return;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    const float g = __ldg(scale + e * M + m);
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
      for (int j = 0; j < NTW; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[m][r][j][q] *= g;
  }
#pragma unroll
  for (int r = 0; r < RT; ++r) {
    if (r >= nt) break;
    const int m0 = base + 16 * r + gq, m1 = m0 + 8;
    const bool v0 = m0 < cnt, v1 = m1 < cnt;
    const int p0 = v0 ? __ldg(members + first + m0) : 0, p1 = v1 ? __ldg(members + first + m1) : 0;
    epilogue<EPI, M, RT>(acc, r, out, N, cb * COLS + 2 * t, p0, p1, v0, v1, limit);
  }
}

template <int M, int EPI, int RT, int WM, int WN, int STAGES>
void launch(const at::Tensor& x, int x_stride, int slots, const at::Tensor& w, const at::Tensor& scale, int kg, int nb,
            const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int n,
            float limit, int skip, int64_t max_items) {
  using P = Pro<M, RT, WM, WN, STAGES>;
  auto* kern = nvfp4_prompt_kernel<M, EPI, RT, WM, WN, STAGES>;
  static bool ready = false;
  if (!ready) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM));
    ready = true;
  }
  const int64_t grid = max_items * ((nb + WN - 1) / WN);
  if (grid < 1) return;
  kern<<<static_cast<unsigned>(grid), P::THREADS, P::SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x_stride, slots,
      reinterpret_cast<const uint4*>(w.data_ptr()), scale.data_ptr<float>(), kg, nb, items.data_ptr<int>(),
      counts.data_ptr<int>(), members.data_ptr<int>(), out.data_ptr(), n, limit, skip);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void nvfp4_experts_prompt_cuda(int64_t epi, const at::Tensor& x, int64_t x_stride, int64_t slots, const at::Tensor& w,
                               const at::Tensor& scale, int64_t kg, int64_t nb, const at::Tensor& items,
                               const at::Tensor& counts, const at::Tensor& members, at::Tensor& out, int64_t n,
                               double limit, int64_t skip, int64_t max_items) {
  const c10::cuda::CUDAGuard guard(x.device());
  const int xs = static_cast<int>(x_stride), sl = static_cast<int>(slots), k = static_cast<int>(kg);
  const int b = static_cast<int>(nb), nn = static_cast<int>(n), sk = static_cast<int>(skip);
  const float lim = static_cast<float>(limit);
  // 8 warps a CTA, 64 pairs x 128 columns: two row tiles a warp, two warps down, four across
#define TF_PROMPT(M_, EPI_, STAGES_) \
  launch<M_, EPI_, 2, 2, 4, STAGES_>(x, xs, sl, w, scale, k, b, items, counts, members, out, nn, lim, sk, max_items)
  if (epi == 2) TF_PROMPT(2, 2, 2);       // two stages keep two CTAs an SM (three would not fit in its shared memory)
  else if (epi == 0) TF_PROMPT(1, 0, 3);
  else if (epi == 3) TF_PROMPT(1, 3, 3);
  else TORCH_CHECK(false, "nvfp4 prompt experts: epilogue 0 (fp32 down), 2 (SwiGLU) or 3 (bf16 down), not ", epi);
#undef TF_PROMPT
}
