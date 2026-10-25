/*
 * hyper_synthesis.cpp -- deterministic h_s evaluation.
 *
 * See hyper_synthesis.h for the determinism contract and the rationale.
 *
 * Numerics notes
 * --------------
 *  * The convolution accumulates in float32 in the order
 *    (oc, oy, ox) x (ic, ky, kx) -- a fixed, explicit loop nest.  No BLAS, no
 *    OpenMP, no reassociation.  The reference (PyTorch CPU / oneDNN) also
 *    accumulates in float32.
 *  * The BatchNorm affine (a, b) with a = gamma / sqrt(var + eps) and
 *    b = beta - mean * a is PRE-COMPUTED by the exporter, so the runtime
 *    performs NO square root and NO division at all.  This makes the runtime
 *    path free of transcendentals and identical on every IEEE-754 platform.
 *  * The bilinear upsample follows PyTorch's align_corners=False formula.
 */
#include "hyper_synthesis.h"

#include <cmath>
#include <cstring>

namespace sscodec {
namespace {

/*
 * ReLU, bit-exact with PyTorch's for finite inputs.
 * (For -0.0f this returns +0.0f, matching torch.relu.)
 */
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

/*
 * Convolution, NCHW, stride 1, padding `pad`, optional bias, optional ReLU.
 * Fixed accumulation order (ic, ky, kx) -> bit-reproducible.
 *
 * PERFORMANCE NOTE (bit-exactness preserved)
 * ------------------------------------------
 * The naive form does `if (ix < 0 || ix >= in_w) continue;` inside the
 * innermost loop, which costs a branch per MAC and blocks vectorisation.
 * Here the valid kx range is computed ONCE per output column, and the
 * accumulation order is still exactly (ic, ky, kx) with the same float32
 * adds -- the skipped terms are exactly the same terms (padding), so the
 * result is bit-identical to the naive form.
 *
 * The kernel weights are re-indexed as w[oc][ic][ky][kx] (OIHW, exactly the
 * PyTorch layout), so the weight pointer arithmetic is also branch-free.
 */
void conv2d_nchw(const float *in, int in_c, int in_h, int in_w, const float *wt,
                 const float *bias, int out_c, int kh, int kw, int pad_h,
                 int pad_w, float *out, int out_h, int out_w, bool apply_relu) {
  /* Pre-compute, for every output column, the KERNEL-index range that maps
   * to a valid input column:
   *     ix = ox + kx - pad_w  must satisfy  0 <= ix < in_w
   *   =>  kx in [pad_w - ox, pad_w - ox + in_w)  intersected with [0, kw).
   */
  std::vector<int> kx_begin(out_w), kx_end(out_w);
  for (int ox = 0; ox < out_w; ++ox) {
    int b = pad_w - ox;
    if (b < 0) b = 0;
    int e = pad_w - ox + in_w;
    if (e > kw) e = kw;
    kx_begin[ox] = b;
    kx_end[ox] = e;
  }

  const size_t w_ic_stride = static_cast<size_t>(kh) * kw;
  const size_t w_oc_stride = static_cast<size_t>(in_c) * w_ic_stride;

  for (int oc = 0; oc < out_c; ++oc) {
    const float *w_oc = wt + static_cast<size_t>(oc) * w_oc_stride;
    const float b0 = bias ? bias[oc] : 0.0f;
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        float acc = b0;
        for (int ic = 0; ic < in_c; ++ic) {
          const float *in_row = in + static_cast<size_t>(ic) * in_h * in_w;
          const float *w_ic = w_oc + static_cast<size_t>(ic) * w_ic_stride;
          for (int ky = 0; ky < kh; ++ky) {
            const int iy = oy + ky - pad_h;
            if (iy < 0 || iy >= in_h) continue;
            const float *src = in_row + static_cast<size_t>(iy) * in_w;
            const float *wp = w_ic + static_cast<size_t>(ky) * kw;
            for (int kx = kx_begin[ox]; kx < kx_end[ox]; ++kx) {
              const int ix = ox + kx - pad_w;
              acc += src[ix] * wp[kx];
            }
          }
        }
        out[(static_cast<size_t>(oc) * out_h + oy) * out_w + ox] =
            apply_relu ? relu(acc) : acc;
      }
    }
  }
}

/*
 * BatchNorm in inference mode.
 *
 * PyTorch's inference formula is
 *     out = (x - mean) * invstd * gamma + beta
 * (aten::batch_norm's CPU kernel computes (x - mean) * invstd * gamma + beta,
 * in that association order).
 * We reproduce that association EXACTLY, so the numerical difference against
 * the PyTorch reference is at the level of a single multiply rounding, not a
 * whole chain.
 *
 * `invstd` is pre-computed by the exporter as 1/sqrt(var + eps), so the
 * runtime performs NO square root and NO division.
 */
void batchnorm_affine(const float *in, int c, int h, int w, const float *gamma,
                      const float *beta, const float *mean, const float *invstd,
                      float *out) {
  for (int ch = 0; ch < c; ++ch) {
    const float g = gamma[ch], b = beta[ch], m = mean[ch], iv = invstd[ch];
    const float *src = in + static_cast<size_t>(ch) * h * w;
    float *dst = out + static_cast<size_t>(ch) * h * w;
    for (int i = 0; i < h * w; ++i) {
      dst[i] = (src[i] - m) * iv * g + b;
    }
  }
}

/*
 * Bilinear upsample, NCHW, align_corners=False (PyTorch default).
 *
 * PyTorch's `area_pixel_compute_source_index` (aten/src/ATen/native/cpu/
 * UpSampleKernel.cpp) is:
 *
 *     src = scale * (dst + 0.5) - 0.5          scale = in_size / out_size
 *     if (src < 0) src = 0                      <-- CLAMPED, not floored!
 *     i0   = floor(src)
 *     i1   = i0 + (i0 < in_size - 1 ? 1 : 0)
 *     l1   = src - i0
 *     l0   = 1 - l1
 *     out  = l0*(l0x*I[i0][j0] + l1x*I[i0][j1])
 *          + l1*(l0x*I[i1][j0] + l1x*I[i1][j1])
 *
 * The clamp matters: with in=8, out=16 the first output pixel has
 * src = 0.5*0.5 - 0.5 = -0.25, which must become 0 (l1 = 0), NOT
 * floor(-0.25) = -1 with a negative lambda.
 *
 * The interpolation is separable (PyTorch does the x pass first, then y),
 * which keeps the rounding identical.
 */
void upsample_bilinear2d(const float *in, int c, int in_h, int in_w, float *out,
                         int out_h, int out_w) {
  const float rh = static_cast<float>(in_h) / static_cast<float>(out_h);
  const float rw = static_cast<float>(in_w) / static_cast<float>(out_w);

  std::vector<int> y0(out_h), y1(out_h), x0(out_w), x1(out_w);
  std::vector<float> wy0(out_h), wy1(out_h), wx0(out_w), wx1(out_w);
  for (int oy = 0; oy < out_h; ++oy) {
    float src = rh * (static_cast<float>(oy) + 0.5f) - 0.5f;
    if (src < 0.0f) src = 0.0f; /* PyTorch's clamp */
    const float f0 = std::floor(src);
    const int i0 = static_cast<int>(f0);
    const int i1 = (i0 < in_h - 1) ? (i0 + 1) : in_h - 1;
    const float l1 = src - f0;
    y0[oy] = i0;
    y1[oy] = i1;
    wy0[oy] = 1.0f - l1;
    wy1[oy] = l1;
  }
  for (int ox = 0; ox < out_w; ++ox) {
    float src = rw * (static_cast<float>(ox) + 0.5f) - 0.5f;
    if (src < 0.0f) src = 0.0f; /* PyTorch's clamp */
    const float f0 = std::floor(src);
    const int i0 = static_cast<int>(f0);
    const int i1 = (i0 < in_w - 1) ? (i0 + 1) : in_w - 1;
    const float l1 = src - f0;
    x0[ox] = i0;
    x1[ox] = i1;
    wx0[ox] = 1.0f - l1;
    wx1[ox] = l1;
  }

  /*
   * PyTorch's CPU kernel interpolates along x first, then along y.  We do the
   * same: for each output row oy we interpolate the two source rows y0[oy]
   * and y1[oy] horizontally, then combine them with the y weights.
   */
  for (int ch = 0; ch < c; ++ch) {
    const float *src = in + static_cast<size_t>(ch) * in_h * in_w;
    float *dst = out + static_cast<size_t>(ch) * out_h * out_w;
    for (int oy = 0; oy < out_h; ++oy) {
      const float *r0 = src + static_cast<size_t>(y0[oy]) * in_w;
      const float *r1 = src + static_cast<size_t>(y1[oy]) * in_w;
      float *drow = dst + static_cast<size_t>(oy) * out_w;
      for (int ox = 0; ox < out_w; ++ox) {
        const float a = r0[x0[ox]] * wx0[ox] + r0[x1[ox]] * wx1[ox];
        const float b = r1[x0[ox]] * wx0[ox] + r1[x1[ox]] * wx1[ox];
        drow[ox] = wy0[oy] * a + wy1[oy] * b;
      }
    }
  }
}

}  // namespace

void HyperSynthesis::forward(const float *z_hat, int z_h, int z_w, float *out,
                             int y_h, int y_w) const {
  (void)y_h;
  (void)y_w;
  if (layers_.empty()) return;

  const size_t in_elems = static_cast<size_t>(layers_.front().in_channels) * z_h * z_w;
  std::vector<float> a(in_elems), b(in_elems);
  /* the input IS the first layer's input */
  std::memcpy(a.data(), z_hat, sizeof(float) * in_elems);

  int cur_c = static_cast<int>(layers_.front().in_channels);
  int cur_h = z_h, cur_w = z_w;

  for (size_t li = 0; li < layers_.size(); ++li) {
    const HsLayer &L = layers_[li];
    const bool relu_after = (L.flags & kHsFlagReluAfter) != 0;

    switch (L.kind) {
      case kHsConv: {
        const int oc = static_cast<int>(L.out_channels);
        const bool has_bias = (L.flags & kHsFlagHasBias) != 0;
        const size_t wcount = static_cast<size_t>(oc) * L.in_channels * L.kernel_h *
                              L.kernel_w;
        const float *wt = L.weight.data();
        const float *bs = has_bias ? L.weight.data() + wcount : nullptr;
        const size_t need = static_cast<size_t>(oc) * cur_h * cur_w;
        if (b.size() < need) b.resize(need);
        conv2d_nchw(a.data(), cur_c, cur_h, cur_w, wt, bs, oc,
                    static_cast<int>(L.kernel_h), static_cast<int>(L.kernel_w),
                    static_cast<int>(L.pad_h), static_cast<int>(L.pad_w), b.data(),
                    cur_h, cur_w, relu_after);
        cur_c = oc;
        a.swap(b);
        break;
      }
      case kHsBn: {
        const int c = static_cast<int>(L.out_channels);
        const size_t need = static_cast<size_t>(c) * cur_h * cur_w;
        if (b.size() < need) b.resize(need);
        /* weight = [gamma(C), beta(C), mean(C), invstd(C)] */
        const float *w = L.weight.data();
        batchnorm_affine(a.data(), c, cur_h, cur_w, w, w + c, w + 2 * c,
                         w + 3 * c, b.data());
        if (relu_after) {
          for (size_t i = 0; i < need; ++i) b[i] = relu(b[i]);
        }
        a.swap(b);
        break;
      }
      case kHsUpsample: {
        const int nh = cur_h * 2, nw = cur_w * 2;
        const size_t need = static_cast<size_t>(cur_c) * nh * nw;
        if (b.size() < need) b.resize(need);
        upsample_bilinear2d(a.data(), cur_c, cur_h, cur_w, b.data(), nh, nw);
        cur_h = nh;
        cur_w = nw;
        a.swap(b);
        break;
      }
      default:
        break;
    }
  }

  std::memcpy(out, a.data(),
              sizeof(float) * static_cast<size_t>(cur_c) * cur_h * cur_w);
}

}  // namespace sscodec
