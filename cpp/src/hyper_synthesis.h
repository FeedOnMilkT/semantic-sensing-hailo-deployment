/*
 * hyper_synthesis.h -- deterministic h_s (hyper-synthesis) evaluation.
 *
 * WHY THIS EXISTS
 * ---------------
 * The sender's Hailo int8 h_s and the FP32 PyTorch h_s produce DIFFERENT
 * scales.  After quantisation to the 64-entry CDF scale table, ~17% of the
 * elements land on a DIFFERENT CDF index.  rANS decoding requires the encoder
 * and the decoder to agree on the index of EVERY element, so a receiver that
 * recomputed h_s itself (in TensorRT or PyTorch) would desynchronise the
 * arithmetic decoder.
 *
 * Therefore h_s is computed ONCE, in ONE place: this C++ code, shared by the
 * sender and the receiver through libsscodec.  It is never executed on the
 * Hailo device (sender_bottleneck_hs.hef is retired) and never recomputed by
 * the receiver's network.
 *
 * DETERMINISM CONTRACT
 * --------------------
 * The output must be BIT-IDENTICAL on:
 *   - the sender build (-O2, possibly aarch64 / Raspberry Pi),
 *   - the receiver build (x86-64 + TensorRT host code),
 *   - any debug build (-O0) and any aggressive build (-O3 -march=native).
 *
 * Rules obeyed here:
 *   1. float32 arithmetic only.
 *   2. Hand-written loops with a FIXED accumulation order (no BLAS, no
 *      OpenMP, no -ffast-math, no reassociation).
 *   3. Compiled with -ffp-contract=off so the compiler may NOT fuse a*b+c
 *      into an FMA.  (On aarch64 GCC contracts by default, which would make
 *      the aarch64 sender disagree with the x86-64 receiver.)
 *   4. No transcendental functions: the h_s of this model is
 *      Conv -> BatchNorm -> ReLU -> bilinear-upsample -> Conv -> BN -> ReLU
 *      -> Conv, so exp/log/sqrt are not needed in the convolution path.  The
 *      single sqrt is in the BatchNormalisation, whose result is a constant
 *      per channel and is therefore folded into the stored gamma/beta by the
 *      exporter (see below) -- the runtime code does no sqrt at all.
 *   5. The layer sequence mirrors PyTorch's module sequence EXACTLY (the
 *      BatchNorm is NOT folded into the preceding convolution).  Folding
 *      would change the floating-point rounding of the convolution, and
 *      keeping the ops separate gives the smallest possible deviation from
 *      the PyTorch reference.  The BN's affine transform is pre-computed into
 *      (a, b) with a = gamma / sqrt(var + eps), b = beta - mean * a, which is
 *      exactly PyTorch's inference-time fused form.
 *
 * The bilinear upsample reproduces PyTorch's F.interpolate(mode="bilinear",
 * align_corners=False) EXACTLY:
 *
 *      scale = 1 / upsample_factor
 *      src   = scale * (dst + 0.5) - 0.5
 *      i0    = floor(src);  i1 = min(i0 + 1, in_size - 1)
 *      w1    = src - i0 ;    w0  = 1 - w1
 *      out   = w0*(w0p*src0 + w1p*src1) + w1*(w0p*src0' + w1p*src1')
 */
#ifndef SSCODEC_HYPER_SYNTHESIS_H
#define SSCODEC_HYPER_SYNTHESIS_H

#include <cstdint>
#include <vector>

#include "tables_format.h"

namespace sscodec {

/* One deterministic h_s layer (see tables_format.h for the on-disk layout). */
struct HsLayer {
  uint32_t kind = 0;   /* HsLayerKind */
  uint32_t flags = 0;  /* HsLayerFlags */
  uint32_t in_channels = 0;
  uint32_t out_channels = 0;
  uint32_t kernel_h = 0, kernel_w = 0;
  uint32_t stride_h = 0, stride_w = 0;
  uint32_t pad_h = 0, pad_w = 0;
  float scalar0 = 0.0f; /* BN: eps (already folded into a/b); Upsample: factor */
  std::vector<float> weight;
  std::vector<float> bias;

  /* BN layers carry gamma/beta/mean/var in `weight`; we also keep the
   * pre-folded (a, b) in `bias` so the runtime never takes a sqrt. */
  bool has_bn_affine = false;
};

/*
 * The h_s network: a linear sequence of layers applied to the z_hat tensor.
 *
 * For this model the layer list is:
 *    CONV(128->128, k3, s1, p1)   (no bias)
 *    BN(128)                        (gamma/beta/mean/var, eps; ReLU after)
 *    UPSAMPLE x2 (bilinear)
 *    CONV(128->128, k3, s1, p1)   (no bias)
 *    BN(128)                        (gamma/beta/mean/var, eps; ReLU after)
 *    CONV(128->384, k3, s1, p1)   (with bias)
 */
class HyperSynthesis {
 public:
  void set_layers(std::vector<HsLayer> layers) { layers_ = std::move(layers); }
  const std::vector<HsLayer> &layers() const { return layers_; }

  /*
   * Evaluate h_s.
   *
   *   z_hat : input,  NCHW, (1, C_in, z_h, z_w), integer-valued float32.
   *   out   : output, NCHW, (1, C_out, y_h, y_w).
   *
   * The output spatial size is determined by the layers themselves (the
   * upsample doubles it); y_h/y_w must match.
   */
  void forward(const float *z_hat, int z_h, int z_w, float *out, int y_h,
               int y_w) const;

  int out_channels() const {
    return layers_.empty() ? 0 : static_cast<int>(layers_.back().out_channels);
  }

 private:
  std::vector<HsLayer> layers_;
};

}  // namespace sscodec

#endif /* SSCODEC_HYPER_SYNTHESIS_H */
