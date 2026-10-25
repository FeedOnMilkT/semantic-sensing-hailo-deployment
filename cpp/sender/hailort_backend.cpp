/*
 * hailtort_backend.cpp -- the HailoRT (Hailo device) inference backend.
 *
 * ==========================================================================
 *  IMPORTANT: THIS FILE IS ONLY COMPILE-CHECKED ON THIS MACHINE.
 * ==========================================================================
 *  This build host has NO Hailo device and NO libhailort, so the backend is
 *  built with -DWITH_HAILORT=ON purely to verify that the HailoRT C++ API
 *  usage compiles against the upstream headers (hailo-ai/hailort, v4.23.0).
 *  It has NEVER been linked against a real libhailort and has NEVER run on
 *  silicon.  See cpp/README.md.
 *
 *  THE CHAIN
 *  ---------
 *      HEF #1 "backbone" : image(0..255, NHWC)  -> (x_h, x_l)
 *      HEF #2 "enc"      : (x_h, x_l)             -> (y, z_pre)
 *      HOST              : z_hat = floor(z_pre + 0.5) + median
 *      (no "hs" HEF any more -- the deterministic h_s lives in libsscodec)
 *
 *  VSTREAM MATCHING BY SHAPE, NOT BY NAME
 *  --------------------------------------
 *  The vstream names in the HEF depend on how the ONNX was exported, and the
 *  `enc` HEF's two inputs must be matched to the backbone's two outputs.  We
 *  therefore match purely on the TENSOR SHAPE:
 *
 *      input  (512,512,3)                  -> the image
 *      input  (64,64,128)  == (128,64,64)  -> x_h
 *      input  (32,32,256)  == (256,32,32)  -> x_l
 *      output (64,64,128)  == (128,64,64)  -> x_h
 *      output (32,32,256)  == (256,32,32)  -> x_l
 *      output (16,16,192)  == (192,16,16)  -> y
 *      output (8,8,128)    == (128,8,8)    -> z_pre
 *
 *  NHWC <-> NCHW: Hailo tensors are channels-LAST, the Python reference is
 *  channels-FIRST, so every tensor is transposed on the way in and out.
 *
 *  FLOAT32 OUTPUTS: the HEFs are int8-quantised; we ask HailoRT for
 *  de-quantised FLOAT32 outputs (HAILO_FORMAT_TYPE_FLOAT_32), so the host
 *  sees the same numbers the DFC emulator's SDK_QUANTIZED context produced
 *  (which is what the reference bitstreams were built from).
 *
 *  HailoRT 4.23 API notes (the API changed a lot since 4.x early releases):
 *    - everything is in the lowercase `hailort` namespace;
 *    - fallible constructors return `hailort::Expected<T>`; the value is
 *      obtained with `*result` (asserts on failure) or `result.value()`;
 *    - a ConfiguredNetworkGroup must be ACTIVATED before InferVStreams::create;
 *    - InferVStreams::infer() takes maps of name -> MemoryView.
 */
#ifdef SSCODEC_WITH_HAILORT

#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hailo/hailort.hpp"

#include "inference_backend.h"

namespace sender {
namespace {

/* Hailo is NHWC, we are NCHW. */
void NhwcToNchw(const float *src, int n, int h, int w, int c, float *dst) {
  for (int b = 0; b < n; ++b) {
    for (int ch = 0; ch < c; ++ch) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          dst[((static_cast<size_t>(b) * c + ch) * h + y) * w + x] =
              src[((static_cast<size_t>(b) * h + y) * w + x) * c + ch];
        }
      }
    }
  }
}

void NchwToNhwc(const float *src, int n, int c, int h, int w, float *dst) {
  for (int b = 0; b < n; ++b) {
    for (int ch = 0; ch < c; ++ch) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          dst[((static_cast<size_t>(b) * h + y) * w + x) * c + ch] =
              src[((static_cast<size_t>(b) * c + ch) * h + y) * w + x];
        }
      }
    }
  }
}

using Shape3 = std::vector<int>;

/* One configured + activated network, with its infer pipeline. */
struct HailoNet {
  std::unique_ptr<hailort::Hef> hef;
  std::shared_ptr<hailort::ConfiguredNetworkGroup> group;
  std::unique_ptr<hailort::ActivatedNetworkGroup> activated;
  std::unique_ptr<hailort::InferVStreams> pipe;
};

class HailoRTBackend : public InferenceBackend {
 public:
  HailoRTBackend(const std::string &hef_dir, std::string *err) {
    /*
     * ONE VDevice, TWO configured networks, run sequentially: the backbone
     * must finish before the enc consumes its outputs.
     */
    auto vdevice_exp = hailort::VDevice::create();
    if (!vdevice_exp.has_value()) {
      if (err) *err = "HailoRT: VDevice::create failed (no device?)";
      return;
    }
    vdevice_ = std::move(vdevice_exp.value());

    const std::string bb_hef = hef_dir + "/sender_backbone.hef";
    const std::string enc_hef = hef_dir + "/sender_bottleneck_enc.hef";

    if (!make_pipeline(bb_hef, &bb_, err)) return;
    if (!make_pipeline(enc_hef, &enc_, err)) return;

    ok_ = true;
  }

  bool infer(const uint8_t *image_u8_nhwc, SenderTensors *out,
             std::string *err) override {
    if (!ok_) {
      if (err) *err = "HailoRT backend not initialised";
      return false;
    }
    const int imgs = 1;

    /* ---- backbone: image (u8 0..255) -> (x_h, x_l) ---- */
    std::vector<float> img_f(static_cast<size_t>(imgs) * 512 * 512 * 3);
    for (size_t i = 0; i < img_f.size(); ++i) {
      img_f[i] = static_cast<float>(image_u8_nhwc[i]);
    }

    std::map<std::string, hailort::MemoryView> bb_in;
    std::map<std::string, hailort::MemoryView> bb_out;
    const std::string img_name = FindByShape(
        bb_.group->get_input_vstream_infos(), {512, 512, 3});
    if (img_name.empty()) {
      if (err) *err = "HailoRT: backbone input (512,512,3) not found";
      return false;
    }
    bb_in[img_name] = hailort::MemoryView(img_f.data(), img_f.size() * sizeof(float));

    const std::string xh_name = FindByShape(bb_.group->get_output_vstream_infos(),
                                            {64, 64, 128});
    const std::string xl_name = FindByShape(bb_.group->get_output_vstream_infos(),
                                            {32, 32, 256});
    if (xh_name.empty() || xl_name.empty()) {
      if (err) *err = "HailoRT: backbone outputs do not match the expected shapes";
      return false;
    }
    std::vector<float> xh_out(static_cast<size_t>(imgs) * 64 * 64 * 128, 0.0f);
    std::vector<float> xl_out(static_cast<size_t>(imgs) * 32 * 32 * 256, 0.0f);
    bb_out[xh_name] = hailort::MemoryView(xh_out.data(), xh_out.size() * sizeof(float));
    bb_out[xl_name] = hailort::MemoryView(xl_out.data(), xl_out.size() * sizeof(float));

    if (bb_.pipe->infer(bb_in, bb_out, imgs) != HAILO_SUCCESS) {
      if (err) *err = "HailoRT: backbone infer failed";
      return false;
    }

    /* ---- enc: (x_h, x_l) -> (y, z_pre) ---- */
    std::vector<float> xh_nchw(static_cast<size_t>(imgs) * 128 * 64 * 64);
    std::vector<float> xl_nchw(static_cast<size_t>(imgs) * 256 * 32 * 32);
    NhwcToNchw(xh_out.data(), imgs, 64, 64, 128, xh_nchw.data());
    NhwcToNchw(xl_out.data(), imgs, 32, 32, 256, xl_nchw.data());

    std::vector<float> xh_in(xh_nchw.size()), xl_in(xl_nchw.size());
    NchwToNhwc(xh_nchw.data(), imgs, 128, 64, 64, xh_in.data());
    NchwToNhwc(xl_nchw.data(), imgs, 256, 32, 32, xl_in.data());

    const std::string enc_xh = FindByShape(enc_.group->get_input_vstream_infos(),
                                           {64, 64, 128});
    const std::string enc_xl = FindByShape(enc_.group->get_input_vstream_infos(),
                                           {32, 32, 256});
    if (enc_xh.empty() || enc_xl.empty()) {
      if (err) *err = "HailoRT: enc inputs do not match the expected shapes";
      return false;
    }
    std::map<std::string, hailort::MemoryView> enc_in;
    enc_in[enc_xh] = hailort::MemoryView(xh_in.data(), xh_in.size() * sizeof(float));
    enc_in[enc_xl] = hailort::MemoryView(xl_in.data(), xl_in.size() * sizeof(float));

    const std::string enc_y = FindByShape(enc_.group->get_output_vstream_infos(),
                                          {16, 16, 192});
    const std::string enc_z = FindByShape(enc_.group->get_output_vstream_infos(),
                                          {8, 8, 128});
    if (enc_y.empty() || enc_z.empty()) {
      if (err) *err = "HailoRT: enc outputs do not match the expected shapes";
      return false;
    }
    std::vector<float> y_out(static_cast<size_t>(imgs) * 16 * 16 * 192, 0.0f);
    std::vector<float> z_out(static_cast<size_t>(imgs) * 8 * 8 * 128, 0.0f);
    std::map<std::string, hailort::MemoryView> enc_out;
    enc_out[enc_y] = hailort::MemoryView(y_out.data(), y_out.size() * sizeof(float));
    enc_out[enc_z] = hailort::MemoryView(z_out.data(), z_out.size() * sizeof(float));

    if (enc_.pipe->infer(enc_in, enc_out, imgs) != HAILO_SUCCESS) {
      if (err) *err = "HailoRT: enc infer failed";
      return false;
    }

    /* ---- collect (NHWC -> NCHW) ---- */
    out->x_h = xh_nchw;
    out->x_l = xl_nchw;
    out->y.assign(static_cast<size_t>(192) * 16 * 16, 0.0f);
    out->z_pre.assign(static_cast<size_t>(128) * 8 * 8, 0.0f);
    {
      std::vector<float> nchw(static_cast<size_t>(192) * 16 * 16);
      NhwcToNchw(y_out.data(), imgs, 16, 16, 192, nchw.data());
      out->y = std::move(nchw);
      std::vector<float> zc(static_cast<size_t>(128) * 8 * 8);
      NhwcToNchw(z_out.data(), imgs, 8, 8, 128, zc.data());
      out->z_pre = std::move(zc);
    }
    return true;
  }

  std::string name() const override { return "hailort"; }

 private:
  /*
   * Find the vstream whose NHWC shape matches `wanted` (e.g. {64,64,128}).
   * The name is never assumed -- only the shape is.
   */
  template <typename InfosT>
  static std::string FindByShape(const InfosT &infos_exp,
                                 const Shape3 &wanted) {
    if (!infos_exp.has_value()) return std::string();
    for (const auto &info : infos_exp.value()) {
      const auto &s = info.shape;
      /* hailo_3d_image_shape_t = {height, width, features} (NHWC). */
      if (static_cast<int>(s.height) == wanted[0] &&
          static_cast<int>(s.width) == wanted[1] &&
          static_cast<int>(s.features) == wanted[2]) {
        return info.name;
      }
    }
    return std::string();
  }

  bool make_pipeline(const std::string &hef_path, HailoNet *net,
                     std::string *err) {
    auto hef_exp = hailort::Hef::create(hef_path);
    if (!hef_exp.has_value()) {
      if (err) *err = "HailoRT: cannot create HEF " + hef_path;
      return false;
    }
    net->hef = std::make_unique<hailort::Hef>(hef_exp.value());

    auto groups_exp = vdevice_->configure(*net->hef);
    if (!groups_exp.has_value() || groups_exp.value().empty()) {
      if (err) *err = "HailoRT: configure returned no groups for " + hef_path;
      return false;
    }
    net->group = std::move(groups_exp.value().front());

    auto act_exp = net->group->activate();
    if (!act_exp.has_value()) {
      if (err) *err = "HailoRT: cannot activate the network group for " + hef_path;
      return false;
    }
    net->activated = std::move(act_exp.value());

    auto in_params_exp = net->group->make_input_vstream_params(
        true, HAILO_FORMAT_TYPE_FLOAT32, 1000, 1);
    auto out_params_exp = net->group->make_output_vstream_params(
        true, HAILO_FORMAT_TYPE_FLOAT32, 1000, 1);
    if (!in_params_exp.has_value() || !out_params_exp.has_value()) {
      if (err) *err = "HailoRT: cannot build the vstream params for " + hef_path;
      return false;
    }

    auto pipe_exp = hailort::InferVStreams::create(
        *net->group, in_params_exp.value(), out_params_exp.value());
    if (!pipe_exp.has_value()) {
      if (err) *err = "HailoRT: infer pipeline not activated for " + hef_path;
      return false;
    }
    /* InferVStreams is move-only. */
    net->pipe = std::make_unique<hailort::InferVStreams>(std::move(pipe_exp.value()));
    return true;
  }

  bool ok_ = false;
  std::unique_ptr<hailort::VDevice> vdevice_;
  HailoNet bb_;
  HailoNet enc_;
};

}  // namespace

std::unique_ptr<InferenceBackend> MakeHailoRTBackend(const std::string &hef_dir,
                                                     std::string *err) {
  return std::unique_ptr<InferenceBackend>(new HailoRTBackend(hef_dir, err));
}

}  // namespace sender

#endif /* SSCODEC_WITH_HAILORT */
