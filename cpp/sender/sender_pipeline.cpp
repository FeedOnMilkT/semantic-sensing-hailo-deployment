/*
 * sender_pipeline.cpp -- the sender's host-side pipeline.
 */
#include "sender_pipeline.h"

#include <cstdio>
#include <cstring>

#include "sscodec.h"

namespace sender {

bool SenderPipeline::init(const std::string &tables_path, std::string *err) {
  close();
  SsCodec *c = ss_open(tables_path.c_str());
  if (!c) {
    if (err) *err = "ss_open failed for " + tables_path;
    return false;
  }
  codec_ = c;
  medians_.assign(SenderShapes::kZC, 0.0f);
  if (ss_get_medians(c, medians_.data(), medians_.size()) != 0) {
    if (err) *err = "ss_get_medians failed";
    close();
    return false;
  }
  return true;
}

void SenderPipeline::close() {
  if (codec_) {
    ss_close(static_cast<SsCodec *>(codec_));
    codec_ = nullptr;
  }
}

bool SenderPipeline::encode(const SenderTensors &t, const std::string &name,
                             EncodedImage *out, std::string *err) {
  if (!codec_) {
    if (err) *err = "pipeline not initialised";
    return false;
  }
  const size_t z_n = static_cast<size_t>(SenderShapes::kZC) * SenderShapes::kZh *
                     SenderShapes::kZw;
  const size_t y_n = static_cast<size_t>(SenderShapes::kYC) * SenderShapes::kYh *
                     SenderShapes::kYw;
  if (t.z_pre.size() != z_n || t.y.size() != y_n) {
    if (err) *err = "unexpected tensor sizes";
    return false;
  }

  /*
   * THE host-side scalar op:
   *     z_hat = floor(z_pre + 0.5) + median
   * (round-half-up, identical to deploy/hailo_split_pipeline.py:host_round_z).
   * It MUST stay on the host: the Hailo DFC compiler drops Floor/Cast(int32)
   * at PARSE time (they are no-ops on chip), and an in-graph rounding emits
   * CONTINUOUS symbols, which inflates the rANS z-stream ~2.3x.
   *
   * If the upstream already applied it (the Hailo emulator dump stores the
   * rounded z_hat), we must NOT apply it again.
   */
  std::vector<float> z_hat(z_n);
  if (z_hat_rounded_) {
    z_hat = t.z_pre;
  } else if (ss_round_z(t.z_pre.data(), SenderShapes::kZh, SenderShapes::kZw,
                        medians_.data(), z_hat.data()) != 0) {
    if (err) *err = "ss_round_z failed";
    return false;
  }

  out->name = name;
  out->bytes.resize(1 << 20);
  size_t out_len = 0;
  const int rc = ss_encode(static_cast<SsCodec *>(codec_), t.y.data(),
                           z_hat.data(), SenderShapes::kYh, SenderShapes::kYw,
                           SenderShapes::kZh, SenderShapes::kZw,
                           SenderShapes::kXhH, SenderShapes::kXhW,
                           SenderShapes::kXlH, SenderShapes::kXlW,
                           out->bytes.data(), out->bytes.size(), &out_len);
  if (rc != 0) {
    if (err) *err = "ss_encode failed with code " + std::to_string(rc);
    return false;
  }
  out->bytes.resize(out_len);
  return true;
}

}  // namespace sender
