/*
 * sender_pipeline.h -- the sender's host-side pipeline.
 *
 * This is the part of the sender that runs on the HOST (the board), i.e.
 * everything that is not the two Hailo HEFs:
 *
 *   1. the host rounding   z_hat = floor(z_pre + 0.5) + median
 *   2. the deterministic h_s + CDF index  (libsscodec)
 *   3. the symbol quantisation             round(y - means)
 *   4. the rANS encoding + RTFF container  (libsscodec)
 *
 * Steps 2-4 are shared with the receiver through libsscodec, so the receiver
 * decodes exactly what the sender encoded.
 */
#ifndef SSCODEC_SENDER_PIPELINE_H
#define SSCODEC_SENDER_PIPELINE_H

#include <cstdint>
#include <string>
#include <vector>

#include "inference_backend.h"

namespace sender {

/* One encoded image + its manifest entry. */
struct EncodedImage {
  std::string name;
  std::vector<uint8_t> bytes;
};

/* The host-side pipeline: owns a codec handle + the per-channel medians. */
class SenderPipeline {
 public:
  /* tables_path: codec_tables.bin
   * The medians come from the tables (the EntropyBottleneck medians), which
   * are the SAME numbers the Python host_round_z() uses. */
  bool init(const std::string &tables_path, std::string *err);
  void close();

  /*
   * Whether `SenderTensors::z_pre` already holds the ROUNDED z_hat
   * (i.e. floor(z_pre + 0.5) + median was already applied upstream).
   *
   * This is the case for the replay backend fed from the Hailo emulator dump
   * (the dump stores the rounded z_hat), and it must NOT round again -- doing
   * so would add the median twice and corrupt the z stream.
   */
  void set_z_hat_already_rounded(bool v) { z_hat_rounded_ = v; }

  /* Encode one image. */
  bool encode(const SenderTensors &t, const std::string &name,
              EncodedImage *out, std::string *err);

  /* The medians used by the host rounding (for diagnostics). */
  const std::vector<float> &medians() const { return medians_; }

 private:
  void *codec_ = nullptr; /* SsCodec* */
  std::vector<float> medians_;
  bool z_hat_rounded_ = false;
};

/* SHA-256 of a byte buffer, lower-case hex (64 chars + NUL). */
std::string Sha256Hex(const uint8_t *data, size_t len);

}  // namespace sender

#endif /* SSCODEC_SENDER_PIPELINE_H */
