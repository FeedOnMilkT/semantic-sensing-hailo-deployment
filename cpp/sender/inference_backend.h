/*
 * inference_backend.h -- the sender's inference abstraction.
 *
 * The sender pipeline is:
 *
 *     image (512x512x3 u8, NHWC)
 *        -> [backend] -> x_h (128,64,64), x_l (256,32,32)
 *        -> [backend] -> y (192,16,16), z_pre (128,8,8)
 *        -> [host]    -> z_hat = floor(z_pre + 0.5) + median
 *        -> [codec]   -> means/scales (deterministic h_s), symbols, rANS
 *        -> [container] -> RTFF
 *
 * Two backends implement the interface:
 *   - HailoRTBackend  : runs the two HEFs on a Hailo device,
 *   - ReplayBackend   : replays tensors dumped earlier (host testing).
 *
 * All tensors are float32 NCHW (channels first), matching the Python
 * reference, EXCEPT the image which is uint8 NHWC (the on-chip
 * normalisation expects raw 0..255 values).
 */
#ifndef SSCODEC_SENDER_INFERENCE_BACKEND_H
#define SSCODEC_SENDER_INFERENCE_BACKEND_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sender {

/* Tensor shapes of the deployed sender chain. */
struct SenderShapes {
  static constexpr int kImageH = 512;
  static constexpr int kImageW = 512;
  static constexpr int kImageC = 3;

  static constexpr int kXhC = 128, kXhH = 64, kXhW = 64;
  static constexpr int kXlC = 256, kXlH = 32, kXlW = 32;
  static constexpr int kYC = 192, kYh = 16, kYw = 16;
  static constexpr int kZC = 128, kZh = 8, kZw = 8;
};

/* The output of one inference step. */
struct SenderTensors {
  /* y    : (192,16,16) float32 NCHW */
  std::vector<float> y;
  /* z_pre: (128,8,8) float32 NCHW (pre-rounding hyper latent) */
  std::vector<float> z_pre;
  /* x_h  : (128,64,64) float32 NCHW (kept for the manifest / diagnostics) */
  std::vector<float> x_h;
  /* x_l  : (256,32,32) float32 NCHW */
  std::vector<float> x_l;
};

/*
 * The inference backend interface.
 *
 * Implementations run the image -> (y, z_pre) chain.  The interface is
 * deliberately narrow so that a backend can be swapped without touching the
 * rest of the pipeline.
 */
class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;

  /* Run one image (uint8 NHWC, 512*512*3).  Returns false on error. */
  virtual bool infer(const uint8_t *image_u8_nhwc, SenderTensors *out,
                     std::string *err) = 0;

  /*
   * Tell the backend which image is about to be processed.  Backends that key
   * on a file name (the replay backend) use this; the HailoRT backend ignores
   * it.
   */
  virtual void set_current_name(const std::string &name) { (void)name; }

  /* Human-readable backend name (for the manifest). */
  virtual std::string name() const = 0;
};

/* Factories. */
std::unique_ptr<InferenceBackend> MakeHailoRTBackend(const std::string &hef_dir,
                                                     std::string *err);
std::unique_ptr<InferenceBackend> MakeReplayBackend(const std::string &replay_dir,
                                                    std::string *err);
/*
 * The replay variant that uses the ALREADY-ROUNDED z_hat from the dump
 * (skipping the host rounding).  This is what the bitstream-package builder
 * uses, because the Hailo emulator dump has already applied the rounding.
 */
std::unique_ptr<InferenceBackend> MakeReplayZhatBackend(
    const std::string &replay_dir, std::string *err);

}  // namespace sender

#endif /* SSCODEC_SENDER_INFERENCE_BACKEND_H */
