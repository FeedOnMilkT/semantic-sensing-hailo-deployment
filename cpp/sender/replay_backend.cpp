/*
 * replay_backend.cpp -- the "tensor replay" inference backend.
 *
 * This backend does NOT run any neural network: it replays tensors that were
 * dumped earlier, so the whole sender pipeline (host rounding + h_s + rANS +
 * container) can be exercised on a machine without a Hailo device.
 *
 * INPUT FORMAT
 * ------------
 * A directory containing one raw binary file per image, named <name>.bin,
 * each holding, in this order:
 *
 *     float32 y     [192*16*16]   NCHW
 *     float32 z_pre [128*8*8]     NCHW     (pre-rounding)
 *
 * Optionally the file may hold a THIRD block with the already-rounded z_hat
 * (that is what the Hailo emulator dumps).  In `zhat` mode the rounded
 * z_hat is used directly (skipping the host rounding) -- this is what the
 * bitstream-package builder does, because the emulator dump has already
 * applied the host rounding.
 *
 * The matching Python dumper lives in cpp/tools/dump_replay_tensors.py.
 */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "inference_backend.h"

namespace sender {

namespace {

bool read_file(const std::string &path, std::vector<uint8_t> *out) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n < 0) {
    fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(n));
  const size_t rd = fread(out->data(), 1, out->size(), f);
  fclose(f);
  return rd == out->size();
}

}  // namespace

class ReplayBackend : public InferenceBackend {
 public:
  ReplayBackend(const std::string &dir, bool use_zhat)
      : dir_(dir), use_zhat_(use_zhat) {}

  bool infer(const uint8_t *image_u8_nhwc, SenderTensors *out,
             std::string *err) override {
    if (!image_u8_nhwc) {
      if (err) *err = "replay backend: no image given";
      return false;
    }
    if (current_name_.empty()) {
      if (err) *err = "replay backend: no current name set";
      return false;
    }
    const std::string path = dir_ + "/" + current_name_ + ".bin";
    std::vector<uint8_t> raw;
    if (!read_file(path, &raw)) {
      if (err) *err = "replay backend: cannot read " + path;
      return false;
    }
    const size_t y_n = static_cast<size_t>(SenderShapes::kYC) * SenderShapes::kYh *
                       SenderShapes::kYw;
    const size_t z_n = static_cast<size_t>(SenderShapes::kZC) * SenderShapes::kZh *
                       SenderShapes::kZw;
    const size_t need = (y_n + z_n) * 4;
    if (raw.size() < need) {
      if (err) *err = "replay backend: file too small: " + path;
      return false;
    }
    out->y.resize(y_n);
    std::memcpy(out->y.data(), raw.data(), y_n * 4);

    if (use_zhat_ && raw.size() >= need + z_n * 4) {
      /* the third block is the already-rounded z_hat */
      out->z_pre.resize(z_n);
      std::memcpy(out->z_pre.data(), raw.data() + need, z_n * 4);
    } else {
      out->z_pre.resize(z_n);
      std::memcpy(out->z_pre.data(), raw.data() + y_n * 4, z_n * 4);
    }
    out->x_h.clear();
    out->x_l.clear();
    return true;
  }

  std::string name() const override {
    return std::string("replay:") + dir_ + (use_zhat_ ? " (z_hat)" : " (z_pre)");
  }

  void set_current_name(const std::string &n) { current_name_ = n; }

 private:
  std::string dir_;
  bool use_zhat_;
  std::string current_name_;
};

std::unique_ptr<InferenceBackend> MakeReplayBackend(const std::string &replay_dir,
                                                    std::string *err) {
  (void)err;
  return std::unique_ptr<InferenceBackend>(new ReplayBackend(replay_dir, false));
}

/* A variant that uses the already-rounded z_hat from the dump. */
std::unique_ptr<InferenceBackend> MakeReplayZhatBackend(
    const std::string &replay_dir, std::string *err) {
  (void)err;
  return std::unique_ptr<InferenceBackend>(new ReplayBackend(replay_dir, true));
}

}  // namespace sender
