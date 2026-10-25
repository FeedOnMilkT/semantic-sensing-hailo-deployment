/*
 * codec.h -- internal codec implementation (shared by the C ABI and the
 *            C++ test suite).
 */
#ifndef SSCODEC_CODEC_H
#define SSCODEC_CODEC_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hyper_synthesis.h"
#include "rans.h"
#include "tables_format.h"

namespace sscodec {

/* RTFF container (see rtformer_fused.py: FusedLatentBottleneck.compress).
 *
 *   magic "RTFF" | u16 z_h | u16 z_w | u16 y_h | u16 y_w
 *               | u16 fh_h | u16 fh_w | u16 fl_h | u16 fl_w | u8 B
 *               | z_strings (B x [u32 len + bytes])
 *               | y_strings (B x [u32 len + bytes])
 * All integers BIG-ENDIAN. */
inline constexpr char kRtffMagic[4] = {'R', 'T', 'F', 'F'};
inline constexpr size_t kRtffHeaderSize = 4 + 8 * 2 + 1; /* = 21 */

struct RtffHeader {
  int z_h = 0, z_w = 0;
  int y_h = 0, y_w = 0;
  int fh_h = 0, fh_w = 0;
  int fl_h = 0, fl_w = 0;
  int batch = 0;
};

/* The codec state: tables + CDFs + h_s. */
class Codec {
 public:
  /* Load from codec_tables.bin.  Returns false and fills `err` on failure. */
  bool open(const std::string &tables_path, std::string *err);

  /* ---- tables accessors ---- */
  const std::vector<float> &scale_table() const { return scale_table_; }
  const std::vector<float> &medians() const { return medians_; }
  float scale_bound() const { return scale_bound_; }

  /* CDF index for one scale value (CompressAI's build_indexes rule). */
  int cdf_index(float scale) const;

  /* ---- hyper synthesis ---- */
  /* z_hat: (1, 128, z_h, z_w) -> means/scales (1, 192, y_h, y_w) + indexes */
  bool hyper_synthesis(const float *z_hat, int z_h, int z_w, float *means,
                       float *scales, int32_t *indexes, int y_h, int y_w,
                       std::string *err) const;

  /* ---- container ---- */
  /* Build the RTFF container from raw z/y streams. */
  static std::vector<uint8_t> build_container(const RtffHeader &hdr,
                                              const std::vector<uint8_t> &z_stream,
                                              const std::vector<uint8_t> &y_stream);

  /* Parse a RTFF container.  On success `z_stream`/`y_stream` receive the
   * raw rANS bytes and `hdr` the header fields. */
  static bool parse_container(const uint8_t *in, size_t n, RtffHeader *hdr,
                              std::vector<uint8_t> *z_stream,
                              std::vector<uint8_t> *y_stream, std::string *err);

  /* ---- streams ---- */
  /* Encode the factorized-prior (z) stream.  z is the PRE-rounding hyper
   * latent (1,128,z_h,z_w); medians are the per-channel medians. */
  std::vector<uint8_t> encode_z_stream(const float *z, int z_h, int z_w,
                                       const float *medians) const;

  /* Decode the factorized-prior stream -> z_hat (1,128,z_h,z_w). */
  bool decode_z_stream(const uint8_t *in, size_t n, int z_h, int z_w,
                       float *z_hat, std::string *err) const;

  /* Encode the conditional (y) stream from symbols+indexes. */
  std::vector<uint8_t> encode_y_stream(const int32_t *symbols,
                                       const int32_t *indexes, size_t count,
                                       std::string *err) const;

  /* Decode the conditional (y) stream. */
  bool decode_y_stream(const uint8_t *in, size_t n, const int32_t *indexes,
                       size_t count, int32_t *symbols,
                       std::string *err) const;

  /* Quantise y into symbols: round(y - means).  Matches CompressAI's
   * quantize(inputs, "symbols", means) -- torch.round is round-half-to-EVEN. */
  static int32_t round_half_even(float v);

  /* Full encode. */
  bool encode(const float *y, const float *z_hat, int y_h, int y_w, int z_h,
              int z_w, int fh_h, int fh_w, int fl_h, int fl_w,
              const int32_t *forced_indexes, const float *forced_means,
              std::vector<uint8_t> *out, std::string *err) const;

  /* Full decode. */
  bool decode(const uint8_t *in, size_t n, float *y_hat, size_t cap_floats,
              int *dims, std::string *err) const;

  const HyperSynthesis &hyper() const { return hyper_; }

 private:
  HyperSynthesis hyper_;
  std::vector<float> scale_table_;
  std::vector<float> medians_;
  float scale_bound_ = 0.11f;

  /* GaussianConditional CDFs */
  std::vector<int32_t> gauss_cdf_;
  std::vector<int32_t> gauss_cdf_length_;
  std::vector<int32_t> gauss_offset_;

  /* EntropyBottleneck CDFs */
  std::vector<int32_t> eb_cdf_;
  std::vector<int32_t> eb_cdf_length_;
  std::vector<int32_t> eb_offset_;

  int eb_channels_ = 0;
  int gauss_levels_ = 0;
};

}  // namespace sscodec

#endif /* SSCODEC_CODEC_H */
