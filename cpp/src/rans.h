/*
 * rans.h -- pure C++ port of CompressAI 1.2.8's rANS coder.
 *
 * SOURCE / COPYRIGHT
 * ------------------
 * The algorithm and the byte layout are taken from CompressAI v1.2.8
 *   compressai/cpp_exts/rans/rans_interface.{hpp,cpp}
 *   (c) 2021-2024, InterDigital Communications, Inc -- BSD-3-Clause-Clear
 * and from Fabian Giesen's public-domain ryg_rans (CC0):
 *   third_party/ryg_rans/rans64.h
 *
 * The reference implementation is a pybind11 extension; this file is a
 * de-pybinded rewrite that keeps the EXACT same semantics so that the byte
 * streams are bit-compatible with `compressai.ans` (RansEncoder/RansDecoder).
 * That bit-compatibility is the whole point: the Python CompressAI model is
 * used as an independent cross-check of the C++ codec.
 *
 * Differences from the reference (behaviour-preserving):
 *   - no pybind11 / std::string / std::vector plumbing in the hot path,
 *   - the CDFs are passed as a flat array + lengths (the reference's
 *     vector<vector<int32_t>> is row-major identical),
 *   - the encoder writes into a caller-provided buffer.
 *
 * See LICENSE (BSD-3-Clause-Clear) and LICENSE.ryg_rans (CC0) next to this
 * file for the full terms.
 */
#ifndef SSCODEC_RANS_H
#define SSCODEC_RANS_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace sscodec {

/* probability range; CompressAI uses 16 bits */
inline constexpr int kPrecision = 16;
/* number of bits written per bypass chunk */
inline constexpr uint16_t kBypassPrecision = 4;
/* largest value representable in one bypass chunk */
inline constexpr uint16_t kMaxBypassVal = (1 << kBypassPrecision) - 1;

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass; /* bypass flag: raw bits written to the stream */
};

/*
 * Buffered rANS encoder.
 *
 * Symbols are collected first (the reference does the same) and then flushed
 * in reverse order.  The flush is byte-for-byte identical to CompressAI's
 * BufferedRansEncoder::flush().
 */
class BufferedRansEncoder {
 public:
  BufferedRansEncoder() = default;

  BufferedRansEncoder(const BufferedRansEncoder &) = delete;
  BufferedRansEncoder &operator=(const BufferedRansEncoder &) = delete;

  /*
   * symbols / indexes: parallel arrays of length n.
   * cdfs: flat, row-major CDF table.  Row i has cdf_lengths[i] entries.
   * offsets: per-row symbol offset.
   */
  void encode_with_indexes(const std::vector<int32_t> &symbols,
                            const std::vector<int32_t> &indexes,
                            const std::vector<int32_t> &cdfs,
                            const std::vector<int32_t> &cdf_lengths,
                            const std::vector<int32_t> &offsets);

  /* Serialize the buffered symbols.  Appends to `out`. */
  void flush(std::vector<uint8_t> &out);

  /* Convenience: encode + flush, returns the byte stream. */
  std::vector<uint8_t> encode_with_indexes_and_flush(
      const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
      const std::vector<int32_t> &cdfs, const std::vector<int32_t> &cdf_lengths,
      const std::vector<int32_t> &offsets);

 private:
  std::vector<RansSymbol> _syms;
};

/*
 * rANS decoder.  Decodes a byte stream into symbols given the same
 * (cdfs, cdf_lengths, offsets) and the per-symbol indexes.
 */
class RansDecoder {
 public:
  RansDecoder() = default;

  RansDecoder(const RansDecoder &) = delete;
  RansDecoder &operator=(const RansDecoder &) = delete;

  std::vector<int32_t> decode_with_indexes(
      const uint8_t *encoded, size_t nbytes,
      const std::vector<int32_t> &indexes, const std::vector<int32_t> &cdfs,
      const std::vector<int32_t> &cdf_lengths, const std::vector<int32_t> &offsets);
};

}  // namespace sscodec

#endif /* SSCODEC_RANS_H */
