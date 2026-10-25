/*
 * rans.cpp -- pure C++ port of CompressAI 1.2.8's rANS coder.
 *
 * See rans.h for the copyright / provenance notes.
 * The logic is a line-by-line de-pybind of
 *   CompressAI-1.2.8/compressai/cpp_exts/rans/rans_interface.cpp
 * with the pybind11 types replaced by plain C++ containers.
 */
#include "rans.h"

namespace sscodec {
namespace {

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}

/* Offset of row `i` inside the flat CDF array. */
inline size_t cdf_row_offset(const std::vector<int32_t> &cdf_lengths, int32_t i) {
  size_t off = 0;
  for (int32_t k = 0; k < i; ++k) off += static_cast<size_t>(cdf_lengths[k]);
  return off;
}

/*
 * Stride (in elements) between two consecutive CDF rows.
 *
 * CompressAI passes `self._quantized_cdf.tolist()` -- the FULL 2-D tensor,
 * whose rows all have the same width (max_length + 2) -- together with the
 * per-row `_cdf_length` telling how many entries of each row are meaningful.
 * We reproduce that layout exactly: the flat array is row-major with a
 * uniform stride, and `cdf_lengths[i]` is the meaningful length of row i.
 */
inline size_t cdf_stride(const std::vector<int32_t> &cdfs,
                         const std::vector<int32_t> &cdf_lengths) {
  if (cdf_lengths.empty()) return 0;
  return cdfs.size() / cdf_lengths.size();
}

}  // namespace

void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<int32_t> &cdfs, const std::vector<int32_t> &cdf_lengths,
    const std::vector<int32_t> &offsets) {
  if (symbols.size() != indexes.size()) {
    throw std::invalid_argument("rans: symbols/indexes size mismatch");
  }

  /* Pre-compute the row offsets so the hot loop is O(1) per symbol. */
  const size_t stride = cdf_stride(cdfs, cdf_lengths);
  const size_t n_rows = cdf_lengths.size();
  for (size_t i = 0; i < symbols.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    if (cdf_idx < 0 || static_cast<size_t>(cdf_idx) >= n_rows) {
      throw std::out_of_range("rans: cdf index out of range");
    }
    const size_t base = static_cast<size_t>(cdf_idx) * stride;
    const int32_t len = cdf_lengths[static_cast<size_t>(cdf_idx)];

    const int32_t max_value = len - 2;
    int32_t value = symbols[i] - offsets[static_cast<size_t>(cdf_idx)];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = static_cast<uint32_t>(-2 * value - 1);
      value = max_value;
    } else if (value >= max_value) {
      raw_val = static_cast<uint32_t>(2 * (value - max_value));
      value = max_value;
    }

    if (value < 0 || value >= len - 1) {
      throw std::out_of_range("rans: symbol out of cdf range");
    }

    _syms.push_back({static_cast<uint16_t>(cdfs[base + static_cast<size_t>(value)]),
                     static_cast<uint16_t>(cdfs[base + static_cast<size_t>(value) + 1] -
                                           cdfs[base + static_cast<size_t>(value)]),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * kBypassPrecision)) != 0) {
        ++n_bypass;
      }

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= kMaxBypassVal) {
        _syms.push_back({kMaxBypassVal, static_cast<uint16_t>(kMaxBypassVal + 1), true});
        val -= kMaxBypassVal;
      }
      _syms.push_back({static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t bval =
            static_cast<int32_t>((raw_val >> (j * kBypassPrecision)) & kMaxBypassVal);
        _syms.push_back({static_cast<uint16_t>(bval), static_cast<uint16_t>(bval + 1), true});
      }
    }
  }
}

void BufferedRansEncoder::flush(std::vector<uint8_t> &out) {
  Rans64State rans;
  Rans64EncInit(&rans);

  /*
   * The reference writes into a uint32 buffer and then trims it to the used
   * bytes.  We do the same (a uint32 buffer is required by the rANS
   * renormalisation), then append the trimmed bytes.
   */
  const size_t n_syms = _syms.size();
  std::vector<uint32_t> buffer(n_syms + 8, 0xCC);
  uint32_t *ptr = buffer.data() + buffer.size();
  if (ptr == nullptr) {
    throw std::runtime_error("rans: null output pointer");
  }

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, kPrecision);
    } else {
      /* unlikely... */
      Rans64EncPutBits(&rans, &ptr, sym.start, kBypassPrecision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const size_t nbytes =
      static_cast<size_t>(std::distance(ptr, buffer.data() + buffer.size())) *
      sizeof(uint32_t);

  const uint8_t *bytes = reinterpret_cast<const uint8_t *>(ptr);
  out.insert(out.end(), bytes, bytes + nbytes);
}

std::vector<uint8_t> BufferedRansEncoder::encode_with_indexes_and_flush(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<int32_t> &cdfs, const std::vector<int32_t> &cdf_lengths,
    const std::vector<int32_t> &offsets) {
  encode_with_indexes(symbols, indexes, cdfs, cdf_lengths, offsets);
  std::vector<uint8_t> out;
  flush(out);
  return out;
}

std::vector<int32_t> RansDecoder::decode_with_indexes(
    const uint8_t *encoded, size_t nbytes, const std::vector<int32_t> &indexes,
    const std::vector<int32_t> &cdfs, const std::vector<int32_t> &cdf_lengths,
    const std::vector<int32_t> &offsets) {
  std::vector<int32_t> output(indexes.size());

  /*
   * The rANS decoder reads 32-bit words and may perform ONE extra
   * renormalisation after the LAST symbol.  That read only affects the
   * decoder state, never a decoded symbol, so we decode from a zero-padded
   * copy of the stream.  This keeps the decoded symbols identical to
   * CompressAI's while removing the out-of-bounds read that the reference
   * implementation performs on the raw buffer.
   */
  const size_t padded = (nbytes + 3) & ~static_cast<size_t>(3);
  /* +16 guard words: enough for the final renormalisation and for the
   * bypass-bit path. */
  std::vector<uint32_t> words(padded / 4 + 16, 0);
  if (nbytes > 0) {
    std::memcpy(words.data(), encoded, nbytes);
  }

  Rans64State rans;
  uint32_t *ptr = words.data();
  Rans64DecInit(&rans, &ptr);

  std::vector<size_t> row_off(cdf_lengths.size(), 0);
  const size_t stride = cdf_stride(cdfs, cdf_lengths);
  for (size_t i = 0; i < cdf_lengths.size(); ++i) {
    row_off[i] = i * stride;
  }

  for (size_t i = 0; i < indexes.size(); ++i) {
    const int32_t cdf_idx = indexes[i];
    if (cdf_idx < 0 || static_cast<size_t>(cdf_idx) >= cdf_lengths.size()) {
      throw std::out_of_range("rans: cdf index out of range");
    }
    const size_t base = row_off[static_cast<size_t>(cdf_idx)];
    const int32_t len = cdf_lengths[static_cast<size_t>(cdf_idx)];

    const int32_t max_value = len - 2;
    const int32_t offset = offsets[static_cast<size_t>(cdf_idx)];

    const uint32_t cum_freq = Rans64DecGet(&rans, kPrecision);

    /* find the first cdf entry strictly greater than cum_freq */
    const int32_t *cdf_begin = cdfs.data() + base;
    const int32_t *cdf_end = cdf_begin + len;
    const int32_t *it = std::find_if(cdf_begin, cdf_end,
                                     [cum_freq](int32_t v) {
                                       return static_cast<uint32_t>(v) > cum_freq;
                                     });
    if (it == cdf_end) {
      throw std::runtime_error("rans: cum_freq outside cdf");
    }
    const uint32_t s = static_cast<uint32_t>(std::distance(cdf_begin, it) - 1);

    Rans64DecAdvance(&rans, &ptr, cdf_begin[s], cdf_begin[s + 1] - cdf_begin[s],
                     kPrecision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = static_cast<int32_t>(Rans64DecGetBits(&rans, &ptr, kBypassPrecision));
      int32_t n_bypass = val;

      while (val == kMaxBypassVal) {
        val = static_cast<int32_t>(Rans64DecGetBits(&rans, &ptr, kBypassPrecision));
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int32_t j = 0; j < n_bypass; ++j) {
        val = static_cast<int32_t>(Rans64DecGetBits(&rans, &ptr, kBypassPrecision));
        raw_val |= val << (j * kBypassPrecision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}

}  // namespace sscodec
