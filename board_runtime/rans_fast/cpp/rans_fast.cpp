/* Fast rANS bindings: contiguous int32 buffers in, bytes out.
 * Algorithm matches CompressAI compressai/cpp_exts/rans (InterDigital BSD-3).
 * Goal: avoid Python list materialisation of symbols / CDF tables.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace py = pybind11;

constexpr int precision = 16;
constexpr uint16_t bypass_precision = 4;
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass;
};

inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
  }
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
  }
  *r = x;
  return val;
}

static inline const int32_t *cdf_row(const int32_t *cdfs, ssize_t stride,
                                     int32_t idx) {
  return cdfs + static_cast<ssize_t>(idx) * stride;
}

py::bytes encode_with_indexes_np(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> symbols,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indexes,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cdfs,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cdfs_sizes,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> offsets) {
  if (symbols.ndim() != 1 || indexes.ndim() != 1)
    throw std::runtime_error("symbols/indexes must be 1-D int32");
  if (symbols.size() != indexes.size())
    throw std::runtime_error("symbols/indexes size mismatch");
  if (cdfs.ndim() != 2)
    throw std::runtime_error("cdfs must be 2-D [n_cdfs, max_len] int32");
  if (cdfs_sizes.ndim() != 1 || offsets.ndim() != 1)
    throw std::runtime_error("cdfs_sizes/offsets must be 1-D int32");
  if (cdfs.shape(0) != cdfs_sizes.shape(0) ||
      cdfs.shape(0) != offsets.shape(0))
    throw std::runtime_error("cdfs / cdfs_sizes / offsets length mismatch");

  auto sym_b = symbols.unchecked<1>();
  auto idx_b = indexes.unchecked<1>();
  auto len_b = cdfs_sizes.unchecked<1>();
  auto off_b = offsets.unchecked<1>();

  const ssize_t n = symbols.size();
  const ssize_t stride = cdfs.shape(1);
  const int32_t *cdf_base = cdfs.data();

  std::vector<RansSymbol> syms;
  syms.reserve(static_cast<size_t>(n) * 2);

  // CompressAI encodes by walking symbols[0..n) pushing to a stack, then
  // flushes in reverse. Match that exactly (including bypass raw coding).
  for (ssize_t i = 0; i < n; ++i) {
    const int32_t cdf_idx = idx_b(i);
    if (cdf_idx < 0 || cdf_idx >= cdfs.shape(0))
      throw std::runtime_error("cdf index out of range");

    const int32_t *cdf = cdf_row(cdf_base, stride, cdf_idx);
    const int32_t max_value = len_b(cdf_idx) - 2;
    if (max_value < 0)
      throw std::runtime_error("invalid cdf length");

    int32_t value = sym_b(i) - off_b(cdf_idx);
    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = static_cast<uint32_t>(-2 * value - 1);
      value = max_value;
    } else if (value >= max_value) {
      raw_val = static_cast<uint32_t>(2 * (value - max_value));
      value = max_value;
    }

    syms.push_back({static_cast<uint16_t>(cdf[value]),
                    static_cast<uint16_t>(cdf[value + 1] - cdf[value]), false});

    if (value == max_value) {
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0)
        ++n_bypass;

      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t v =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        syms.push_back(
            {static_cast<uint16_t>(v), static_cast<uint16_t>(v + 1), true});
      }
    }
  }

  Rans64State rans;
  Rans64EncInit(&rans);
  std::vector<uint32_t> output(syms.size() + 2, 0);
  uint32_t *ptr = output.data() + output.size();

  while (!syms.empty()) {
    const RansSymbol sym = syms.back();
    syms.pop_back();
    if (!sym.bypass)
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    else
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
  }
  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      static_cast<int>(std::distance(ptr, output.data() + output.size())) *
      static_cast<int>(sizeof(uint32_t));
  return py::bytes(reinterpret_cast<char *>(ptr), nbytes);
}

py::array_t<int32_t> decode_with_indexes_np(
    const std::string &encoded,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> indexes,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cdfs,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cdfs_sizes,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> offsets) {
  if (indexes.ndim() != 1)
    throw std::runtime_error("indexes must be 1-D int32");
  if (cdfs.ndim() != 2)
    throw std::runtime_error("cdfs must be 2-D int32");
  if (cdfs_sizes.ndim() != 1 || offsets.ndim() != 1)
    throw std::runtime_error("cdfs_sizes/offsets must be 1-D int32");
  if (cdfs.shape(0) != cdfs_sizes.shape(0) ||
      cdfs.shape(0) != offsets.shape(0))
    throw std::runtime_error("cdfs / cdfs_sizes / offsets length mismatch");

  auto idx_b = indexes.unchecked<1>();
  auto len_b = cdfs_sizes.unchecked<1>();
  auto off_b = offsets.unchecked<1>();
  const ssize_t n = indexes.size();
  const ssize_t stride = cdfs.shape(1);
  const int32_t *cdf_base = cdfs.data();

  py::array_t<int32_t> out(n);
  auto out_b = out.mutable_unchecked<1>();

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  Rans64DecInit(&rans, &ptr);

  for (ssize_t i = 0; i < n; ++i) {
    const int32_t cdf_idx = idx_b(i);
    const int32_t *cdf = cdf_row(cdf_base, stride, cdf_idx);
    const int32_t cdf_size = len_b(cdf_idx);
    const int32_t max_value = cdf_size - 2;
    const int32_t offset = off_b(cdf_idx);

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);

    // Linear search identical to CompressAI (find_if first v > cum_freq).
    uint32_t s = 0;
    for (int32_t j = 0; j < cdf_size; ++j) {
      if (static_cast<uint32_t>(cdf[j]) > cum_freq) {
        s = static_cast<uint32_t>(j - 1);
        break;
      }
    }

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);
    if (value == max_value) {
      int32_t val = static_cast<int32_t>(
          Rans64DecGetBits(&rans, &ptr, bypass_precision));
      int32_t n_bypass = val;
      while (val == max_bypass_val) {
        val = static_cast<int32_t>(
            Rans64DecGetBits(&rans, &ptr, bypass_precision));
        n_bypass += val;
      }
      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = static_cast<int32_t>(
            Rans64DecGetBits(&rans, &ptr, bypass_precision));
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1)
        value = -value - 1;
      else
        value += max_value;
    }
    out_b(i) = value + offset;
  }
  return out;
}

PYBIND11_MODULE(_rans_fast, m) {
  m.doc() = "Contiguous-buffer rANS (CompressAI-compatible, no Python lists)";
  // Do NOT release the GIL while touching py::array / returning py::bytes.
  m.def("encode_with_indexes", &encode_with_indexes_np,
        "encode symbols with CDF indexes → bytes");
  m.def("decode_with_indexes", &decode_with_indexes_np,
        "decode bytes with CDF indexes → int32 ndarray");
}
