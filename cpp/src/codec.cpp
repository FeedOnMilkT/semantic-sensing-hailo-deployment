/*
 * codec.cpp -- the shared codec: tables loading, deterministic h_s, the rANS
 *               streams and the RTFF container.
 */
#include "codec.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace sscodec {

namespace {

/* --- little-endian readers -------------------------------------------- */
uint32_t rd_u32(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t rd_u64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | static_cast<uint64_t>(p[i]);
  return v;
}

float rd_f32(const uint8_t *p) {
  uint32_t u = rd_u32(p);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int32_t rd_i32(const uint8_t *p) { return static_cast<int32_t>(rd_u32(p)); }

/* --- big-endian writers (RTFF) ---------------------------------------- */
void wr_u16be(std::vector<uint8_t> &v, uint16_t x) {
  v.push_back(static_cast<uint8_t>(x >> 8));
  v.push_back(static_cast<uint8_t>(x & 0xFF));
}

void wr_u32be(std::vector<uint8_t> &v, uint32_t x) {
  v.push_back(static_cast<uint8_t>(x >> 24));
  v.push_back(static_cast<uint8_t>((x >> 16) & 0xFF));
  v.push_back(static_cast<uint8_t>((x >> 8) & 0xFF));
  v.push_back(static_cast<uint8_t>(x & 0xFF));
}

uint16_t rd_u16be(const uint8_t *p) {
  return static_cast<uint16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

uint32_t rd_u32be(const uint8_t *p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
         (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

}  // namespace

/* ====================================================================== */
/*  tables loading                                                        */
/* ====================================================================== */

bool Codec::open(const std::string &tables_path, std::string *err) {
  std::ifstream f(tables_path, std::ios::binary);
  if (!f) {
    if (err) *err = "cannot open tables file: " + tables_path;
    return false;
  }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
  if (buf.size() < kTablesHeaderSize) {
    if (err) *err = "tables file too short";
    return false;
  }
  if (std::memcmp(buf.data(), kTablesMagic, 4) != 0) {
    if (err) *err = "bad tables magic";
    return false;
  }
  const uint32_t version = rd_u32(buf.data() + 4);
  if (version != kTablesFormatVersion) {
    if (err) *err = "unsupported tables version " + std::to_string(version);
    return false;
  }
  const uint32_t n_layers = rd_u32(buf.data() + 12);
  const uint32_t n_tables = rd_u32(buf.data() + 16);

  size_t off = kTablesHeaderSize;

  /* ---- h_s layers ---- */
  std::vector<HsLayer> layers;
  layers.reserve(n_layers);
  for (uint32_t i = 0; i < n_layers; ++i) {
    if (off + kHsLayerDescriptorSize > buf.size()) {
      if (err) *err = "tables truncated in h_s layer table";
      return false;
    }
    HsLayer L;
    const uint8_t *p = buf.data() + off;
    L.kind = rd_u32(p + 0);
    L.flags = rd_u32(p + 4);
    L.in_channels = rd_u32(p + 8);
    L.out_channels = rd_u32(p + 12);
    L.kernel_h = rd_u32(p + 16);
    L.kernel_w = rd_u32(p + 20);
    L.stride_h = rd_u32(p + 24);
    L.stride_w = rd_u32(p + 28);
    L.pad_h = rd_u32(p + 32);
    L.pad_w = rd_u32(p + 36);
    const uint64_t param_count = rd_u64(p + 44);
    L.scalar0 = rd_f32(p + 52);
    off += kHsLayerDescriptorSize;
    if (off + param_count * 4 > buf.size()) {
      if (err) *err = "tables truncated in h_s params";
      return false;
    }
    L.weight.resize(static_cast<size_t>(param_count));
    for (uint64_t k = 0; k < param_count; ++k) {
      L.weight[static_cast<size_t>(k)] = rd_f32(buf.data() + off + k * 4);
    }
    off += static_cast<size_t>(param_count) * 4;
    layers.push_back(std::move(L));
  }
  hyper_.set_layers(std::move(layers));

  /* ---- tables ---- */
  for (uint32_t i = 0; i < n_tables; ++i) {
    if (off + 24 > buf.size()) {
      if (err) *err = "tables truncated in table descriptors";
      return false;
    }
    const uint8_t *p = buf.data() + off;
    const uint32_t tag = rd_u32(p + 0);
    const uint32_t dtype = rd_u32(p + 4);
    const uint64_t count = rd_u64(p + 16);
    off += 24;
    if (off + count * 4 > buf.size()) {
      if (err) *err = "tables truncated in table data";
      return false;
    }
    std::vector<float> fvals;
    std::vector<int32_t> ivals;
    if (dtype == 0) {
      fvals.resize(static_cast<size_t>(count));
      for (uint64_t k = 0; k < count; ++k) {
        fvals[static_cast<size_t>(k)] = rd_f32(buf.data() + off + k * 4);
      }
    } else {
      ivals.resize(static_cast<size_t>(count));
      for (uint64_t k = 0; k < count; ++k) {
        ivals[static_cast<size_t>(k)] = rd_i32(buf.data() + off + k * 4);
      }
    }
    off += static_cast<size_t>(count) * 4;

    switch (tag) {
      case kTagScaleTable: scale_table_ = std::move(fvals); break;
      case kTagGaussCdf: gauss_cdf_ = std::move(ivals); break;
      case kTagGaussCdfLength: gauss_cdf_length_ = std::move(ivals); break;
      case kTagGaussOffset: gauss_offset_ = std::move(ivals); break;
      case kTagEbCdf: eb_cdf_ = std::move(ivals); break;
      case kTagEbCdfLength: eb_cdf_length_ = std::move(ivals); break;
      case kTagEbOffset: eb_offset_ = std::move(ivals); break;
      case kTagMedians: medians_ = std::move(fvals); break;
      case kTagQuantiles: break; /* informational only */
      case kTagScaleBound:
        if (!fvals.empty()) scale_bound_ = fvals[0];
        break;
      default:
        if (err) *err = "unknown table tag " + std::to_string(tag);
        return false;
    }
  }

  /* ---- sanity checks ---- */
  if (scale_table_.empty()) {
    if (err) *err = "tables: missing scale table";
    return false;
  }
  gauss_levels_ = static_cast<int>(scale_table_.size());
  if (gauss_cdf_length_.size() != static_cast<size_t>(gauss_levels_) ||
      gauss_offset_.size() != static_cast<size_t>(gauss_levels_)) {
    if (err) *err = "tables: gauss cdf length/offset mismatch";
    return false;
  }
  {
    /*
     * CompressAI stores the CDF as a 2-D tensor whose rows all have the same
     * width (max_length + 2); `_cdf_length[i]` gives the meaningful length of
     * row i.  The flat array we read is therefore row-major with a uniform
     * stride, and its total size is rows * stride.
     */
    const size_t rows = gauss_cdf_length_.size();
    if (rows == 0 || gauss_cdf_.size() % rows != 0) {
      if (err) *err = "tables: gauss cdf size mismatch";
      return false;
    }
    for (int32_t l : gauss_cdf_length_) {
      if (l < 2 || static_cast<size_t>(l) > gauss_cdf_.size() / rows) {
        if (err) *err = "tables: gauss cdf row length out of range";
        return false;
      }
    }
  }
  eb_channels_ = static_cast<int>(eb_cdf_length_.size());
  if (eb_channels_ == 0 || eb_offset_.size() != eb_cdf_length_.size()) {
    if (err) *err = "tables: entropy bottleneck cdf mismatch";
    return false;
  }
  {
    const size_t rows = eb_cdf_length_.size();
    if (rows == 0 || eb_cdf_.size() % rows != 0) {
      if (err) *err = "tables: eb cdf size mismatch";
      return false;
    }
    for (int32_t l : eb_cdf_length_) {
      if (l < 2 || static_cast<size_t>(l) > eb_cdf_.size() / rows) {
        if (err) *err = "tables: eb cdf row length out of range";
        return false;
      }
    }
  }
  if (medians_.size() != static_cast<size_t>(eb_channels_)) {
    if (err) *err = "tables: medians size mismatch";
    return false;
  }
  if (hyper_.out_channels() == 0) {
    if (err) *err = "tables: no h_s layers";
    return false;
  }
  return true;
}

/* ====================================================================== */
/*  CDF index                                                             */
/* ====================================================================== */

int Codec::cdf_index(float scale) const {
  /*
   * CompressAI's build_indexes:
   *     scales = lower_bound_scale(scales)          # max(scale, scale_bound)
   *     indexes = full(size, len(scale_table) - 1)
   *     for s in scale_table[:-1]:
   *         indexes -= (scales <= s)
   *
   * i.e. the index counts the scale-table entries STRICTLY BELOW the scale,
   * capped at len-1.
   */
  float s = scale;
  if (!(s > scale_bound_)) s = scale_bound_; /* NaN-safe lower bound */
  int idx = gauss_levels_ - 1;
  for (int i = 0; i + 1 < gauss_levels_; ++i) {
    if (s <= scale_table_[static_cast<size_t>(i)]) --idx;
  }
  return idx;
}

/* ====================================================================== */
/*  hyper synthesis                                                       */
/* ====================================================================== */

bool Codec::hyper_synthesis(const float *z_hat, int z_h, int z_w, float *means,
                             float *scales, int32_t *indexes, int y_h, int y_w,
                             std::string *err) const {
  const int c_out = hyper_.out_channels();
  if (c_out % 2 != 0) {
    if (err) *err = "h_s output channels must be even";
    return false;
  }
  const int c_half = c_out / 2;
  const size_t n = static_cast<size_t>(c_half) * y_h * y_w;

  std::vector<float> hs(static_cast<size_t>(c_out) * y_h * y_w);
  hyper_.forward(z_hat, z_h, z_w, hs.data(), y_h, y_w);

  for (size_t i = 0; i < n; ++i) {
    const float m = hs[i];
    const float s = hs[static_cast<size_t>(c_half) * y_h * y_w + i];
    means[i] = m;
    scales[i] = s;
    if (indexes) indexes[i] = cdf_index(s);
  }
  return true;
}

/* ====================================================================== */
/*  container                                                             */
/* ====================================================================== */

std::vector<uint8_t> Codec::build_container(
    const RtffHeader &hdr, const std::vector<uint8_t> &z_stream,
    const std::vector<uint8_t> &y_stream) {
  std::vector<uint8_t> out;
  out.reserve(kRtffHeaderSize + 8 + z_stream.size() + y_stream.size());
  out.insert(out.end(), kRtffMagic, kRtffMagic + 4);
  wr_u16be(out, static_cast<uint16_t>(hdr.z_h));
  wr_u16be(out, static_cast<uint16_t>(hdr.z_w));
  wr_u16be(out, static_cast<uint16_t>(hdr.y_h));
  wr_u16be(out, static_cast<uint16_t>(hdr.y_w));
  wr_u16be(out, static_cast<uint16_t>(hdr.fh_h));
  wr_u16be(out, static_cast<uint16_t>(hdr.fh_w));
  wr_u16be(out, static_cast<uint16_t>(hdr.fl_h));
  wr_u16be(out, static_cast<uint16_t>(hdr.fl_w));
  out.push_back(static_cast<uint8_t>(hdr.batch));
  wr_u32be(out, static_cast<uint32_t>(z_stream.size()));
  out.insert(out.end(), z_stream.begin(), z_stream.end());
  wr_u32be(out, static_cast<uint32_t>(y_stream.size()));
  out.insert(out.end(), y_stream.begin(), y_stream.end());
  return out;
}

bool Codec::parse_container(const uint8_t *in, size_t n, RtffHeader *hdr,
                            std::vector<uint8_t> *z_stream,
                            std::vector<uint8_t> *y_stream, std::string *err) {
  if (n < kRtffHeaderSize) {
    if (err) *err = "container too short";
    return false;
  }
  if (std::memcmp(in, kRtffMagic, 4) != 0) {
    if (err) *err = "bad container magic";
    return false;
  }
  hdr->z_h = rd_u16be(in + 4);
  hdr->z_w = rd_u16be(in + 6);
  hdr->y_h = rd_u16be(in + 8);
  hdr->y_w = rd_u16be(in + 10);
  hdr->fh_h = rd_u16be(in + 12);
  hdr->fh_w = rd_u16be(in + 14);
  hdr->fl_h = rd_u16be(in + 16);
  hdr->fl_w = rd_u16be(in + 18);
  hdr->batch = in[20];

  size_t off = kRtffHeaderSize;
  auto read_one = [&](std::vector<uint8_t> *dst) -> bool {
    if (off + 4 > n) {
      if (err) *err = "container truncated (length)";
      return false;
    }
    const uint32_t len = rd_u32be(in + off);
    off += 4;
    if (off + len > n) {
      if (err) *err = "container truncated (payload)";
      return false;
    }
    dst->assign(in + off, in + off + len);
    off += len;
    return true;
  };
  if (!read_one(z_stream)) return false;
  if (!read_one(y_stream)) return false;
  return true;
}

/* ====================================================================== */
/*  streams                                                               */
/* ====================================================================== */

int32_t Codec::round_half_even(float v) {
  /*
   * torch.round() is round-half-to-EVEN.  We implement it explicitly so the
   * result does not depend on the FPU rounding mode.
   */
  if (!std::isfinite(v)) return 0;
  const float fl = std::floor(v);
  const float diff = v - fl;
  int64_t i = static_cast<int64_t>(fl);
  if (diff > 0.5f) {
    i += 1;
  } else if (diff == 0.5f) {
    if (i & 1) i += 1; /* round half to even */
  }
  return static_cast<int32_t>(i);
}

std::vector<uint8_t> Codec::encode_z_stream(const float *z, int z_h, int z_w,
                                            const float *medians) const {
  const size_t hw = static_cast<size_t>(z_h) * z_w;
  const size_t n = static_cast<size_t>(eb_channels_) * hw;
  std::vector<int32_t> symbols(n);
  std::vector<int32_t> indexes(n);

  for (int ch = 0; ch < eb_channels_; ++ch) {
    const float med = medians ? medians[ch] : 0.0f;
    for (size_t p = 0; p < hw; ++p) {
      const size_t idx = static_cast<size_t>(ch) * hw + p;
      symbols[idx] = round_half_even(z[idx] - med);
      /* EntropyBottleneck uses one CDF row per channel. */
      indexes[idx] = ch;
    }
  }
  BufferedRansEncoder enc;
  return enc.encode_with_indexes_and_flush(symbols, indexes, eb_cdf_,
                                            eb_cdf_length_, eb_offset_);
}

bool Codec::decode_z_stream(const uint8_t *in, size_t n, int z_h, int z_w,
                            float *z_hat, std::string *err) const {
  const size_t hw = static_cast<size_t>(z_h) * z_w;
  const size_t count = static_cast<size_t>(eb_channels_) * hw;
  std::vector<int32_t> indexes(count);
  for (int ch = 0; ch < eb_channels_; ++ch) {
    std::fill(indexes.begin() + static_cast<ptrdiff_t>(ch) * hw,
              indexes.begin() + static_cast<ptrdiff_t>(ch + 1) * hw, ch);
  }
  RansDecoder dec;
  std::vector<int32_t> syms;
  try {
    syms = dec.decode_with_indexes(in, n, indexes, eb_cdf_, eb_cdf_length_,
                                   eb_offset_);
  } catch (const std::exception &e) {
    if (err) *err = std::string("rans z decode failed: ") + e.what();
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    z_hat[i] =
        static_cast<float>(syms[i]) + medians_[i / hw];
  }
  return true;
}

std::vector<uint8_t> Codec::encode_y_stream(const int32_t *symbols,
                                            const int32_t *indexes, size_t count,
                                            std::string *err) const {
  BufferedRansEncoder enc;
  try {
    return enc.encode_with_indexes_and_flush(
        std::vector<int32_t>(symbols, symbols + count),
        std::vector<int32_t>(indexes, indexes + count), gauss_cdf_,
        gauss_cdf_length_, gauss_offset_);
  } catch (const std::exception &e) {
    if (err) *err = std::string("rans y encode failed: ") + e.what();
    return {};
  }
}

bool Codec::decode_y_stream(const uint8_t *in, size_t n, const int32_t *indexes,
                            size_t count, int32_t *symbols,
                            std::string *err) const {
  RansDecoder dec;
  std::vector<int32_t> syms;
  try {
    syms = dec.decode_with_indexes(
        in, n, std::vector<int32_t>(indexes, indexes + count), gauss_cdf_,
        gauss_cdf_length_, gauss_offset_);
  } catch (const std::exception &e) {
    if (err) *err = std::string("rans y decode failed: ") + e.what();
    return false;
  }
  std::copy(syms.begin(), syms.end(), symbols);
  return true;
}

/* ====================================================================== */
/*  full encode / decode                                                  */
/* ====================================================================== */

bool Codec::encode(const float *y, const float *z_hat, int y_h, int y_w,
                   int z_h, int z_w, int fh_h, int fh_w, int fl_h, int fl_w,
                   const int32_t *forced_indexes, const float *forced_means,
                   std::vector<uint8_t> *out, std::string *err) const {
  const int c_y = hyper_.out_channels() / 2;
  if (c_y <= 0) {
    if (err) *err = "invalid channel count";
    return false;
  }
  const size_t y_count = static_cast<size_t>(c_y) * y_h * y_w;

  /* ---- z stream ---- */
  /*
   * z_hat holds the ROUNDED hyper latent: z_hat = round(z - median) + median.
   * The transmitted symbol is round(z_hat - median), which is exactly the
   * integer.  `encode_z_stream` subtracts the median itself (matching
   * CompressAI's EntropyBottleneck.compress, which codes round(x - median)),
   * so we hand it z_hat unchanged.
   */
  const size_t z_count = static_cast<size_t>(eb_channels_) * z_h * z_w;
  const std::vector<uint8_t> z_stream =
      encode_z_stream(z_hat, z_h, z_w, medians_.data());

  /* ---- means / scales / indexes ---- */
  std::vector<float> means(y_count), scales(y_count);
  std::vector<int32_t> indexes(y_count);
  if (forced_indexes && forced_means) {
    std::copy(forced_means, forced_means + y_count, means.begin());
    std::copy(forced_indexes, forced_indexes + y_count, indexes.begin());
  } else {
    if (!hyper_synthesis(z_hat, z_h, z_w, means.data(), scales.data(),
                         indexes.data(), y_h, y_w, err)) {
      return false;
    }
  }

  /* ---- y symbols ---- */
  std::vector<int32_t> y_symbols(y_count);
  for (size_t i = 0; i < y_count; ++i) {
    y_symbols[i] = round_half_even(y[i] - means[i]);
  }
  const std::vector<uint8_t> y_stream =
      encode_y_stream(y_symbols.data(), indexes.data(), y_count, err);
  if (err && !err->empty()) return false;

  RtffHeader hdr;
  hdr.z_h = z_h;
  hdr.z_w = z_w;
  hdr.y_h = y_h;
  hdr.y_w = y_w;
  hdr.fh_h = fh_h;
  hdr.fh_w = fh_w;
  hdr.fl_h = fl_h;
  hdr.fl_w = fl_w;
  hdr.batch = 1;
  *out = build_container(hdr, z_stream, y_stream);
  return true;
}

bool Codec::decode(const uint8_t *in, size_t n, float *y_hat, size_t cap_floats,
                   int *dims, std::string *err) const {
  RtffHeader hdr;
  std::vector<uint8_t> z_stream, y_stream;
  if (!parse_container(in, n, &hdr, &z_stream, &y_stream, err)) return false;

  const int c_y = hyper_.out_channels() / 2;
  const size_t y_count = static_cast<size_t>(c_y) * hdr.y_h * hdr.y_w;
  const size_t z_count = static_cast<size_t>(eb_channels_) * hdr.z_h * hdr.z_w;

  if (dims) {
    dims[0] = hdr.z_h;
    dims[1] = hdr.z_w;
    dims[2] = hdr.y_h;
    dims[3] = hdr.y_w;
    dims[4] = hdr.fh_h;
    dims[5] = hdr.fh_w;
    dims[6] = hdr.fl_h;
    dims[7] = hdr.fl_w;
  }
  if (cap_floats < y_count) {
    if (err) *err = "y_hat buffer too small";
    return false;
  }

  /* ---- z ---- */
  std::vector<float> z_hat(z_count);
  if (!decode_z_stream(z_stream.data(), z_stream.size(), hdr.z_h, hdr.z_w,
                       z_hat.data(), err)) {
    return false;
  }

  /* ---- means / scales / indexes ---- */
  std::vector<float> means(y_count), scales(y_count);
  std::vector<int32_t> indexes(y_count);
  if (!hyper_synthesis(z_hat.data(), hdr.z_h, hdr.z_w, means.data(),
                       scales.data(), indexes.data(), hdr.y_h, hdr.y_w, err)) {
    return false;
  }

  /* ---- y ---- */
  std::vector<int32_t> symbols(y_count);
  if (!decode_y_stream(y_stream.data(), y_stream.size(), indexes.data(),
                       y_count, symbols.data(), err)) {
    return false;
  }
  for (size_t i = 0; i < y_count; ++i) {
    y_hat[i] = static_cast<float>(symbols[i]) + means[i];
  }
  return true;
}

}  // namespace sscodec
