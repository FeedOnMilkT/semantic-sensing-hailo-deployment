/*
 * sscodec_api.cpp -- the C ABI (see include/sscodec.h).
 *
 * Thin wrapper around sscodec::Codec.  Every entry point validates its
 * arguments and returns a negative value on error (never throws across the
 * ABI boundary).
 */
#include "sscodec.h"

#include <cmath>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "codec.h"

using sscodec::Codec;

struct SsCodec {
  Codec codec;
};

extern "C" {

SsCodec *ss_open(const char *tables_path) {
  if (!tables_path) return nullptr;
  SsCodec *h = new (std::nothrow) SsCodec();
  if (!h) return nullptr;
  std::string err;
  if (!h->codec.open(tables_path, &err)) {
    delete h;
    return nullptr;
  }
  return h;
}

void ss_close(SsCodec *c) { delete c; }

const char *ss_version(void) { return "sscodec 1.0.0"; }

int ss_encode(SsCodec *c, const float *y, const float *z_hat, int y_h, int y_w,
              int z_h, int z_w, int fh_h, int fh_w, int fl_h, int fl_w,
              uint8_t *out, size_t cap, size_t *out_len) {
  if (!c || !y || !z_hat || !out || !out_len) return -1;
  if (y_h <= 0 || y_w <= 0 || z_h <= 0 || z_w <= 0) return -2;
  std::vector<uint8_t> buf;
  std::string err;
  if (!c->codec.encode(y, z_hat, y_h, y_w, z_h, z_w, fh_h, fh_w, fl_h, fl_w,
                       nullptr, nullptr, &buf, &err)) {
    return -3;
  }
  if (buf.size() > cap) return -4;
  std::memcpy(out, buf.data(), buf.size());
  *out_len = buf.size();
  return 0;
}

int ss_encode_with_indexes(SsCodec *c, const float *y, const float *z_hat,
                            int y_h, int y_w, int z_h, int z_w, int fh_h,
                            int fh_w, int fl_h, int fl_w, const int32_t *indexes,
                            const float *means, uint8_t *out, size_t cap,
                            size_t *out_len) {
  if (!c || !y || !z_hat || !indexes || !means || !out || !out_len) return -1;
  if (y_h <= 0 || y_w <= 0 || z_h <= 0 || z_w <= 0) return -2;
  std::vector<uint8_t> buf;
  std::string err;
  if (!c->codec.encode(y, z_hat, y_h, y_w, z_h, z_w, fh_h, fh_w, fl_h, fl_w,
                       indexes, means, &buf, &err)) {
    return -3;
  }
  if (buf.size() > cap) return -4;
  std::memcpy(out, buf.data(), buf.size());
  *out_len = buf.size();
  return 0;
}

int ss_decode(SsCodec *c, const uint8_t *in, size_t n, float *y_hat,
              size_t cap_floats, int *dims) {
  if (!c || !in || !y_hat) return -1;
  std::string err;
  if (!c->codec.decode(in, n, y_hat, cap_floats, dims, &err)) return -3;
  return 0;
}

int ss_hyper_synthesis(SsCodec *c, const float *z_hat, int z_h, int z_w,
                       float *means, float *scales, int32_t *indexes) {
  if (!c || !z_hat || !means || !scales) return -1;
  if (z_h <= 0 || z_w <= 0) return -2;
  /*
   * The output spatial size is determined by the h_s network (the upsamples).
   * The caller does not pass it, so we derive it: the network doubles the
   * spatial size once (hyper_down = 1 -> num_up = 1).
   */
  const int out_c = c->codec.hyper().out_channels();
  int y_h = z_h, y_w = z_w;
  for (const auto &L : c->codec.hyper().layers()) {
    if (L.kind == sscodec::kHsUpsample) {
      y_h *= 2;
      y_w *= 2;
    }
  }
  (void)out_c;
  std::string err;
  if (!c->codec.hyper_synthesis(z_hat, z_h, z_w, means, scales, indexes, y_h,
                                y_w, &err)) {
    return -3;
  }
  return 0;
}

int ss_cdf_index(SsCodec *c, float scale) {
  if (!c) return -1;
  return c->codec.cdf_index(scale);
}

int ss_scale_table_size(SsCodec *c) {
  if (!c) return -1;
  return static_cast<int>(c->codec.scale_table().size());
}

int ss_get_scale_table(SsCodec *c, float *out, size_t cap_floats) {
  if (!c || !out) return -1;
  const auto &st = c->codec.scale_table();
  if (cap_floats < st.size()) return -2;
  std::memcpy(out, st.data(), st.size() * sizeof(float));
  return 0;
}

int ss_get_medians(SsCodec *c, float *out, size_t cap_floats) {
  if (!c || !out) return -1;
  const auto &m = c->codec.medians();
  if (cap_floats < m.size()) return -2;
  std::memcpy(out, m.data(), m.size() * sizeof(float));
  return 0;
}

int ss_decode_z(SsCodec *c, const uint8_t *in, size_t n, float *z_hat,
                size_t cap_floats, int *dims) {
  if (!c || !in || !z_hat) return -1;
  sscodec::RtffHeader hdr;
  std::vector<uint8_t> z_stream, y_stream;
  std::string err;
  if (!sscodec::Codec::parse_container(in, n, &hdr, &z_stream, &y_stream, &err)) {
    return -3;
  }
  const size_t count =
      static_cast<size_t>(c->codec.medians().size()) * hdr.z_h * hdr.z_w;
  if (cap_floats < count) return -4;
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
  if (!c->codec.decode_z_stream(z_stream.data(), z_stream.size(), hdr.z_h,
                                hdr.z_w, z_hat, &err)) {
    return -5;
  }
  return 0;
}

int ss_encode_z_stream(SsCodec *c, const float *z, int z_h, int z_w,
                        const float *medians, uint8_t *out, size_t cap,
                        size_t *out_len) {
  if (!c || !z || !out || !out_len) return -1;
  if (z_h <= 0 || z_w <= 0) return -2;
  const auto buf =
      c->codec.encode_z_stream(z, z_h, z_w, medians ? medians : c->codec.medians().data());
  if (buf.size() > cap) return -4;
  std::memcpy(out, buf.data(), buf.size());
  *out_len = buf.size();
  return 0;
}

int ss_encode_y_stream(SsCodec *c, const int32_t *symbols, const int32_t *indexes,
                        int y_h, int y_w, uint8_t *out, size_t cap,
                        size_t *out_len) {
  if (!c || !symbols || !indexes || !out || !out_len) return -1;
  if (y_h <= 0 || y_w <= 0) return -2;
  const size_t count =
      static_cast<size_t>(c->codec.hyper().out_channels() / 2) * y_h * y_w;
  std::string err;
  const auto buf = c->codec.encode_y_stream(symbols, indexes, count, &err);
  if (!err.empty()) return -3;
  if (buf.size() > cap) return -4;
  std::memcpy(out, buf.data(), buf.size());
  *out_len = buf.size();
  return 0;
}

int ss_decode_y_stream(SsCodec *c, const uint8_t *in, size_t n,
                        const int32_t *indexes, int y_h, int y_w,
                        int32_t *symbols_out, size_t cap_ints) {
  if (!c || !in || !indexes || !symbols_out) return -1;
  if (y_h <= 0 || y_w <= 0) return -2;
  const size_t count =
      static_cast<size_t>(c->codec.hyper().out_channels() / 2) * y_h * y_w;
  if (cap_ints < count) return -3;
  std::string err;
  if (!c->codec.decode_y_stream(in, n, indexes, count, symbols_out, &err)) {
    return -4;
  }
  return 0;
}

int ss_decode_z_stream(SsCodec *c, const uint8_t *in, size_t n, int z_h,
                        int z_w, float *z_hat_out, size_t cap_floats) {
  if (!c || !in || !z_hat_out) return -1;
  if (z_h <= 0 || z_w <= 0) return -2;
  const size_t count =
      static_cast<size_t>(c->codec.medians().size()) * z_h * z_w;
  if (cap_floats < count) return -3;
  std::string err;
  if (!c->codec.decode_z_stream(in, n, z_h, z_w, z_hat_out, &err)) return -4;
  return 0;
}

int ss_round_z(const float *z_pre, int z_h, int z_w, const float *median,
               float *z_hat) {
  if (!z_pre || !median || !z_hat) return -1;
  if (z_h <= 0 || z_w <= 0) return -2;
  const int c = 128;
  const size_t hw = static_cast<size_t>(z_h) * z_w;
  for (int ch = 0; ch < c; ++ch) {
    const float m = median[ch];
    for (size_t p = 0; p < hw; ++p) {
      const size_t i = static_cast<size_t>(ch) * hw + p;
      /* floor(x + 0.5) -- round-half-up, exactly like the Python
       * host_round_z() in deploy/hailo_split_pipeline.py. */
      z_hat[i] = std::floor(z_pre[i] + 0.5f) + m;
    }
  }
  return 0;
}

} /* extern "C" */
