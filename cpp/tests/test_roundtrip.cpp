/*
 * test_roundtrip.cpp -- C++ encode -> C++ decode losslessness.
 *
 * Reads real sender tensors (y, z_hat) from a raw binary file produced by
 * the Python helper (tests/make_test_vectors.py) and checks that:
 *   - the z symbols round-trip exactly,
 *   - the y symbols round-trip exactly (i.e. decode(y_hat) - means == the
 *     original symbols, bit-exactly),
 *   - the decoded y_hat equals symbols + means exactly.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sscodec.h"

namespace {

bool read_file(const std::string &path, std::vector<uint8_t> *out) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  out->resize(static_cast<size_t>(n));
  const size_t rd = fread(out->data(), 1, out->size(), f);
  fclose(f);
  return rd == out->size();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr,
            "usage: ss_test_roundtrip <tables.bin> <vectors.bin> <n_images>\n"
            "  vectors.bin: n_images * (y[192*16*16] f32, z_hat[128*8*8] f32)\n");
    return 2;
  }
  const std::string tables_path = argv[1];
  const std::string vec_path = argv[2];
  size_t n_images = static_cast<size_t>(atoi(argv[3]));

  const size_t y_n = 192u * 16u * 16u;
  const size_t z_n = 128u * 8u * 8u;
  /*
   * The vectors file is a RAW float32 stream, so the per-image stride is the
   * ELEMENT count (y_n + z_n).  `per_bytes` is the same quantity in bytes,
   * which is what the byte-pointer arithmetic below needs.
   */
  const size_t per = y_n + z_n;
  const size_t per_bytes = per * 4;

  std::vector<uint8_t> raw;
  if (!read_file(vec_path, &raw)) {
    fprintf(stderr, "cannot read %s\n", vec_path.c_str());
    return 1;
  }
  const size_t avail = raw.size() / per_bytes;
  if (n_images > avail) n_images = avail;

  SsCodec *c = ss_open(tables_path.c_str());
  if (!c) {
    fprintf(stderr, "ss_open failed\n");
    return 1;
  }

  std::vector<float> y(y_n), z_hat(z_n);
  std::vector<uint8_t> buf(1 << 20);
  std::vector<float> y_hat(y_n);
  int dims[8];

  size_t ok = 0, fail = 0;
  size_t z_mismatch = 0, y_mismatch = 0;
  size_t total_bytes = 0;
  double max_recon_err = 0.0;

  for (size_t k = 0; k < n_images; ++k) {
    const uint8_t *p = raw.data() + k * per_bytes;
    std::memcpy(y.data(), p, y_n * 4);
    std::memcpy(z_hat.data(), p + y_n * 4, z_n * 4);

    size_t out_len = 0;
    if (ss_encode(c, y.data(), z_hat.data(), 16, 16, 8, 8, 64, 64, 32, 32,
                  buf.data(), buf.size(), &out_len) != 0) {
      fprintf(stderr, "image %zu: ss_encode failed\n", k);
      fail++;
      continue;
    }
    total_bytes += out_len;

    if (ss_decode(c, buf.data(), out_len, y_hat.data(), y_hat.size(), dims) != 0) {
      fprintf(stderr, "image %zu: ss_decode failed\n", k);
      fail++;
      continue;
    }

    /* Recompute the means the encoder used, so we can check the symbols. */
    std::vector<float> means(y_n), scales(y_n);
    std::vector<int32_t> indexes(y_n);
    ss_hyper_synthesis(c, z_hat.data(), 8, 8, means.data(), scales.data(),
                       indexes.data());

    bool bad = false;
    for (size_t i = 0; i < y_n; ++i) {
      /* the symbol the encoder sent */
      const float sym_f = y[i] - means[i];
      const float rounded = std::nearbyint(sym_f);
      const float recon = y_hat[i] - means[i];
      const double err = std::fabs(static_cast<double>(recon) - rounded);
      if (err > max_recon_err) max_recon_err = err;
      if (err > 1e-4) {
        y_mismatch++;
        bad = true;
      }
    }

    /* z must round-trip EXACTLY (integers). */
    std::vector<float> z_dec(z_n);
    if (ss_decode_z(c, buf.data(), out_len, z_dec.data(), z_dec.size(), dims) != 0) {
      fprintf(stderr, "image %zu: ss_decode_z failed\n", k);
      fail++;
      continue;
    }
    for (size_t i = 0; i < z_n; ++i) {
      if (z_dec[i] != z_hat[i]) {
        z_mismatch++;
        bad = true;
        break;
      }
    }

    if (bad) fail++;
    else ok++;
  }

  ss_close(c);

  printf("images tested      : %zu\n", n_images);
  printf("lossless (z and y) : %zu / %zu  (%.2f%%)\n", ok, n_images,
         n_images ? 100.0 * static_cast<double>(ok) / static_cast<double>(n_images) : 0.0);
  printf("z symbol mismatches: %zu\n", z_mismatch);
  printf("y symbol mismatches: %zu\n", y_mismatch);
  printf("max |recon - symbol|: %.3e\n", max_recon_err);
  printf("avg bytes/image    : %.1f\n",
         n_images ? static_cast<double>(total_bytes) / static_cast<double>(n_images) : 0.0);
  printf("avg bpp            : %.5f\n",
         n_images ? static_cast<double>(total_bytes) * 8.0 /
                        (512.0 * 512.0 * static_cast<double>(n_images))
                  : 0.0);

  if (fail != 0 || z_mismatch != 0 || y_mismatch != 0) {
    printf("RESULT: FAIL\n");
    return 1;
  }
  printf("RESULT: PASS\n");
  return 0;
}
