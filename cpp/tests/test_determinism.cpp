/*
 * test_determinism.cpp -- deterministic h_s / CDF index dump.
 *
 * Two modes:
 *   --dump <file>   : compute h_s + indexes for the input vectors and write
 *                     them to <file> as raw float32/int32.
 *   --compare <a> <b>
 *                   : compare two dumps BIT-EXACTLY.
 *
 * The three-build reproducibility check (-O0 vs -O2 vs -O3 -march=native) is
 * performed by dumping with each build and comparing the files byte for byte.
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
  if (argc < 3) {
    fprintf(stderr,
            "usage:\n"
            "  ss_test_determinism --dump <out.bin> --tables <t.bin> "
            "--vectors <v.bin> [--n N]\n"
            "  ss_test_determinism --compare <a.bin> <b.bin>\n");
    return 2;
  }

  if (std::string(argv[1]) == "--compare") {
    if (argc < 4) {
      fprintf(stderr, "--compare needs two files\n");
      return 2;
    }
    std::vector<uint8_t> a, b;
    if (!read_file(argv[2], &a) || !read_file(argv[3], &b)) {
      fprintf(stderr, "cannot read files\n");
      return 1;
    }
    if (a.size() != b.size()) {
      printf("sizes differ: %zu vs %zu\n", a.size(), b.size());
      printf("RESULT: FAIL (not bit-identical)\n");
      return 1;
    }
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) diff++;
    }
    printf("compared %zu bytes, %zu differing\n", a.size(), diff);
    if (diff != 0) {
      printf("RESULT: FAIL (not bit-identical)\n");
      return 1;
    }
    printf("RESULT: PASS (bit-identical)\n");
    return 0;
  }

  if (std::string(argv[1]) != "--dump") {
    fprintf(stderr, "unknown mode\n");
    return 2;
  }

  std::string out_path, tables_path, vec_path;
  size_t n_images = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](std::string &dst) {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        exit(2);
      }
      dst = argv[++i];
    };
    /* "--dump" is the MODE (already consumed at argv[1]); the output path
     * comes from "--out". */
    if (a == "--out") need(out_path);
    else if (a == "--tables") need(tables_path);
    else if (a == "--vectors") need(vec_path);
    else if (a == "--n") {
      std::string s;
      need(s);
      n_images = static_cast<size_t>(atoll(s.c_str()));
    }
  }
  if (out_path.empty() || tables_path.empty() || vec_path.empty()) {
    fprintf(stderr, "--dump needs --out --tables --vectors\n");
    return 2;
  }

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
  if (n_images == 0 || n_images > avail) n_images = avail;

  SsCodec *c = ss_open(tables_path.c_str());
  if (!c) {
    fprintf(stderr, "ss_open failed\n");
    return 1;
  }

  std::vector<float> means(y_n), scales(y_n);
  std::vector<int32_t> indexes(y_n);
  std::vector<float> z_hat(z_n);

  FILE *f = fopen(out_path.c_str(), "wb");
  if (!f) {
    fprintf(stderr, "cannot write %s\n", out_path.c_str());
    ss_close(c);
    return 1;
  }

  for (size_t k = 0; k < n_images; ++k) {
    const uint8_t *p = raw.data() + k * per_bytes;
    std::memcpy(z_hat.data(), p + y_n * 4, z_n * 4);
    if (ss_hyper_synthesis(c, z_hat.data(), 8, 8, means.data(), scales.data(),
                           indexes.data()) != 0) {
      fprintf(stderr, "ss_hyper_synthesis failed on image %zu\n", k);
      fclose(f);
      ss_close(c);
      return 1;
    }
    fwrite(means.data(), sizeof(float), y_n, f);
    fwrite(scales.data(), sizeof(float), y_n, f);
    fwrite(indexes.data(), sizeof(int32_t), y_n, f);
  }
  fclose(f);
  ss_close(c);

  printf("dumped %zu image(s) to %s\n", n_images, out_path.c_str());
  printf("RESULT: PASS\n");
  return 0;
}
