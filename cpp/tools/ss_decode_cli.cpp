/*
 * ss_decode_cli.cpp -- decode an RTFF container back into the y_hat latent.
 *
 * Usage:
 *   ss_decode_cli --tables codec_tables.bin --in out.rtff --out y_hat.bin
 *   ss_decode_cli --tables ... --in out.rtff --out y_hat.bin --stats
 */
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
  const char *tables = nullptr;
  const char *in_path = nullptr;
  const char *out_path = nullptr;
  bool stats = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *&dst) {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        exit(2);
      }
      dst = argv[++i];
    };
    if (a == "--tables") need(tables);
    else if (a == "--in") need(in_path);
    else if (a == "--out") need(out_path);
    else if (a == "--stats") stats = true;
    else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (!tables || !in_path) {
    fprintf(stderr,
            "usage: ss_decode_cli --tables <f> --in <rtff> [--out <bin>] "
            "[--stats]\n");
    return 2;
  }

  SsCodec *c = ss_open(tables);
  if (!c) {
    fprintf(stderr, "ss_open failed for %s\n", tables);
    return 1;
  }

  std::vector<uint8_t> raw;
  if (!read_file(in_path, &raw)) {
    fprintf(stderr, "cannot read %s\n", in_path);
    ss_close(c);
    return 1;
  }

  std::vector<float> y_hat(192 * 16 * 16);
  int dims[8] = {0};
  const int rc = ss_decode(c, raw.data(), raw.size(), y_hat.data(), y_hat.size(),
                           dims);
  if (rc != 0) {
    fprintf(stderr, "ss_decode failed: %d\n", rc);
    ss_close(c);
    return 1;
  }

  printf("dims: z=%dx%d y=%dx%d fh=%dx%d fl=%dx%d\n", dims[0], dims[1], dims[2],
         dims[3], dims[4], dims[5], dims[6], dims[7]);
  printf("bytes: %zu  bpp = %.5f\n", raw.size(),
         raw.size() * 8.0 / (512.0 * 512.0));

  if (stats) {
    double sum = 0.0, mn = 1e30, mx = -1e30;
    for (float v : y_hat) {
      sum += v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    printf("y_hat: n=%zu mean=%.6f min=%.4f max=%.4f\n", y_hat.size(),
           sum / static_cast<double>(y_hat.size()), mn, mx);
  }

  if (out_path) {
    FILE *f = fopen(out_path, "wb");
    if (!f) {
      fprintf(stderr, "cannot write %s\n", out_path);
      ss_close(c);
      return 1;
    }
    fwrite(y_hat.data(), sizeof(float), y_hat.size(), f);
    fclose(f);
    printf("wrote %s\n", out_path);
  }

  ss_close(c);
  return 0;
}
