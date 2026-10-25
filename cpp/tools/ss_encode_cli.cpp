/*
 * ss_encode_cli.cpp -- encode one or more images into RTFF containers.
 *
 * The input is a raw binary file containing, for each image:
 *      float32 y   (192 * 16 * 16)   NCHW
 *      float32 z_hat (128 * 8 * 8)   NCHW
 * optionally preceded by the z_pre tensor (128*8*8) instead of z_hat
 * (see --input-mode).
 *
 * Usage:
 *   ss_encode_cli --tables codec_tables.bin --in tensors.bin --out out.rtff
 *   ss_encode_cli --tables ... --in tensors.bin --out out.rtff --input-mode zpre
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sscodec.h"

namespace {

const int kYc = 192, kYh = 16, kYw = 16;
const int kZc = 128, kZh = 8, kZw = 8;
const int kFhh = 64, kFhw = 64, kFlh = 32, kFlw = 32;

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
  const char *input_mode = "zhat";
  const char *median_path = nullptr;

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
    else if (a == "--input-mode") need(input_mode);
    else if (a == "--medians") need(median_path);
    else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (!tables || !in_path || !out_path) {
    fprintf(stderr,
            "usage: ss_encode_cli --tables <f> --in <raw> --out <rtff> "
            "[--input-mode zhat|zpre] [--medians <f>]\n");
    return 2;
  }

  SsCodec *c = ss_open(tables);
  if (!c) {
    fprintf(stderr, "ss_open failed for %s\n", tables);
    return 1;
  }

  /* medians: needed for the zpre input mode */
  std::vector<float> medians(kZc, 0.0f);
  if (median_path) {
    std::vector<uint8_t> raw;
    if (!read_file(median_path, &raw) || raw.size() < kZc * 4) {
      fprintf(stderr, "cannot read medians from %s\n", median_path);
      ss_close(c);
      return 1;
    }
    std::memcpy(medians.data(), raw.data(), kZc * 4);
  } else {
    ss_get_medians(c, medians.data(), medians.size());
  }

  std::vector<uint8_t> raw;
  if (!read_file(in_path, &raw)) {
    fprintf(stderr, "cannot read %s\n", in_path);
    ss_close(c);
    return 1;
  }

  const size_t y_n = static_cast<size_t>(kYc) * kYh * kYw;
  const size_t z_n = static_cast<size_t>(kZc) * kZh * kZw;
  const bool zpre = (std::string(input_mode) == "zpre");
  const size_t per = y_n + z_n;
  if (raw.size() < per) {
    fprintf(stderr, "input too small: %zu bytes, need %zu\n", raw.size(), per);
    ss_close(c);
    return 1;
  }

  std::vector<float> y(y_n), z_hat(z_n);
  std::vector<uint8_t> out(per * 2);

  const size_t n_images = raw.size() / per;
  printf("%s: %zu image(s), %zu bytes each\n", in_path, n_images, per);

  for (size_t k = 0; k < n_images; ++k) {
    const uint8_t *p = raw.data() + k * per;
    std::memcpy(y.data(), p, y_n * 4);
    const float *z_src = reinterpret_cast<const float *>(p + y_n * 4);
    if (zpre) {
      ss_round_z(z_src, kZh, kZw, medians.data(), z_hat.data());
    } else {
      std::memcpy(z_hat.data(), z_src, z_n * 4);
    }
    size_t out_len = 0;
    const int rc = ss_encode(c, y.data(), z_hat.data(), kYh, kYw, kZh, kZw,
                              kFhh, kFhw, kFlh, kFlw, out.data(), out.size(),
                              &out_len);
    if (rc != 0) {
      fprintf(stderr, "ss_encode failed: %d\n", rc);
      ss_close(c);
      return 1;
    }
    if (k == 0) {
      FILE *f = fopen(out_path, "wb");
      if (!f) {
        fprintf(stderr, "cannot write %s\n", out_path);
        ss_close(c);
        return 1;
      }
      fwrite(out.data(), 1, out_len, f);
      fclose(f);
      printf("wrote %s: %zu bytes (bpp = %.5f)\n", out_path, out_len,
             out_len * 8.0 / (512.0 * 512.0));
    } else {
      /* additional images are appended to the same file, each prefixed by its
       * length so the container stream stays parseable */
      FILE *f = fopen(out_path, "ab");
      if (!f) break;
      fwrite(out.data(), 1, out_len, f);
      fclose(f);
    }
  }

  ss_close(c);
  return 0;
}
