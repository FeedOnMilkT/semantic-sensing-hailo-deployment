/*
 * sender_main.cpp -- the sender pipeline CLI.
 *
 * Reads 512x512x3 uint8 images (raw .u8 files or a directory of them), runs
 * the inference backend, applies the host rounding, encodes with the shared
 * codec and writes one RTFF container per image plus a manifest.json.
 *
 * Usage:
 *   sender_cpp --backend hailo|replay --tables codec_tables.bin \
 *              [--hef-dir DIR] [--replay-dir DIR] \
 *              --input <file.u8 | DIR> --out-dir DIR [--limit N]
 */
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "inference_backend.h"
#include "sender_pipeline.h"
#include "sha256.h"

namespace {

bool is_dir(const std::string &p) {
  struct stat st;
  if (stat(p.c_str(), &st) != 0) return false;
  return S_ISDIR(st.st_mode);
}

bool read_file(const std::string &path, std::vector<uint8_t> *out) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  const long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n < 0) {
    fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(n));
  const size_t rd = fread(out->data(), 1, out->size(), f);
  fclose(f);
  return rd == out->size();
}

std::vector<std::string> list_dir(const std::string &dir, const char *ext) {
  std::vector<std::string> out;
  DIR *d = opendir(dir.c_str());
  if (!d) return out;
  struct dirent *e;
  while ((e = readdir(d)) != nullptr) {
    const std::string n = e->d_name;
    if (n == "." || n == "..") continue;
    if (ext && *ext) {
      const size_t p = n.rfind(ext);
      if (p == std::string::npos || p + strlen(ext) != n.size()) continue;
    }
    out.push_back(n);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

std::string basename_noext(const std::string &p) {
  const size_t s = p.find_last_of('/');
  std::string b = (s == std::string::npos) ? p : p.substr(s + 1);
  const size_t d = b.rfind('.');
  if (d != std::string::npos) b = b.substr(0, d);
  return b;
}

std::string json_escape(const std::string &s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default: o += c;
    }
  }
  return o;
}

}  // namespace

int main(int argc, char **argv) {
  std::string backend = "replay";
  std::string tables;
  std::string hef_dir = "/root/deploy_artifacts/hailo/sender";
  std::string replay_dir;
  std::string input;
  std::string out_dir;
  std::string manifest_path_opt;  /* default: <out_dir>/manifest.json */
  int limit = 0;
  bool zhat_mode = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](std::string &dst) {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", a.c_str());
        exit(2);
      }
      dst = argv[++i];
    };
    if (a == "--backend") need(backend);
    else if (a == "--tables") need(tables);
    else if (a == "--hef-dir") need(hef_dir);
    else if (a == "--replay-dir") need(replay_dir);
    else if (a == "--input") need(input);
    else if (a == "--out-dir") need(out_dir);
    else if (a == "--manifest") need(manifest_path_opt);
    else if (a == "--limit") {
      std::string s;
      need(s);
      limit = atoi(s.c_str());
    } else if (a == "--replay-zhat") {
      zhat_mode = true;
    } else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (tables.empty() || input.empty() || out_dir.empty()) {
    fprintf(stderr,
            "usage: sender_cpp --backend <hailo|replay> --tables <f> "
            "--input <file|dir> --out-dir <dir>\n"
            "       [--hef-dir DIR] [--replay-dir DIR] [--limit N] "
            "[--manifest PATH] [--replay-zhat]\n");
    return 2;
  }
  if (mkdir(out_dir.c_str(), 0755) != 0 && !is_dir(out_dir)) {
    fprintf(stderr, "cannot create %s\n", out_dir.c_str());
    return 1;
  }

  /* ---- backend ---- */
  std::unique_ptr<sender::InferenceBackend> be;
  std::string err;
  if (backend == "hailo") {
    be = sender::MakeHailoRTBackend(hef_dir, &err);
    if (!err.empty()) {
      fprintf(stderr, "HailoRT backend: %s\n", err.c_str());
      return 1;
    }
  } else if (backend == "replay") {
    if (replay_dir.empty()) {
      fprintf(stderr, "replay backend needs --replay-dir\n");
      return 2;
    }
    if (zhat_mode) {
      be = sender::MakeReplayZhatBackend(replay_dir, &err);
    } else {
      be = sender::MakeReplayBackend(replay_dir, &err);
    }
  } else {
    fprintf(stderr, "unknown backend '%s'\n", backend.c_str());
    return 2;
  }
  if (!be) {
    fprintf(stderr, "cannot create backend '%s'\n", backend.c_str());
    return 1;
  }

  /* ---- codec ---- */
  sender::SenderPipeline pipe;
  if (!pipe.init(tables, &err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  /*
   * The replay-zhat backend reads tensors that were ALREADY rounded on the
   * host (the Hailo emulator dump stores floor(z_pre + 0.5) + median), so the
   * host rounding must NOT be applied a second time.
   */
  if (zhat_mode) pipe.set_z_hat_already_rounded(true);

  /* ---- input list ---- */
  std::vector<std::string> files;
  if (is_dir(input)) {
    files = list_dir(input, ".u8");
    for (auto &f : files) f = input + "/" + f;
  } else {
    files.push_back(input);
  }
  if (limit > 0 && static_cast<size_t>(limit) < files.size()) {
    files.resize(static_cast<size_t>(limit));
  }
  if (files.empty()) {
    fprintf(stderr, "no input files found in %s\n", input.c_str());
    return 1;
  }
  printf("backend   : %s\n", be->name().c_str());
  printf("tables    : %s\n", tables.c_str());
  printf("input     : %s (%zu file(s))\n", input.c_str(), files.size());
  printf("out-dir   : %s\n\n", out_dir.c_str());

  /* ---- run ---- */
  std::vector<uint8_t> image(512 * 512 * 3);
  std::vector<sender::EncodedImage> results;
  results.reserve(files.size());
  size_t total_bytes = 0;

  for (size_t i = 0; i < files.size(); ++i) {
    const std::string &path = files[i];
    const std::string name = basename_noext(path);
    if (!read_file(path, &image) || image.size() < 512 * 512 * 3) {
      fprintf(stderr, "skip (bad image): %s\n", path.c_str());
      continue;
    }
    be->set_current_name(name);
    sender::SenderTensors t;
    if (!be->infer(image.data(), &t, &err)) {
      fprintf(stderr, "skip (infer failed): %s: %s\n", path.c_str(), err.c_str());
      continue;
    }
    sender::EncodedImage enc;
    if (!pipe.encode(t, name, &enc, &err)) {
      fprintf(stderr, "skip (encode failed): %s: %s\n", path.c_str(), err.c_str());
      continue;
    }
    total_bytes += enc.bytes.size();

    const std::string out_path = out_dir + "/" + name + ".rtff";
    FILE *f = fopen(out_path.c_str(), "wb");
    if (!f) {
      fprintf(stderr, "cannot write %s\n", out_path.c_str());
      continue;
    }
    fwrite(enc.bytes.data(), 1, enc.bytes.size(), f);
    fclose(f);
    results.push_back(std::move(enc));

    if ((i + 1) % 50 == 0) {
      printf("  %zu/%zu encoded (avg %.1f bytes)\n", i + 1, files.size(),
             total_bytes / static_cast<double>(results.size()));
      fflush(stdout);
    }
  }

  /* ---- manifest ---- */
  double sum_bpp = 0.0;
  for (const auto &r : results) {
    sum_bpp += r.bytes.size() * 8.0 / (512.0 * 512.0);
  }
  const double avg_bpp =
      results.empty() ? 0.0 : sum_bpp / static_cast<double>(results.size());

  const std::string manifest_path =
      manifest_path_opt.empty() ? out_dir + "/manifest.json" : manifest_path_opt;
  FILE *mf = fopen(manifest_path.c_str(), "w");
  if (!mf) {
    fprintf(stderr, "cannot write %s\n", manifest_path.c_str());
    return 1;
  }
  fprintf(mf, "{\n");
  fprintf(mf, "  \"backend\": \"%s\",\n", json_escape(be->name()).c_str());
  fprintf(mf, "  \"codec_version\": \"sscodec 1.0.0\",\n");
  fprintf(mf, "  \"container\": \"RTFF\",\n");
  fprintf(mf, "  \"num_images\": %zu,\n", results.size());
  fprintf(mf, "  \"total_bytes\": %zu,\n", total_bytes);
  fprintf(mf, "  \"avg_bytes\": %.2f,\n",
          results.empty() ? 0.0 : static_cast<double>(total_bytes) /
                                    static_cast<double>(results.size()));
  fprintf(mf, "  \"avg_bpp\": %.6f,\n", avg_bpp);
  /*
   * `avg_bpp` counts the WHOLE container (header + 2 length fields + the two
   * rANS payloads).  `avg_bpp_payload` counts only the rANS payloads, which is
   * the convention used by the deployment guide (0.17583 for the Hailo int8
   * sender, 0.17467 for the PyTorch FP32 reference) -- the guide's numbers
   * come from `EntropyBottleneck.compress` / `GaussianConditional.compress`
   * and do not include any container framing.
   */
  {
    const double overhead = 21.0 + 4.0 + 4.0; /* RTFF header + 2 x u32 len */
    double sum_payload = 0.0;
    for (const auto &r : results) {
      sum_payload += (static_cast<double>(r.bytes.size()) - overhead) * 8.0 /
                     (512.0 * 512.0);
    }
    fprintf(mf, "  \"avg_bpp_payload\": %.6f,\n",
            results.empty() ? 0.0
                            : sum_payload / static_cast<double>(results.size()));
    fprintf(mf, "  \"container_overhead_bytes\": 29,\n");
  }
  fprintf(mf, "  \"images\": [\n");
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &r = results[i];
    const double bpp = r.bytes.size() * 8.0 / (512.0 * 512.0);
    const std::string sha = sender::Sha256Hex(r.bytes.data(), r.bytes.size());
    fprintf(mf,
            "    {\"name\": \"%s\", \"file\": \"%s.rtff\", \"bytes\": %zu, "
            "\"bpp\": %.6f, \"sha256\": \"%s\"}%s\n",
            json_escape(r.name).c_str(), json_escape(r.name).c_str(),
            r.bytes.size(), bpp, sha.c_str(),
            (i + 1 == results.size()) ? "" : ",");
  }
  fprintf(mf, "  ]\n}\n");
  fclose(mf);

  printf("\nencoded %zu image(s)\n", results.size());
  printf("total bytes : %zu\n", total_bytes);
  printf("avg bytes   : %.2f\n",
         results.empty() ? 0.0 : static_cast<double>(total_bytes) /
                                   static_cast<double>(results.size()));
  printf("avg bpp     : %.6f\n", avg_bpp);
  printf("manifest    : %s\n", manifest_path.c_str());

  pipe.close();
  return 0;
}
