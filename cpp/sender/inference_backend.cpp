/*
 * inference_backend.cpp -- backend factories.
 *
 * The HailoRT backend lives in hailtort_backend.cpp and is only compiled
 * when the project is configured with -DWITH_HAILORT=ON.  Without it, the
 * factory below returns nullptr with a clear error, so the sender CLI still
 * builds and runs with the replay backend on a machine that has no Hailo
 * device.
 *
 * NOTE: this stub is NOT compiled into the same binary as the real HailoRT
 * backend.  The two definitions of `MakeHailoRTBackend` would collide at
 * link time; the CMake build keeps them apart (see cpp/CMakeLists.txt):
 *   - WITH_HAILORT=OFF          -> this stub is used;
 *   - WITH_HAILORT=ON, LINK=ON  -> the real backend (hailort_backend.cpp);
 *   - WITH_HAILORT=ON, LINK=OFF -> a compile-check build: the real backend
 *     is compiled (proving it builds against the HailoRT headers) but cannot
 *     be linked because libhailort is absent, so this stub is linked
 *     instead and the CLI reports the situation clearly.
 */
#include "inference_backend.h"

#ifndef SSCODEC_WITH_HAILORT

namespace sender {

std::unique_ptr<InferenceBackend> MakeHailoRTBackend(const std::string &hef_dir,
                                                     std::string *err) {
  (void)hef_dir;
  if (err) {
#ifdef SSCODEC_HAILORT_COMPILE_CHECK
    *err =
        "the HailoRT backend was compiled (COMPILE CHECK ONLY) but not "
        "linked, because libhailort is not available on this machine. "
        "Rebuild with -DHAILORT_LINK=ON on a machine that has HailoRT "
        "installed (e.g. the Raspberry Pi), or use --backend replay.";
#else
    *err =
        "the HailoRT backend was not compiled in (configure with "
        "-DWITH_HAILORT=ON and -DHAILORT_INCLUDE_DIR=<hailort checkout>)";
#endif
  }
  return nullptr;
}

}  // namespace sender

#endif /* SSCODEC_WITH_HAILORT */
