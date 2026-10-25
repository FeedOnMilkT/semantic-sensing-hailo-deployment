/*
 * sscodec.h -- shared codec C ABI for the semantic-sensing Hailo deployment.
 *
 * The SAME shared library is used by the sender (C++ pipeline) and by the
 * receiver (TensorRT / PyTorch, via ctypes).  Both ends therefore agree
 * bit-exactly on:
 *   - the deterministic hyper-synthesis h_s (means / scales),
 *   - the scale -> CDF-index quantisation,
 *   - the rANS symbol streams (bit-compatible with CompressAI 1.2.8),
 *   - the RTFF container framing.
 *
 * This is deliberate: the Hailo int8 hyper-synthesis and the FP32 PyTorch
 * hyper-synthesis disagree on ~17% of the CDF "档位" (index) elements, so a
 * receiver that recomputed h_s itself would desynchronise the rANS decoder.
 * h_s is therefore NEVER computed on the device and NEVER recomputed on the
 * receiver -- it is computed here, in one place, deterministically.
 *
 * All tensors are float32 NCHW with batch size 1, channels first.
 *
 * Threading: an SsCodec handle is not internally synchronised; use one handle
 * per thread (or serialise access externally).
 */
#ifndef SSCODEC_H
#define SSCODEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque codec handle (tables + CDFs + h_s weights). */
typedef struct SsCodec SsCodec;

/*
 * Open a codec handle from a codec_tables.bin file.
 * Returns NULL on failure (missing file, bad magic, bad version, ...).
 */
SsCodec *ss_open(const char *tables_path);

/* Release a handle. Safe to call with NULL. */
void ss_close(SsCodec *c);

/*
 * Encode one image into the RTFF container.
 *
 *   y      : float32 NCHW, shape (1, 192, y_h, y_w)
 *   z_hat  : float32 NCHW, shape (1, 128, z_h, z_w); MUST hold integer values
 *            (z_hat = floor(z_pre + 0.5) + median, computed on the host).
 *   fh_h/fh_w : x_h feature map size (64, 64) -- stored in the container header.
 *   fl_h/fl_w : x_l feature map size (32, 32) -- stored in the container header.
 *   out/out_len : output buffer / its capacity in bytes; on success *out_len
 *            holds the number of bytes written.
 *
 * Returns 0 on success, negative on error.
 */
int ss_encode(SsCodec *c,
              const float *y, const float *z_hat,
              int y_h, int y_w, int z_h, int z_w,
              int fh_h, int fh_w, int fl_h, int fl_w,
              uint8_t *out, size_t cap, size_t *out_len);

/*
 * Decode one RTFF container.
 *
 *   y_hat : output buffer, float32 NCHW (1, 192, y_h, y_w) -- the dequantised
 *           latent, i.e. symbols + means.
 *   cap_floats : capacity of y_hat in floats.
 *   dims : on success receives {z_h, z_w, y_h, y_w, fh_h, fh_w, fl_h, fl_w}.
 *
 * Returns 0 on success, negative on error.
 */
int ss_decode(SsCodec *c,
              const uint8_t *in, size_t n,
              float *y_hat, size_t cap_floats, int *dims);

/*
 * Deterministic hyper-synthesis + CDF index computation (test / verification).
 *
 *   z_hat  : float32 NCHW (1, 128, z_h, z_w), integer valued.
 *   means  : output float32 (1, 192, y_h, y_w)   [h_s output, first half]
 *   scales : output float32 (1, 192, y_h, y_w)   [h_s output, second half]
 *   indexes: output int32   (1, 192, y_h, y_w)   [CDF index per element]
 *
 * NOTE: the scales used here are the RAW h_s scales (not lower-bound clipped).
 *       The CDF index uses the same lower bound as CompressAI
 *       (lower_bound_scale) -- see ss_cdf_index().
 *
 * Returns 0 on success, negative on error.
 */
int ss_hyper_synthesis(SsCodec *c,
                       const float *z_hat, int z_h, int z_w,
                       float *means, float *scales, int32_t *indexes);

/* Library version string, e.g. "sscodec 1.0.0". Never NULL. */
const char *ss_version(void);

/* ---------------------------------------------------------------------- *
 * Supplementary entry points.  These ADD to the contract above; they do not
 * change its semantics.  They exist so the Python side (and the C++ test
 * suite) can cross-validate against CompressAI 1.2.8.
 * ---------------------------------------------------------------------- */

/*
 * CDF index for a single scale value, using exactly CompressAI's rule:
 *   scale -> clamp to lower bound -> count of scale_table entries strictly
 *   greater than the (clamped) scale, minus 1  ==  CompressAI's build_indexes.
 * Returns the index, or a negative value on error.
 */
int ss_cdf_index(SsCodec *c, float scale);

/* Number of scale-table entries (64 for the shipped model). */
int ss_scale_table_size(SsCodec *c);

/*
 * Copy the scale table (float32, ss_scale_table_size entries) into out.
 * Returns 0 on success.
 */
int ss_get_scale_table(SsCodec *c, float *out, size_t cap_floats);

/*
 * Copy the per-channel entropy-bottleneck medians (float32, 128 entries) into
 * out.  These are the medians added on the host after floor(z_pre + 0.5).
 * Returns 0 on success.
 */
int ss_get_medians(SsCodec *c, float *out, size_t cap_floats);

/*
 * Encode with externally supplied CDF indexes and means (test-only entry
 * point).  Semantics identical to ss_encode(), except the y-stream symbols
 * are  round(y - means)  and the CDF index of every element is taken from the
 * caller-supplied `indexes` array instead of being derived from `scales`.
 *
 *   indexes : int32 (192 * y_h * y_w), row-major NCHW (1,192,y_h,y_w).
 *   means   : float32 (192 * y_h * y_w), row-major NCHW (1,192,y_h,y_w).
 *
 * This is what makes the CompressAI cross-check (same indexes -> same bytes)
 * possible.
 * Returns 0 on success, negative on error.
 */
int ss_encode_with_indexes(SsCodec *c,
                            const float *y, const float *z_hat,
                            int y_h, int y_w, int z_h, int z_w,
                            int fh_h, int fh_w, int fl_h, int fl_w,
                            const int32_t *indexes, const float *means,
                            uint8_t *out, size_t cap, size_t *out_len);

/*
 * Decode only the z-stream and return the reconstructed z_hat as float32
 * (1, 128, z_h, z_w).  Used by the CompressAI cross-check to compare the
 * factorized-prior stream independently of the conditional stream.
 * Returns 0 on success, negative on error.
 */
int ss_decode_z(SsCodec *c, const uint8_t *in, size_t n,
                float *z_hat, size_t cap_floats, int *dims);

/*
 * Encode only the z-stream (factorized prior) from a z tensor, exactly like
 * CompressAI's EntropyBottleneck.compress(z).  `z` is the PRE-rounding
 * hyper-latent; `medians` are the per-channel medians (use ss_get_medians).
 * The z symbols are round(z - median).
 *
 *   out/out_len : raw rANS bytes (NOT wrapped in a container).
 *
 * Returns 0 on success, negative on error.
 */
int ss_encode_z_stream(SsCodec *c, const float *z, int z_h, int z_w,
                        const float *medians,
                        uint8_t *out, size_t cap, size_t *out_len);

/*
 * Encode only the y-stream (conditional) with given symbols+indexes, exactly
 * like CompressAI's GaussianConditional.compress(y, indexes, means=means).
 * `symbols` are the already-quantised integer symbols (NCHW 1,192,y_h,y_w).
 * `out/out_len` : raw rANS bytes (NOT wrapped in a container).
 * Returns 0 on success, negative on error.
 */
int ss_encode_y_stream(SsCodec *c, const int32_t *symbols, const int32_t *indexes,
                        int y_h, int y_w,
                        uint8_t *out, size_t cap, size_t *out_len);

/*
 * Decode a raw y-stream (no container) with the given indexes.
 * `symbols_out` : int32 (192 * y_h * y_w) output.
 * Returns 0 on success, negative on error.
 */
int ss_decode_y_stream(SsCodec *c, const uint8_t *in, size_t n,
                        const int32_t *indexes, int y_h, int y_w,
                        int32_t *symbols_out, size_t cap_ints);

/* Raw z-stream decode (no container). z_hat_out : float32 (128*z_h*z_w). */
int ss_decode_z_stream(SsCodec *c, const uint8_t *in, size_t n,
                        int z_h, int z_w, float *z_hat_out, size_t cap_floats);

/*
 * Host-side integer rounding, identical to the Python
 *   z_hat = floor(z_pre + 0.5) + median
 * (see deploy/hailo_split_pipeline.py:host_round_z).
 *   z_pre  : float32 (128 * z_h * z_w) NCHW.
 *   median : float32 (128) per channel.
 *   z_hat  : float32 (128 * z_h * z_w) NCHW output.
 * Returns 0 on success, negative on error.
 */
int ss_round_z(const float *z_pre, int z_h, int z_w,
               const float *median, float *z_hat);

/* Container header size in bytes (magic + 9 fields). */
#define SSCODEC_RTFF_HEADER_SIZE 21

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SSCODEC_H */
