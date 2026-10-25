/*
 * tables_format.h -- binary layout of codec_tables.bin.
 *
 * The file is produced by cpp/tools/export_codec_tables.py and consumed by
 * libsscodec (ss_open).  It carries everything the codec needs that is NOT
 * computed at runtime:
 *
 *   - the deterministic h_s hyper-synthesis weights (Conv -> BatchNorm ->
 *     ReLU -> bilinear upsample -> Conv -> BatchNorm -> ReLU -> Conv),
 *   - the GaussianConditional scale table and its quantised CDFs,
 *   - the EntropyBottleneck quantiles/medians and its quantised CDFs,
 *   - the per-channel entropy-bottleneck medians used by the host rounding.
 *
 * ALL multi-byte integers are LITTLE-ENDIAN.  All floats are IEEE-754
 * binary32, little-endian.
 *
 * FILE LAYOUT
 * -----------
 *   offset 0   : magic "SSTB"            (4 bytes)
 *   offset 4   : u32 format_version      (= 1)
 *   offset 8   : u32 flags               (reserved, must be 0)
 *   offset 12  : u32 h_s_layer_count
 *   offset 16  : u32 table_count
 *   offset 20  : u32 reserved0            (= 0)
 *   offset 24  : u32 reserved1            (= 0)
 *
 *   then h_s_layer_count layer descriptors (see LAYER DESCRIPTOR below),
 *   then table_count table descriptors (see TABLE DESCRIPTOR below).
 *
 * LAYER DESCRIPTOR (56-byte header + params)
 * -----------------------------------------
 *   u32  kind            (HsLayerKind)
 *   u32  flags           (HsLayerFlags)
 *   u32  in_channels
 *   u32  out_channels
 *   u32  kernel_h
 *   u32  kernel_w
 *   u32  stride_h
 *   u32  stride_w
 *   u32  pad_h
 *   u32  pad_w
 *   u32  reserved0       (= 0)
 *   u64  param_count     (number of float32 values in `params`)
 *   f32  scalar0         (eps for BatchNorm, scale factor for Upsample,
 *                         0 otherwise)
 *   f32  params[param_count]
 *
 *   kind == kHsConv      params = weight[O*I*KH*KW] (OIHW) then bias[O]
 *                         (bias present only if kHsFlagHasBias)
 *   kind == kHsBn        params = gamma[C], beta[C], mean[C], invstd[C]
 *                         (invstd = 1/sqrt(var + eps), computed by the
 *                          exporter so the runtime needs no square root)
 *   kind == kHsUpsample  params = (empty)
 *
 * TABLE DESCRIPTOR
 * ----------------
 *   u32  tag             (TableTag)
 *   u32  dtype           (0 = float32, 1 = int32)
 *   u32  ndim
 *   u32  reserved        (= 0)
 *   u64  elem_count
 *   ...  elem_count elements of `dtype`
 */
#ifndef SSCODEC_TABLES_FORMAT_H
#define SSCODEC_TABLES_FORMAT_H

#include <cstdint>

namespace sscodec {

inline constexpr uint32_t kTablesFormatVersion = 1;
inline constexpr char kTablesMagic[4] = {'S', 'S', 'T', 'B'};
inline constexpr uint32_t kTablesHeaderSize = 28;
inline constexpr uint32_t kHsLayerDescriptorSize = 56;

/* h_s layer kinds */
enum HsLayerKind : uint32_t {
  kHsConv = 1,
  kHsBn = 2,
  kHsUpsample = 3,
};

/* layer flags */
enum HsLayerFlags : uint32_t {
  kHsFlagReluAfter = 1u << 0,
  kHsFlagHasBias = 1u << 1,
};

/* table tags */
enum TableTag : uint32_t {
  kTagScaleTable = 1,      /* float32[64]    GaussianConditional scale table     */
  kTagGaussCdf = 2,        /* int32[64*L]    GaussianConditional quantised CDFs  */
  kTagGaussCdfLength = 3,  /* int32[64]      CDF row lengths                     */
  kTagGaussOffset = 4,     /* int32[64]      CDF symbol offsets                  */
  kTagEbCdf = 5,           /* int32[128*L]   EntropyBottleneck quantised CDFs    */
  kTagEbCdfLength = 6,     /* int32[128]     CDF row lengths                     */
  kTagEbOffset = 7,        /* int32[128]     CDF symbol offsets                  */
  kTagMedians = 8,         /* float32[128]   per-channel EB medians               */
  kTagQuantiles = 9,       /* float32[128*3] EB quantiles (min, median, max)    */
  kTagScaleBound = 10,     /* float32[1]     GaussianConditional scale bound      */
};

}  // namespace sscodec

#endif /* SSCODEC_TABLES_FORMAT_H */
