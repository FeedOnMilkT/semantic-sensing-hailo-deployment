from __future__ import annotations

import unittest

from hailo_deploy.io.fullmodule_selection import canonical_fingerprint, select_candidate
from scripts.evaluate_fullmodule_hn_chain import IMAGE_PIXELS, bitstream_totals, bpp_from_totals


THRESHOLDS = {
    "max_total_bpp": 0.4,
    "min_final_logits_cosine": 0.95,
    "max_argmax_gap_pp": 1.0,
    "max_fixed19_gap_pp": 2.0,
    "rate_absolute_bpp": 0.01,
    "rate_relative_fraction": 0.03,
    "simplicity_order": ["C1", "C2", "C3"],
}


def bundle(bpp: float, cosine: float = 0.99, argmax: float = 0.98, fixed19: float = 0.35):
    return {
        "golden": {
            "status": "PASS", "real_rans_success_count": 8,
            "final_logits_cumulative_metrics": {"cosine": {"mean": cosine}},
            "final_argmax": {"agreement": argmax},
            "semantic": {"sdk_quantized_chain": {"fixed_19_large_object_macro_iou": fixed19}},
        },
        "screen": {
            "status": "PASS", "real_rans_success_count": 64,
            "bitstream": {"sdk_quantized": {"mean": bpp}},
        },
    }


class FullmoduleSelectiveCampaignTests(unittest.TestCase):
    def test_canonical_fingerprint_ignores_mapping_order(self):
        self.assertEqual(canonical_fingerprint({"a": 1, "b": 2}),
                         canonical_fingerprint({"b": 2, "a": 1}))

    def test_rate_close_prefers_simpler_c1(self):
        decision = select_candidate({"C1": bundle(0.350), "C2": bundle(0.344), "C3": bundle(0.343)}, THRESHOLDS)
        self.assertEqual(decision["status"], "UNIQUE_SELECT")
        self.assertEqual(decision["selected"], "C1")

    def test_qft_requires_absolute_and_relative_improvement(self):
        decision = select_candidate({"C1": bundle(0.36), "C2": bundle(0.35), "C3": bundle(0.33)}, THRESHOLDS)
        self.assertEqual(decision["selected"], "C3")
        weak = select_candidate({"C1": bundle(0.36), "C2": bundle(0.35), "C3": bundle(0.345)}, THRESHOLDS)
        self.assertNotEqual(weak["selected"], "C3")

    def test_quality_rate_conflict_pauses(self):
        decision = select_candidate({"C1": bundle(0.36, fixed19=0.37),
                                     "C2": bundle(0.33, fixed19=0.34)}, THRESHOLDS)
        self.assertEqual(decision["status"], "PAUSED_AMBIGUOUS_SELECTION")

    def test_no_eligible_candidate_stops(self):
        decision = select_candidate({"C1": bundle(0.41), "C2": bundle(0.42)}, THRESHOLDS)
        self.assertEqual(decision["status"], "STOP_NO_ELIGIBLE_CANDIDATE")

    def test_integer_bitstream_totals_keep_z_y_overhead(self):
        rows = [{"fp_z_bytes": 10, "fp_y_bytes": 20, "quant_z_bytes": 11,
                 "quant_y_bytes": 22, "fp_bytes": 59, "quant_bytes": 62}]
        totals = bitstream_totals(rows)
        self.assertEqual(totals["container_overhead_bytes"], 29)
        self.assertEqual(totals["quant_z_bytes"], 11)
        self.assertEqual(totals["quant_y_bytes"], 22)
        self.assertEqual(totals["image_pixels"], IMAGE_PIXELS)
        self.assertAlmostEqual(bpp_from_totals(totals)["sdk_quantized"]["mean"], 62 * 8 / IMAGE_PIXELS)


if __name__ == "__main__":
    unittest.main()
