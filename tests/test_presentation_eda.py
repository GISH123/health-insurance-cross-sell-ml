import unittest

import numpy as np

from src.presentation_eda import cumulative_gains
from src.evaluate import top_k_metrics


class GainsTests(unittest.TestCase):
    def test_curve_counts_denominators_and_endpoints(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([.8, .8, .2, .1])
        curve = cumulative_gains(labels, scores)
        np.testing.assert_array_equal(curve["responders_captured"], [0, 1, 1, 2, 2])
        np.testing.assert_allclose(curve["capacity"], [0, .25, .5, .75, 1])
        np.testing.assert_allclose(curve["capture_rate"], [0, .5, .5, 1, 1])
        reference = top_k_metrics(labels, scores, fractions=(.25, .5, .75))
        for fraction in [.25, .5, .75]:
            count = int(fraction * len(labels))
            self.assertEqual(curve["capture_rate"][count], reference[f"top_{fraction*100:g}_pct"]["responder_capture_rate"])

    def test_score_ties_follow_input_order(self):
        curve = cumulative_gains([0, 1, 1, 0], [.8, .8, .2, .1])
        self.assertEqual(curve["responders_captured"][1], 0)

    def test_undefined_or_invalid_curve_inputs_are_rejected(self):
        for labels, scores in [([0, 0], [.2, .1]), ([1], [.1, .2]), ([1, 0], [.2, np.nan])]:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    cumulative_gains(labels, scores)


if __name__ == "__main__":
    unittest.main()
