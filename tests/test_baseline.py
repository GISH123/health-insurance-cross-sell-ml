import unittest

import numpy as np
import pandas as pd

from src.data import split_train_validation
from src.evaluate import top_k_metrics
from src.train import build_pipeline


def example_frame():
    return pd.DataFrame(
        {
            "id": range(100),
            "Gender": ["Male", "Female"] * 50,
            "Age": [25, 50] * 50,
            "Driving_License": [1] * 100,
            "Region_Code": [28.0, 8.0] * 50,
            "Previously_Insured": [0, 1] * 50,
            "Vehicle_Age": ["< 1 Year", "1-2 Year"] * 50,
            "Vehicle_Damage": ["Yes", "No"] * 50,
            "Annual_Premium": [2630.0, 40000.0] * 50,
            "Policy_Sales_Channel": [26.0, 152.0] * 50,
            "Vintage": [100, 200] * 50,
            "Response": [1, 0] * 50,
        }
    )


class BaselineTests(unittest.TestCase):
    def test_split_excludes_id_and_target_from_predictors(self):
        x_train, x_valid, y_train, y_valid = split_train_validation(example_frame())
        self.assertNotIn("id", x_train.columns)
        self.assertNotIn("Response", x_train.columns)
        self.assertNotIn("id", x_valid.columns)
        self.assertEqual((len(x_train), len(x_valid)), (80, 20))

    def test_split_is_reproducible_and_stratified(self):
        frame = example_frame()
        first = split_train_validation(frame)
        second = split_train_validation(frame)
        self.assertTrue(first[1].index.equals(second[1].index))
        self.assertEqual(int(first[2].sum()), 40)
        self.assertEqual(int(first[3].sum()), 10)

    def test_preprocessing_is_fitted_on_training_only(self):
        frame = example_frame()
        x_train, x_valid, y_train, _ = split_train_validation(frame)
        x_valid = x_valid.copy()
        x_valid["Annual_Premium"] = 9999999.0
        x_valid["Policy_Sales_Channel"] = 999.0
        model = build_pipeline().fit(x_train, y_train)
        preprocessing = model.named_steps["preprocess"]
        scaler = preprocessing.named_transformers_["numeric"]
        channel_encoder = preprocessing.named_transformers_["categorical"]
        self.assertAlmostEqual(scaler.mean_[1], x_train["Annual_Premium"].mean())
        self.assertNotIn(999.0, channel_encoder.categories_[-1])
        self.assertEqual(len(model.predict_proba(x_valid)), len(x_valid))

    def test_unseen_sales_channel_does_not_crash(self):
        frame = example_frame()
        x_train, x_valid, y_train, _ = split_train_validation(frame)
        model = build_pipeline().fit(x_train, y_train)
        x_valid = x_valid.copy()
        x_valid["Policy_Sales_Channel"] = 999.0
        probabilities = model.predict_proba(x_valid)[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())

    def test_top_k_metrics_uses_ceil_and_correct_denominators(self):
        labels = np.array([1, 0, 1, 0, 0] + [0] * 15)
        scores = np.arange(20, 0, -1) / 20
        results = top_k_metrics(labels, scores)
        self.assertEqual(results["top_5_pct"]["selected_count"], 1)
        self.assertEqual(results["top_5_pct"]["responders_captured"], 1)
        self.assertEqual(results["top_5_pct"]["responder_capture_rate"], 0.5)
        self.assertEqual(results["top_5_pct"]["response_rate"], 1.0)
        self.assertEqual(results["top_5_pct"]["lift"], 10.0)
        self.assertEqual(results["top_10_pct"]["selected_count"], 2)
        self.assertEqual(results["top_10_pct"]["responders_captured"], 1)
        self.assertEqual(results["top_20_pct"]["selected_count"], 4)
        self.assertEqual(results["top_20_pct"]["responders_captured"], 2)
        self.assertEqual(results["top_20_pct"]["responder_capture_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
