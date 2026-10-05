import importlib.util
import unittest

import numpy as np
from sklearn.base import clone

from src.tree_features import TreeFrameEncoder
from test_baseline import example_frame


class TreeEncodingTests(unittest.TestCase):
    def test_id_excluded_and_premium_preserved(self):
        frame = example_frame()
        encoder = TreeFrameEncoder().fit(frame.iloc[:80])
        encoded = encoder.transform(frame.iloc[80:])
        self.assertNotIn("id", encoded.columns)
        self.assertNotIn("Response", encoded.columns)
        self.assertEqual(encoded.iloc[0]["Annual_Premium"], 2630.0)
        self.assertTrue(all(dtype == np.float32 for dtype in encoded.dtypes))

    def test_category_vocabulary_is_training_local(self):
        frame = example_frame()
        frame["Policy_Sales_Channel"] = np.arange(100) + 1000.0
        encoder = TreeFrameEncoder().fit(frame.iloc[:80])
        self.assertEqual(set(encoder.encoder_.categories_[-1]), {str(i) for i in range(1000, 1080)})
        self.assertEqual(encoder.fit_rows_, 80)
        encoded = encoder.transform(frame.iloc[80:])
        self.assertEqual(len(encoded), 20)
        self.assertEqual(set(encoder.encoder_.categories_[-1]), {str(i) for i in range(1000, 1080)})

    def test_unseen_codes_use_consistent_zero_encoding(self):
        frame = example_frame()
        encoder = TreeFrameEncoder().fit(frame.iloc[:80])
        unseen = frame.iloc[80:].copy()
        unseen["Policy_Sales_Channel"] = [141.0, 142.0] * 10
        encoded = encoder.transform(unseen)
        self.assertTrue(np.isfinite(encoded.to_numpy()).all())
        self.assertFalse(any("<" in column for column in encoded.columns))
        strings = unseen.copy()
        strings["Policy_Sales_Channel"] = ["141.0", "142.0"] * 10
        np.testing.assert_array_equal(encoded.to_numpy(), encoder.transform(strings).to_numpy())


class TreeModelTests(unittest.TestCase):
    def check_model(self, builder):
        frame = example_frame()
        x, y = frame.drop(columns=["Response"]), frame["Response"]
        model = clone(builder({"n_estimators": 6, "n_jobs": 1})).fit(x.iloc[:80], y.iloc[:80])
        unseen = x.iloc[80:].copy()
        unseen["Policy_Sales_Channel"] = [141.0, 142.0] * 10
        unseen["Region_Code"] = 999.0
        probabilities = model.predict_proba(unseen)[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(model.named_steps["prepare"].fit_rows_, 80)
        self.assertNotIn("id", model.named_steps["prepare"].get_feature_names_out())

    @unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM unavailable")
    def test_lightgbm_trains_and_scores_unseen_codes(self):
        from src.train_lightgbm import build_lightgbm_pipeline
        self.check_model(build_lightgbm_pipeline)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "XGBoost unavailable")
    def test_xgboost_trains_and_scores_unseen_codes(self):
        from src.train_xgboost import build_xgboost_pipeline
        self.check_model(build_xgboost_pipeline)


if __name__ == "__main__":
    unittest.main()
