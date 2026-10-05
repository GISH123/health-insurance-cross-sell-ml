import unittest

import numpy as np
import pandas as pd

from src.train_catboost import CAT_FEATURES, FEATURES, build_catboost_pipeline
from test_baseline import example_frame


class CatBoostTests(unittest.TestCase):
    def setUp(self):
        self.frame = example_frame()
        self.x = self.frame.drop(columns=["Response"])
        self.y = self.frame["Response"]

    def build_model(self):
        return build_catboost_pipeline({"iterations": 6, "thread_count": 1})

    def test_small_fixture_trains_scores_and_excludes_id(self):
        model = self.build_model().fit(self.x.iloc[:80], self.y.iloc[:80])
        scores = model.predict_proba(self.x.iloc[80:])[:, 1]
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(model.named_steps["classifier"].feature_names_, FEATURES)
        self.assertNotIn("id", model.named_steps["classifier"].feature_names_)
        prepared = model.named_steps["prepare"].transform(self.x)
        self.assertEqual(prepared.loc[0, "Annual_Premium"], 2630.0)
        self.assertTrue(all(isinstance(prepared.iloc[0][c], str) for c in CAT_FEATURES))

    def test_unseen_categories_and_numeric_code_representations(self):
        model = self.build_model().fit(self.x.iloc[:80], self.y.iloc[:80])
        unseen = self.x.iloc[80:].copy()
        unseen["Policy_Sales_Channel"] = [141.0, 142.0] * 10
        unseen["Region_Code"] = 999.0
        unseen["Gender"] = "Unseen"
        scores = model.predict_proba(unseen)[:, 1]
        self.assertTrue(np.isfinite(scores).all())
        string_codes = unseen.copy()
        string_codes["Policy_Sales_Channel"] = ["141.0", "142.0"] * 10
        pd.testing.assert_frame_equal(
            model.named_steps["prepare"].transform(unseen),
            model.named_steps["prepare"].transform(string_codes),
        )

    def test_validation_is_not_supplied_to_fit(self):
        model = self.build_model().fit(self.x.iloc[:80], self.y.iloc[:80])
        prepare = model.named_steps["prepare"]
        self.assertEqual(prepare.fit_rows_, 80)
        self.assertEqual(model.named_steps["classifier"].tree_count_, 6)
        before = model.predict_proba(self.x.iloc[80:])
        changed = self.x.iloc[80:].copy()
        changed["Annual_Premium"] = 9999999.0
        model.predict_proba(changed)
        np.testing.assert_array_equal(before, model.predict_proba(self.x.iloc[80:]))


if __name__ == "__main__":
    unittest.main()
