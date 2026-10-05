import unittest

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.compare_models import calibration_table, compare_holdout, evaluate_folds, make_folds, score_model
from src.train import build_pipeline
from src.train_catboost import build_catboost_pipeline
from test_baseline import example_frame


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    records = []

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        self.__class__.records.append(tuple(x.index))
        self.training_mean_ = x["Annual_Premium"].mean()
        return self

    def predict_proba(self, x):
        positive = np.full(len(x), 0.5)
        return np.column_stack([1 - positive, positive])


class ComparisonTests(unittest.TestCase):
    def models(self):
        return {
            "logistic": build_pipeline(),
            "catboost": build_catboost_pipeline({"iterations": 6, "thread_count": 1}),
        }

    def test_models_use_identical_holdout_indices(self):
        result, fitted, validation = compare_holdout(example_frame(), self.models())
        self.assertEqual(len(validation), 20)
        self.assertEqual(result["logistic"]["validation_index_sha256"], result["catboost"]["validation_index_sha256"])
        self.assertEqual(result["logistic"]["validation_count"], 20)
        self.assertEqual(fitted["catboost"].named_steps["prepare"].fit_rows_, 80)

    def test_calibration_omits_empty_bins_and_counts_all_rows(self):
        table = calibration_table([0, 1, 0, 1], [0.0, 0.1, 0.1, 1.0])
        self.assertEqual(sum(r["count"] for r in table), 4)
        self.assertTrue(all(r["count"] > 0 for r in table))
        high = table[-1]
        self.assertEqual(high["mean_predicted_probability"], 1.0)
        self.assertEqual(high["observed_response_rate"], 1.0)

    def test_logistic_handles_both_known_test_only_channel_codes(self):
        frame = example_frame()
        x, y = frame.drop(columns=["id", "Response"]), frame["Response"]
        model = build_pipeline().fit(x.iloc[:80], y.iloc[:80])
        unseen = x.iloc[80:].copy()
        unseen["Policy_Sales_Channel"] = [141.0, 142.0] * 10
        self.assertTrue(np.isfinite(model.predict_proba(unseen)).all())

    def test_calibration_singleton_does_not_imply_zero_uncertainty(self):
        row = calibration_table([0], [0.65])[0]
        self.assertEqual(row["observed_rate_wilson_lower_95"], 0.0)
        self.assertGreater(row["observed_rate_wilson_upper_95"], 0.5)

    def test_ap_is_named_correctly_and_probability_quantiles_present(self):
        metrics = score_model([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertIn("log_loss", metrics)
        self.assertIn("brier_score", metrics)
        self.assertIn("p05", metrics["probability_summary"])
        self.assertIn("p95", metrics["probability_summary"])

    def test_folds_are_stratified_and_cover_each_row_once(self):
        labels = example_frame()["Response"]
        folds = make_folds(labels)
        covered = []
        for train, valid in folds:
            self.assertEqual(len(train), 80)
            self.assertEqual(len(valid), 20)
            self.assertEqual(labels.iloc[valid].sum(), 10)
            self.assertFalse(set(train) & set(valid))
            covered.extend(valid)
        self.assertEqual(sorted(covered), list(range(100)))

    def test_cv_fit_calls_use_fold_training_rows_only(self):
        frame = example_frame()
        x, y = frame.drop(columns=["id", "Response"]), frame["Response"]
        RecordingClassifier.records = []
        folds = make_folds(y)
        rows = evaluate_folds(x, y, {"first": RecordingClassifier(), "second": RecordingClassifier()}, folds)
        self.assertEqual(len(rows), 10)
        for fold_number, (train, valid) in enumerate(folds):
            expected = tuple(x.iloc[train].index)
            self.assertEqual(RecordingClassifier.records[2 * fold_number], expected)
            self.assertEqual(RecordingClassifier.records[2 * fold_number + 1], expected)
            self.assertFalse(set(expected) & set(x.iloc[valid].index))

    def test_real_cv_preprocessing_is_fold_local(self):
        frame = example_frame()
        frame["Annual_Premium"] = np.arange(100) ** 2 + 2630.0
        x, y = frame.drop(columns=["id", "Response"]), frame["Response"]
        folds = make_folds(y)
        rows = evaluate_folds(x, y, self.models(), folds)
        for row in rows:
            train, _ = folds[row["fold"] - 1]
            if row["model"] == "logistic":
                self.assertAlmostEqual(row["training_premium_scaler_mean"], x.iloc[train]["Annual_Premium"].mean())
            else:
                self.assertEqual(row["fit_rows"], 80)


if __name__ == "__main__":
    unittest.main()
