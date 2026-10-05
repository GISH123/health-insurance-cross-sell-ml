import unittest
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.benchmark_models import blend_predictions, ensemble_definitions, run_experiment, run_from_training_data
from src.data import split_train_validation
from src.train_catboost import index_digest
from test_baseline import example_frame


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    records = []

    def __init__(self, name):
        self.name = name

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        self.__class__.records.append((self.name, tuple(x.index), tuple(x.columns)))
        return self

    def predict_proba(self, x):
        scores = np.where(x["Previously_Insured"].to_numpy() == 0, 0.8, 0.2)
        return np.column_stack([1 - scores, scores])


class FamilyBenchmarkTests(unittest.TestCase):
    def templates(self):
        return {name: RecordingClassifier(name) for name in ("logistic", "catboost", "lightgbm", "xgboost")}

    def test_probability_blend_arithmetic_and_indices(self):
        index = pd.Index([9, 2, 7])
        predictions = {
            "catboost": pd.Series([0.2, 0.4, 0.6], index=index),
            "lightgbm": pd.Series([0.4, 0.6, 0.8], index=index),
            "xgboost": pd.Series([0.6, 0.5, 0.4], index=index),
        }
        result = blend_predictions(predictions, ["catboost", "lightgbm", "xgboost"])
        np.testing.assert_allclose(result.to_numpy(), [0.4, 0.5, 0.6])
        self.assertTrue(result.index.equals(index))
        np.testing.assert_array_equal(result, blend_predictions(predictions, ["catboost", "lightgbm", "xgboost"]))

    def test_blend_rejects_misaligned_arrays(self):
        predictions = {"a": pd.Series([.1, .2], index=[1, 2]), "b": pd.Series([.2, .1], index=[2, 1])}
        with self.assertRaises(ValueError):
            blend_predictions(predictions, ["a", "b"])

    def test_blend_rejects_invalid_probabilities(self):
        predictions = {"a": pd.Series([.1, .2]), "b": pd.Series([.2, np.nan])}
        with self.assertRaises(ValueError):
            blend_predictions(predictions, ["a", "b"])

    def test_ensemble_members_are_predefined(self):
        definitions = ensemble_definitions(["logistic", "catboost", "lightgbm", "xgboost"])
        self.assertEqual(definitions["ensemble_equal_boost"], ["catboost", "lightgbm", "xgboost"])
        self.assertEqual(definitions["ensemble_cat_light_50_50"], ["catboost", "lightgbm"])

    def test_nonfinite_parameter_sentinel_has_valid_json_representation(self):
        from src.benchmark_models import parameter_record

        class ExampleEstimator:
            def get_params(self):
                return {"missing": np.nan, "n_estimators": 600}

        recorded = parameter_record(ExampleEstimator())
        self.assertEqual(recorded["missing"], "NaN")
        json.dumps(recorded, allow_nan=False)

    def test_runtime_summary_uses_sample_standard_deviation(self):
        from src.benchmark_models import summarize_cv

        rows = [{"model": "lightgbm", "roc_auc": .85, "average_precision": .37,
                 "log_loss": .26, "brier_score": .086, "top_10_lift": 3.2,
                 "top_10_responder_capture_rate": .32, "fit_and_score_seconds": seconds}
                for seconds in [10.0, 14.0]]
        runtime = [row for row in summarize_cv(rows) if row["metric"] == "fit_and_score_seconds"]
        self.assertEqual(len(runtime), 1)
        self.assertEqual(runtime[0]["mean"], 12.0)
        self.assertAlmostEqual(runtime[0]["std"], np.sqrt(8))

    def test_same_holdout_folds_and_fold_local_fit_rows(self):
        frame = example_frame().sample(frac=1, random_state=9)
        RecordingClassifier.records = []
        result = run_experiment(frame, self.templates())
        x_train, x_valid, _, _ = split_train_validation(frame)
        for name in self.templates():
            self.assertEqual(result["holdout"][name]["validation_index_sha256"], index_digest(x_valid.index))
            self.assertTrue(result["oof_predictions"][name].index.equals(frame.index))
        for _, group in pd.DataFrame(result["cv_results"]).groupby("fold"):
            self.assertEqual(group["validation_index_sha256"].nunique(), 1)
            self.assertEqual(group["training_index_sha256"].nunique(), 1)
        self.assertEqual(len(RecordingClassifier.records), 24)
        for name, indices, columns in RecordingClassifier.records[:4]:
            self.assertEqual(indices, tuple(x_train.index))
            self.assertNotIn("id", columns)
            self.assertNotIn("Response", columns)
        for number, (train, valid) in enumerate(result["folds"]):
            self.assertEqual(frame.iloc[valid]["Response"].sum(), 10)
            for _, indices, _ in RecordingClassifier.records[4 + 4 * number:8 + 4 * number]:
                self.assertEqual(indices, tuple(frame.iloc[train].index))
                self.assertFalse(set(indices) & set(frame.iloc[valid].index))
        self.assertTrue(np.isfinite(result["oof_predictions"]["ensemble_equal_boost"]).all())
        np.testing.assert_array_equal(
            result["oof_predictions"]["catboost"].to_numpy(),
            np.where(frame["Previously_Insured"].to_numpy() == 0, 0.8, 0.2),
        )
        self.assertEqual(result["oof_metrics"]["catboost"]["average_precision"], 1.0)

    def test_entrypoint_uses_only_validated_training_loader(self):
        with patch("src.benchmark_models.load_train", return_value=example_frame()) as loader:
            with patch("pandas.read_csv", side_effect=AssertionError("Unexpected CSV access")):
                result = run_from_training_data(self.templates())
        loader.assert_called_once_with()
        self.assertEqual(len(result["oof_predictions"]["catboost"]), 100)


if __name__ == "__main__":
    unittest.main()
