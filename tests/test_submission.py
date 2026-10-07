import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import submit_catboost
from src.train_catboost import CATBOOST_PARAMETERS, FEATURES, build_catboost_pipeline
from test_baseline import example_frame


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.train = example_frame()
        self.test = self.train.iloc[[9, 2, 7, 0]].drop(columns="Response").copy()
        self.test["id"] = ["0009", "0002", "0007", "0000"]

    def small_model(self):
        return build_catboost_pipeline({"iterations": 6, "thread_count": 1})

    def test_full_fit_probability_submission_and_csv_preserve_rows_and_ids(self):
        model = self.small_model()
        submission = submit_catboost.make_submission(self.train, self.test, model=model)
        self.assertEqual(len(submission), len(self.test))
        self.assertEqual(list(submission.columns), ["id", "Response"])
        self.assertEqual(submission["id"].tolist(), self.test["id"].tolist())
        self.assertTrue(pd.api.types.is_numeric_dtype(submission["Response"]))
        self.assertTrue(submission["Response"].between(0, 1).all())
        self.assertFalse(submission["Response"].isin([0, 1]).all())
        self.assertEqual(model.named_steps["prepare"].fit_rows_, len(self.train))
        self.assertEqual(model.named_steps["classifier"].feature_names_, FEATURES)
        self.assertNotIn("id", model.named_steps["classifier"].feature_names_)
        self.assertNotIn("Response", model.named_steps["classifier"].feature_names_)
        expected = model.predict_proba(self.test.loc[:, FEATURES])[:, list(model.classes_).index(1)]
        np.testing.assert_array_equal(submission["Response"], expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            submit_catboost.write_submission(submission, self.test, path)
            saved = pd.read_csv(path, dtype={"id": str})
            self.assertEqual(list(saved.columns), ["id", "Response"])
            self.assertEqual(saved["id"].tolist(), self.test["id"].tolist())
            self.assertEqual(len(saved), len(self.test))

    def test_target_schema_is_enforced_before_fitting(self):
        with self.assertRaisesRegex(ValueError, "train.*Response"):
            submit_catboost.make_submission(self.train.drop(columns="Response"), self.test)
        with self.assertRaisesRegex(ValueError, "test.*Response"):
            submit_catboost.make_submission(self.train, self.test.assign(Response=0))

    def test_only_approved_predictors_reach_fit_and_predict(self):
        class RecordingModel:
            classes_ = np.array([1, 0])

            def fit(inner, x, y):
                self.assertEqual(list(x.columns), FEATURES)
                self.assertEqual(len(x), len(self.train))
                pd.testing.assert_series_equal(y, self.train["Response"])
                return inner

            def predict_proba(inner, x):
                self.assertEqual(list(x.columns), FEATURES)
                return np.tile([0.3, 0.7], (len(x), 1))

        submission = submit_catboost.make_submission(
            self.train.assign(unapproved=1), self.test.assign(unapproved=2), model=RecordingModel()
        )
        np.testing.assert_array_equal(submission["Response"], np.full(len(self.test), 0.3))

    def test_invalid_submission_outputs_are_rejected(self):
        valid = pd.DataFrame({"id": self.test["id"].to_numpy(), "Response": [0.2, 0.3, 0.4, 0.5]})
        invalid = [valid.iloc[:-1], valid[["Response", "id"]], valid.assign(id="wrong"),
                   valid.assign(Response="text"), valid.assign(Response=np.nan),
                   valid.assign(Response=1.1), valid.assign(Response=-0.1), valid.assign(Response=0)]
        for submission in invalid:
            with self.subTest(submission=submission):
                with self.assertRaises(ValueError):
                    submit_catboost.validate_submission(submission, self.test)

    def test_standalone_uses_repository_configuration_and_same_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catboost_submission_code.py"
            submit_catboost.export_standalone(path)
            specification = importlib.util.spec_from_file_location("standalone_submission", path)
            standalone = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(standalone)
            self.assertEqual(standalone.CATBOOST_PARAMETERS, CATBOOST_PARAMETERS)
            self.assertEqual(standalone.FEATURES, FEATURES)
            actual = standalone.make_submission(
                self.train, self.test,
                model=standalone.build_catboost_pipeline({"iterations": 6, "thread_count": 1}),
            )
            expected = submit_catboost.make_submission(self.train, self.test, model=self.small_model())
            pd.testing.assert_frame_equal(actual, expected)
            train_path, test_path = Path(directory) / "train.csv", Path(directory) / "test.csv"
            output_path = Path(directory) / "submission.csv"
            self.train.to_csv(train_path, index=False)
            self.test.to_csv(test_path, index=False)
            completed = subprocess.run(
                [sys.executable, "-B", str(path), "--train", str(train_path),
                 "--test", str(test_path), "--output", str(output_path)],
                cwd=directory, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["training_rows"], len(self.train))
            saved = pd.read_csv(output_path, dtype={"id": str})
            submit_catboost.validate_submission(saved, self.test)


if __name__ == "__main__":
    unittest.main()
