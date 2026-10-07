# Standalone Submission A for Analytics Vidhya Janatahack Cross-sell Prediction.
# Generated from repository source; do not edit this copy.
# Dependencies: pandas, numpy, scikit-learn, catboost.
# Run from a directory containing data/raw/train.csv and data/raw/test.csv:
# python -B catboost_submission_code.py
# Or supply --train PATH --test PATH --output PATH.


# Source: data.py; SHA256: d4e870c0cb45184368c2d4765f3abbf233ce28e9d78f7102b97ce838d1e42718

RANDOM_STATE = 42

# Source: train.py; SHA256: 736aa7e2cc44e26c5a44c9692a283d40cb261a60b5aa6c014d8acbe12156e1f7

NUMERIC = ["Age", "Annual_Premium", "Vintage"]

# Source: train_catboost.py; SHA256: 7407d2a7c3a9201f3e160b3c6df61cad5367acb6838e80e67d56c4019c789995

"""Fixed CatBoost challenger with stateless, consistent categorical formatting."""

import hashlib

import numpy as np

import pandas as pd

from catboost import CatBoostClassifier

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.utils.validation import check_is_fitted

CAT_FEATURES = [
    "Gender", "Driving_License", "Region_Code", "Previously_Insured",
    "Vehicle_Age", "Vehicle_Damage", "Policy_Sales_Channel",
]

FEATURES = NUMERIC + CAT_FEATURES

CODE_FEATURES = ["Driving_License", "Region_Code", "Previously_Insured", "Policy_Sales_Channel"]

CATBOOST_PARAMETERS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "loss_function": "Logloss",
    "random_seed": RANDOM_STATE,
    "thread_count": 4,
    "task_type": "CPU",
    "allow_writing_files": False,
    "verbose": False,
    "use_best_model": False,
    # Immutable tuple keeps CatBoost's deep-copied get_params sklearn-clone compatible.
    "cat_features": tuple(CAT_FEATURES),
}

def index_digest(index):
    """Fingerprint ordered original CSV row indices for split verification."""
    return hashlib.sha256(np.asarray(index, dtype="int64").tobytes()).hexdigest()

class CategoricalFrame(TransformerMixin, BaseEstimator):
    """Select approved features and stringify categories without learning mappings.

    No category vocabulary, normalization statistics, or label information is
    learned here. Fit row metadata is retained solely for leakage checks.
    """

    def fit(self, x, y=None):
        self.fit_rows_ = len(x)
        self.fit_index_sha256_ = index_digest(x.index)
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, x):
        check_is_fitted(self, "fit_rows_")
        result = x.loc[:, FEATURES].copy()
        for column in CAT_FEATURES:
            if column in CODE_FEATURES:
                # 141, 141.0 and "141.0" have the same categorical representation.
                values = pd.to_numeric(result[column], errors="raise")
                result[column] = values.map(
                    lambda value: str(int(value)) if float(value).is_integer() else str(float(value))
                )
            else:
                result[column] = result[column].astype(str)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)

def build_catboost_pipeline(parameters=None):
    """Create an unfitted challenger. Overrides are intended for tiny unit tests."""
    configuration = dict(CATBOOST_PARAMETERS)
    if parameters:
        configuration.update(parameters)
    return Pipeline([
        ("prepare", CategoricalFrame()),
        ("classifier", CatBoostClassifier(**configuration)),
    ])

# Source: submit_catboost.py; SHA256: 41a685744f88c2052337639f7210b5e266790e3b23ccd535afb8b42743f2478f

"""Submission A: fixed CatBoost fitted on all labeled rows, without tuning.

Run from the repository root: python -B -m src.submit_catboost
The exported standalone file requires pandas, numpy, scikit-learn and catboost.
"""

import argparse

import ast

import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if __package__ else Path.cwd()

SUBMISSION_PATH = ROOT / "outputs/submissions/catboost_full_train_submission.csv"

CODE_PATH = ROOT / "outputs/submissions/catboost_submission_code.py"

def validate_submission(submission, test):
    """Reject changed rows/IDs, incorrect schema, or hard-label output."""
    if list(submission.columns) != ["id", "Response"]:
        raise ValueError("Submission columns must be exactly id,Response in that order")
    if len(submission) != len(test):
        raise ValueError("Submission row count differs from test")
    if not submission["id"].reset_index(drop=True).equals(test["id"].reset_index(drop=True)):
        raise ValueError("Submission IDs or their order differ from test")
    scores = submission["Response"]
    if not pd.api.types.is_numeric_dtype(scores):
        raise ValueError("Response probabilities must be numeric")
    if not np.isfinite(scores.to_numpy()).all() or not scores.between(0, 1).all():
        raise ValueError("Response probabilities must be finite and within [0, 1]")
    if scores.isin([0, 1]).all():
        raise ValueError("Response must contain continuous probabilities, not all binary labels")

def make_submission(train, test, model=None):
    """Fit every labeled row; select only approved features and class-1 scores."""
    if "Response" not in train.columns:
        raise ValueError("train must contain Response")
    if "Response" in test.columns:
        raise ValueError("test must not contain Response")
    for name, frame in [("train", train), ("test", test)]:
        missing = [column for column in ["id", *FEATURES] if column not in frame.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
    if "id" in FEATURES or "Response" in FEATURES:
        raise ValueError("Approved features must exclude id and Response")
    if train["Response"].isna().any() or set(train["Response"].unique()) != {0, 1}:
        raise ValueError("train Response must contain both binary classes 0 and 1")
    if model is None:
        model = build_catboost_pipeline()
    model.fit(train.loc[:, FEATURES], train["Response"])
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Fitted classifier must expose binary classes 0 and 1")
    probabilities = model.predict_proba(test.loc[:, FEATURES])[:, classes.index(1)]
    submission = pd.DataFrame({"id": test["id"].to_numpy(copy=True), "Response": probabilities})
    validate_submission(submission, test)
    return submission

def write_submission(submission, test, path):
    """Write without an index, then check the CSV actually saved to disk."""
    validate_submission(submission, test)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    saved = pd.read_csv(path, dtype={"id": test["id"].dtype})
    validate_submission(saved, test)
    np.testing.assert_allclose(saved["Response"], submission["Response"], rtol=0, atol=1e-15)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/raw/test.csv")
    parser.add_argument("--output", type=Path, default=SUBMISSION_PATH)
    args = parser.parse_args()
    train = pd.read_csv(args.train) if not __package__ else load_train(args.train)
    # Read IDs as text to preserve their original CSV representation, including leading zeros.
    test = pd.read_csv(args.test, dtype={"id": str})
    model = build_catboost_pipeline()
    submission = make_submission(train, test, model=model)
    write_submission(submission, test, args.output)
    if __package__:
        export_standalone()
    import catboost
    import sklearn
    print(json.dumps({
        "submission": str(args.output),
        "standalone_code": str(CODE_PATH) if __package__ else str(Path(__file__).resolve()),
        "training_rows": model.named_steps["prepare"].fit_rows_,
        "test_rows": len(test), "submission_rows": len(submission),
        "min_probability": float(submission["Response"].min()),
        "max_probability": float(submission["Response"].max()),
        "mean_probability": float(submission["Response"].mean()),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
        "classifier_parameters": model.named_steps["classifier"].get_params(),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "catboost": catboost.__version__},
    }, indent=2))

if __name__ == "__main__":
    main()
