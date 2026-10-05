"""Fixed CatBoost challenger with stateless, consistent categorical formatting."""

import hashlib

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .data import RANDOM_STATE
from .train import NUMERIC


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
