"""One fixed, conservative LightGBM challenger; no early stopping or tuning."""

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from .data import RANDOM_STATE
from .tree_features import TreeFrameEncoder


LIGHTGBM_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 100,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def build_lightgbm_pipeline(parameters=None):
    configuration = dict(LIGHTGBM_PARAMETERS)
    if parameters:
        configuration.update(parameters)
    return Pipeline([
        ("prepare", TreeFrameEncoder()),
        ("classifier", LGBMClassifier(**configuration)),
    ])
