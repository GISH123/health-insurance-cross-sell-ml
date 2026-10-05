"""One fixed, conservative XGBoost challenger; no early stopping or tuning."""

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data import RANDOM_STATE
from .tree_features import TreeFrameEncoder


XGBOOST_PARAMETERS = {
    "objective": "binary:logistic",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10.0,
    "reg_lambda": 5.0,
    "reg_alpha": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "max_bin": 256,
    "eval_metric": "logloss",
    "device": "cpu",
    "random_state": RANDOM_STATE,
    "n_jobs": 4,
}


def build_xgboost_pipeline(parameters=None):
    configuration = dict(XGBOOST_PARAMETERS)
    if parameters:
        configuration.update(parameters)
    return Pipeline([
        ("prepare", TreeFrameEncoder()),
        ("classifier", XGBClassifier(**configuration)),
    ])
