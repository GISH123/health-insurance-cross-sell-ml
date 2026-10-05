"""Training-local one-hot representation shared by the two new tree families."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .train import NUMERIC
from .train_catboost import CAT_FEATURES, CategoricalFrame, index_digest


class TreeFrameEncoder(TransformerMixin, BaseEstimator):
    """Keep numeric values and fit categorical vocabulary solely on training rows.

    Dense float32 provides explicit zero values, avoiding sparse-missing ambiguity
    in XGBoost. Unknown categories become all-zero blocks. Generic encoded column
    names also avoid XGBoost's restrictions on '<', '[' and ']' in feature names.
    """

    def fit(self, x, y=None):
        self.formatter_ = CategoricalFrame().fit(x)
        prepared = self.formatter_.transform(x)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.encoder_.fit(prepared[CAT_FEATURES])
        self.category_feature_names_ = self.encoder_.get_feature_names_out(CAT_FEATURES)
        self.output_columns_ = NUMERIC + [f"category_{i:03d}" for i in range(len(self.category_feature_names_))]
        self.fit_rows_ = len(x)
        self.fit_index_sha256_ = index_digest(x.index)
        return self

    def transform(self, x):
        check_is_fitted(self, "encoder_")
        prepared = self.formatter_.transform(x)
        values = np.column_stack([
            prepared[NUMERIC].to_numpy(dtype=np.float32),
            self.encoder_.transform(prepared[CAT_FEATURES]),
        ])
        return pd.DataFrame(values, index=x.index, columns=self.output_columns_)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "encoder_")
        return np.asarray(self.output_columns_, dtype=object)
