"""Run the leakage-safe Logistic Regression baseline.

From the repository root: .venv\\Scripts\\python.exe -m src.train
"""

import json

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import RANDOM_STATE, load_train, split_train_validation
from .evaluate import evaluate_predictions


NUMERIC = ["Age", "Annual_Premium", "Vintage"]
CATEGORICAL = [
    "Gender",
    "Driving_License",
    "Previously_Insured",
    "Vehicle_Damage",
    "Region_Code",
    "Policy_Sales_Channel",
]
VEHICLE_AGE = ["Vehicle_Age"]


def build_pipeline():
    """Build unfitted preprocessing and a regularized logistic classifier."""
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
            (
                "vehicle_age",
                OrdinalEncoder(
                    categories=[["< 1 Year", "1-2 Year", "> 2 Years"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                VEHICLE_AGE,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def main():
    frame = load_train()
    x_train, x_valid, y_train, y_valid = split_train_validation(frame)
    model = build_pipeline()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    print(json.dumps(evaluate_predictions(y_valid, probabilities), indent=2))


if __name__ == "__main__":
    main()
