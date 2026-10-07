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

from .data import load_train
from .train_catboost import FEATURES, build_catboost_pipeline


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


def export_standalone(path=CODE_PATH):
    """Bundle repository source verbatim, excluding package-only imports/exporter."""
    sources = [ROOT / "src/data.py", ROOT / "src/train.py",
               ROOT / "src/train_catboost.py", Path(__file__).resolve()]
    parts = [
        '# Standalone Submission A for Analytics Vidhya Janatahack Cross-sell Prediction.\n'
        '# Generated from repository source; do not edit this copy.\n'
        '# Dependencies: pandas, numpy, scikit-learn, catboost.\n'
        '# Run from a directory containing data/raw/train.csv and data/raw/test.csv:\n'
        '# python -B catboost_submission_code.py\n'
        '# Or supply --train PATH --test PATH --output PATH.\n'
    ]
    for source_path in sources:
        source = source_path.read_text(encoding="utf-8")
        parts.append(f"# Source: {source_path.name}; SHA256: {hashlib.sha256(source.encode()).hexdigest()}")
        for node in ast.parse(source).body:
            if source_path.name in {"data.py", "train.py"}:
                wanted = "RANDOM_STATE" if source_path.name == "data.py" else "NUMERIC"
                if not (isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == wanted for target in node.targets)):
                    continue
            elif isinstance(node, ast.ImportFrom) and node.level:
                continue
            elif isinstance(node, ast.FunctionDef) and node.name == "export_standalone":
                continue
            parts.append(ast.get_source_segment(source, node))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return path


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
