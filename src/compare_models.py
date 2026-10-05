r"""Reproduce the local holdout comparison and fixed five-fold robustness audit.

Run from the repository root:
    .venv\Scripts\python.exe -B -m src.compare_models

No test CSV, network, tuning, early stopping, or model recalibration is used.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from .data import RANDOM_STATE, load_train, split_train_validation
from .evaluate import evaluate_predictions
from .train import build_pipeline
from .train_catboost import CATBOOST_PARAMETERS, build_catboost_pipeline, index_digest


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
LOG_PATH = ROOT / "outputs" / "logs" / "overnight_catboost_robustness.log"
CALIBRATION_EDGES = np.array([0, .05, .10, .15, .20, .25, .30, .40, .50, .60, .80, 1.0])
CV_METRICS = ["roc_auc", "average_precision", "top_10_lift", "top_10_responder_capture_rate"]


def score_model(labels, probabilities):
    """AP is sklearn Average Precision, not trapezoidal PR-AUC."""
    result = evaluate_predictions(labels, probabilities)
    result["log_loss"] = float(log_loss(labels, probabilities, labels=[0, 1]))
    result["brier_score"] = float(brier_score_loss(labels, probabilities))
    result["threshold_note"] = "Threshold 0.5 diagnostic only"
    for name, fraction in [("p05", .05), ("p95", .95)]:
        result["probability_summary"][name] = float(np.quantile(probabilities, fraction))
    return result


def calibration_table(labels, probabilities):
    """Fixed unequal-width probability bins; emit populated bins only.

    Intervals are [lower, upper), except the final interval includes 1.0.
    Wilson intervals show bin support uncertainty, including singleton bins.
    """
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    bin_numbers = np.clip(np.searchsorted(CALIBRATION_EDGES, probabilities, side="right") - 1,
                          0, len(CALIBRATION_EDGES) - 2)
    rows = []
    for number, (lower, upper) in enumerate(zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:])):
        selected = bin_numbers == number
        count = int(selected.sum())
        if not count:
            continue
        observed = float(labels[selected].mean())
        predicted = float(probabilities[selected].mean())
        lower_ci, upper_ci = wilson_interval(observed, count)
        rows.append({
            "bin_lower": float(lower), "bin_upper": float(upper), "count": count,
            "responders": int(labels[selected].sum()),
            "mean_predicted_probability": predicted, "observed_response_rate": observed,
            "observed_minus_predicted": observed - predicted,
            "observed_rate_wilson_lower_95": lower_ci,
            "observed_rate_wilson_upper_95": upper_ci,
        })
    return rows


def wilson_interval(observed_rate, count):
    """Approximate 95% binomial interval; does not collapse at zero responders."""
    z = 1.959963984540054
    denominator = 1 + z ** 2 / count
    center = (observed_rate + z ** 2 / (2 * count)) / denominator
    radius = z * np.sqrt(observed_rate * (1 - observed_rate) / count + z ** 2 / (4 * count ** 2)) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def fit_and_score(model, x_train, y_train, x_valid, y_valid):
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_valid)[:, 1]
    metrics = score_model(y_valid, probabilities)
    metrics.update({
        "fit_and_score_seconds": time.perf_counter() - started,
        "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
        "training_count": len(x_train),
        "training_index_sha256": index_digest(x_train.index),
        "validation_index_sha256": index_digest(x_valid.index),
    })
    return metrics, probabilities


def compare_holdout(frame, models=None, on_result=None):
    """One shared split; both estimators receive exactly the same row objects."""
    x_train, x_valid, y_train, y_valid = split_train_validation(frame)
    if models is None:
        models = {"logistic": build_pipeline(), "catboost": build_catboost_pipeline()}
    results, fitted = {}, {}
    for name, template in models.items():
        model = clone(template)
        metrics, probabilities = fit_and_score(model, x_train, y_train, x_valid, y_valid)
        metrics["calibration"] = calibration_table(y_valid, probabilities)
        results[name], fitted[name] = metrics, model
        if on_result:
            on_result(name, metrics, model)
    return results, fitted, x_valid.index


def make_folds(labels):
    """Materialize one shared set of shuffled, stratified folds."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(labels)), labels))


def evaluate_folds(x, y, models, folds=None, on_result=None):
    """Clone and fit entire pipelines inside each fold, never preprocessing globally."""
    if folds is None:
        folds = make_folds(y)
    rows = []
    for number, (train_rows, valid_rows) in enumerate(folds, start=1):
        x_train, x_valid = x.iloc[train_rows], x.iloc[valid_rows]
        y_train, y_valid = y.iloc[train_rows], y.iloc[valid_rows]
        for name, template in models.items():
            model = clone(template)
            metrics, _ = fit_and_score(model, x_train, y_train, x_valid, y_valid)
            top10 = metrics["ranking"]["top_10_pct"]
            row = {
                "fold": number, "model": name,
                "training_count": len(x_train), "validation_count": len(x_valid),
                "training_response_rate": float(y_train.mean()),
                "validation_response_rate": float(y_valid.mean()),
                "training_index_sha256": metrics["training_index_sha256"],
                "validation_index_sha256": metrics["validation_index_sha256"],
                "roc_auc": metrics["roc_auc"], "average_precision": metrics["average_precision"],
                "top_10_lift": top10["lift"],
                "top_10_responder_capture_rate": top10["responder_capture_rate"],
                "fit_and_score_seconds": metrics["fit_and_score_seconds"],
                "warnings": " | ".join(metrics["warnings"]),
            }
            if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
                scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
                row["training_premium_scaler_mean"] = float(scaler.mean_[1])
            if hasattr(model, "named_steps") and "prepare" in model.named_steps:
                row["fit_rows"] = model.named_steps["prepare"].fit_rows_
                if model.named_steps["prepare"].fit_index_sha256_ != row["training_index_sha256"]:
                    raise AssertionError("CatBoost preparation saw rows outside its training fold")
            rows.append(row)
            if on_result:
                on_result(rows)
    return rows


def cv_summary(rows):
    table = pd.DataFrame(rows)
    summary = []
    for model, group in table.groupby("model"):
        for metric in CV_METRICS:
            summary.append({
                "model": model, "metric": metric, "completed_folds": len(group),
                "mean": float(group[metric].mean()), "std": float(group[metric].std(ddof=1)),
            })
    return summary


def calibration_diagnostics(results):
    """Descriptive comparison with bin support, rather than a calibrated/not-calibrated claim."""
    summary = {
        "probability_bin_edges": CALIBRATION_EDGES.tolist(),
        "interval_method": "95% Wilson observed-rate interval; descriptive bin support, not a calibration guarantee",
        "sparse_plot_marker_count_threshold": 100,
        "models": {},
        "findings": [
            "Compare log loss, Brier score and populated bin gaps jointly; none alone proves future calibration.",
            "Similar mean prediction and overall prevalence alone does not establish calibration.",
            "Brier and log loss reflect both probability estimation and discrimination.",
            "Empty bins are omitted. Bins with fewer than 100 rows are separate markers, not joined to reliability lines.",
            "Neither model was recalibrated. Bin boundaries were specified before examining results.",
        ],
    }
    for name, metrics in results.items():
        rows = metrics["calibration"]
        summary["models"][name] = {
            "weighted_absolute_bin_gap": sum(r["count"] * abs(r["observed_minus_predicted"]) for r in rows) / metrics["validation_count"],
            "log_loss": metrics["log_loss"], "brier_score": metrics["brier_score"],
            "populated_bins": len(rows), "sparse_bins": [r for r in rows if r["count"] < 100],
        }
    return summary


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def calibration_plot(table):
    """Headless plotting, with all matplotlib cache/config paths inside this repo."""
    figure_directory = ROOT / "outputs" / "figures"
    config_directory = figure_directory / "matplotlib_config"
    config_directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_directory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (reliability, support) = plt.subplots(1, 2, figsize=(11, 4.5))
    reliability.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    for model, group in table.groupby("model"):
        supported = group[group["count"] >= 100]
        sparse = group[group["count"] < 100]
        line, = reliability.plot(supported["mean_predicted_probability"], supported["observed_response_rate"],
                                 marker="o", markersize=4, label=model)
        reliability.scatter(sparse["mean_predicted_probability"], sparse["observed_response_rate"],
                            marker="x", color=line.get_color())
        for row in sparse.to_dict(orient="records"):
            reliability.annotate(f"n={row['count']}",
                                 (row["mean_predicted_probability"], row["observed_response_rate"]),
                                 xytext=(5, 5), textcoords="offset points", fontsize=8)
        support.plot(group["mean_predicted_probability"], group["count"], marker="o", label=model)
    reliability.set(xlabel="Mean predicted probability", ylabel="Observed response rate",
                    title="Holdout calibration (x: fewer than 100 rows)", xlim=(0, 1), ylim=(0, 1))
    support.set(xlabel="Mean predicted probability", ylabel="Bin observations (log scale)",
                title="Calibration bin support", yscale="log")
    reliability.legend()
    support.legend()
    figure.tight_layout()
    figure.savefig(figure_directory / "calibration.png", dpi=160)
    plt.close(figure)


def render_log(state):
    """Overwrite the single report with the current checkpoint and complete evidence."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    concerns = [
        "Customer prioritization under limited campaign capacity; no costs, revenue, ROI or budget assumed.",
        "Average Precision uses sklearn average_precision_score; it is not trapezoidal PR-AUC.",
        "Top-K: ceil(fraction*n) selected in descending probability order; ties follow original validation input order.",
        "Capture=selected responders/all validation responders; response rate=selected responders/selected count; lift=segment rate/overall rate.",
        "Threshold 0.5 diagnostic only, not an operational decision rule.",
        "Previously_Insured and Vehicle_Damage require confirmation of availability before campaign scoring; importance is association, not causation.",
        "Annual_Premium=2630 retained unchanged; meaning still unverified. No suspicious values deleted.",
        "No time/customer key is available; random stratification remains conditional on the actual sampling unit and feature timing.",
        "Exact predictor duplicates do not establish repeated customers; entity overlap cannot be ruled out with this schema.",
        "Vehicle_Age is an ordinal linear term in Logistic, but a categorical feature in CatBoost. This is the specified baseline/challenger comparison.",
        "CV uses all labeled train.csv rows with fixed parameters; fold dispersion is descriptive, not a confidence interval or tuning result.",
        "Holdout is reused from the baseline; no independent external or temporal test has been performed.",
        "Calibration bins are predeclared; sparse bins are noisy. Similar mean score and prevalence alone does not establish calibration.",
        "No class balancing, resampling, tuning, early stopping, ensembles, SHAP, recalibration, Kaggle predictions or external services used.",
        "Native importance is PredictionValuesChange on the holdout-trained CatBoost model; correlated predictors can share or redistribute importance.",
    ]
    LOG_PATH.write_text(
        "Overnight objective: baseline integrity -> fixed CatBoost -> identical holdout -> calibration -> five-fold robustness -> interpretation -> tests -> outputs.\n"
        "Reproduction command: .\\.venv\\Scripts\\python.exe -B -m src.compare_models\n"
        "Validation: one 80/20 split, stratify=Response, random_state=42; both models use identical indices.\n"
        "Cross-validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42); fresh entire pipelines in every fold.\n"
        "CatBoost: configured limit 600, all 600 fitted, no early stopping or best-iteration selection.\n"
        "Baseline source files were not modified.\n\n"
        + json.dumps(state, indent=2, allow_nan=False)
        + "\n\nMethodological concerns and metric definitions:\n- " + "\n- ".join(concerns) + "\n",
        encoding="utf-8",
    )


def local_command(arguments):
    completed = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, errors="replace")
    return {"command": arguments, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def main():
    started = time.perf_counter()
    METRICS.mkdir(parents=True, exist_ok=True)
    integrity_path = METRICS / "baseline_integrity.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8")) if integrity_path.exists() else {}
    state = {
        "status": "RUNNING", "baseline_integrity": integrity,
        "python": sys.version, "configured_catboost_parameters": CATBOOST_PARAMETERS,
        "initial_existing_tests": local_command([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_baseline.py", "-v"]),
        "initial_git_diff_check": local_command(["git", "diff", "--check"]),
        "warnings": [], "blockers": [], "incomplete_steps": [],
        "implementation_notes": [
            "New comparison tests exposed CatBoost get_params deep-copying list cat_features, which fails sklearn clone identity checks. An immutable tuple resolves this; tested with fresh full pipelines in CV.",
            "A development docstring escape warning was resolved with a raw docstring; no baseline source was changed.",
        ],
    }
    state["actual_git_branch"] = local_command(["git", "branch", "--show-current"])["stdout"].strip()
    state["requested_worktree_or_branch"] = "GISH123/loggerhead"
    state["repository_note"] = "No branch or worktree changes made; actual Git branch differs from the user-provided name."
    state["started_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    state["raw_sha256_start"] = {
        str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in [Path("data/raw/train.csv"), Path("data/raw/test.csv")]
    }
    write_json(METRICS / "run_manifest.json", state)
    render_log(state)
    try:
        if state["initial_existing_tests"]["returncode"] or state["initial_git_diff_check"]["returncode"]:
            state["baseline_integrity"] = {"status": "FAIL", "note": "Initial baseline tests or diff check failed"}
            raise RuntimeError("Baseline gate failed; challenger must not be trained")
        import catboost, matplotlib, sklearn
        state["versions"] = {"pandas": pd.__version__, "numpy": np.__version__,
                             "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                             "catboost": catboost.__version__}
        frame = load_train()
        state["raw_shape"] = list(frame.shape)
        state["split"] = {"test_size": .2, "random_state": RANDOM_STATE, "stratify": "Response"}
        state["logistic_configuration"] = build_pipeline().named_steps["classifier"].get_params()

        def holdout_checkpoint(name, metrics, model):
            if name == "logistic":
                checks = {
                    "roc_auc_reproduced": abs(metrics["roc_auc"] - .8500034) < 1e-5,
                    "ap_reproduced": abs(metrics["average_precision"] - .3383213) < 1e-5,
                    "validation_count": metrics["validation_count"] == 76222,
                    "validation_responders": metrics["responders"] == 9342,
                    "top10_lift": abs(metrics["ranking"]["top_10_pct"]["lift"] - 2.959441) < 1e-5,
                    "top20_capture": abs(metrics["ranking"]["top_20_pct"]["responder_capture_rate"] - .5608007) < 1e-5,
                    "no_convergence_warning": not any("ConvergenceWarning" in w for w in metrics["warnings"]),
                }
                if not all(checks.values()):
                    state["baseline_integrity"] = {"status": "FAIL", "checks": checks}
                    raise RuntimeError("Baseline integrity failed; challenger must not be trained")
                state["baseline_integrity"].update({"status": "PASS", "rerun_checks": checks})
                scaler = model.named_steps["preprocess"].named_transformers_["numeric"]
                state["logistic_training_numeric_means"] = scaler.mean_.tolist()
            write_json(METRICS / f"{name}_holdout.json", metrics)
            state[f"{name}_holdout"] = metrics
            state["warnings"].extend(metrics["warnings"])
            render_log(state)
            print(f"Holdout {name}: AUC={metrics['roc_auc']:.6f}, AP={metrics['average_precision']:.6f}, "
                  f"seconds={metrics['fit_and_score_seconds']:.1f}", flush=True)

        results, fitted, validation_indices = compare_holdout(frame, on_result=holdout_checkpoint)
        pd.DataFrame({"original_csv_row_index": validation_indices}).to_csv(METRICS / "holdout_indices.csv", index=False)
        if results["logistic"]["validation_index_sha256"] != results["catboost"]["validation_index_sha256"]:
            raise AssertionError("Holdout index mismatch")
        classifier = fitted["catboost"].named_steps["classifier"]
        parameters = {"configured": CATBOOST_PARAMETERS, "resolved": classifier.get_all_params(),
                      "early_stopping": False, "configured_iteration_limit": 600,
                      "best_iteration": None, "tree_count": int(classifier.tree_count_)}
        write_json(METRICS / "catboost_parameters.json", parameters)
        state["catboost_configuration"] = parameters
        importance = pd.DataFrame({"feature": classifier.feature_names_,
                                   "importance": classifier.get_feature_importance(type="PredictionValuesChange")})
        importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
        importance.insert(0, "rank", np.arange(1, len(importance) + 1))
        importance.to_csv(METRICS / "catboost_feature_importance.csv", index=False)
        state["catboost_native_importance"] = importance.to_dict(orient="records")
        topk = [{"model": name, "segment": segment, **values}
                for name, result in results.items() for segment, values in result["ranking"].items()]
        pd.DataFrame(topk).to_csv(METRICS / "topk_comparison.csv", index=False)
        state["topk_comparison"] = topk
        calibration = pd.DataFrame([{ "model": name, **row}
                                    for name, result in results.items() for row in result["calibration"]])
        calibration.to_csv(METRICS / "calibration.csv", index=False)
        state["calibration_diagnostics"] = calibration_diagnostics(results)
        write_json(METRICS / "calibration_summary.json", state["calibration_diagnostics"])
        try:
            calibration_plot(calibration)
        except Exception as error:
            state["warnings"].append(f"Optional calibration plot skipped: {error}")
        render_log(state)

        x, y = frame.drop(columns=["id", "Response"]), frame["Response"]
        templates = {"logistic": build_pipeline(), "catboost": build_catboost_pipeline()}

        def cv_checkpoint(rows):
            pd.DataFrame(rows).to_csv(METRICS / "cv_results.csv", index=False)
            summary = cv_summary(rows)
            # A single completed fold has no sample standard deviation.
            for record in summary:
                if np.isnan(record["std"]):
                    record["std"] = None
            pd.DataFrame(summary).to_csv(METRICS / "cv_summary.csv", index=False)
            state["cv_results"], state["cv_summary"] = rows, summary
            state["elapsed_seconds"] = time.perf_counter() - started
            render_log(state)
            last = rows[-1]
            print(f"Fold {last['fold']} {last['model']}: AUC={last['roc_auc']:.6f}, "
                  f"AP={last['average_precision']:.6f}, seconds={last['fit_and_score_seconds']:.1f}", flush=True)

        rows = evaluate_folds(x, y, templates, make_folds(y), on_result=cv_checkpoint)
        state["warnings"].extend(row["warnings"] for row in rows if row["warnings"])
        state["cv_complete"] = len(rows) == 10
        current_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in state["raw_sha256_start"]}
        state["raw_csv_sha256_unchanged"] = current_hashes == state["raw_sha256_start"]
        if not state["raw_csv_sha256_unchanged"]:
            raise AssertionError("Raw CSV hash mismatch")
        state["verification"] = {"tests": local_command([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]),
                                 "git_diff_check": local_command(["git", "diff", "--check"])}
        write_json(METRICS / "verification.json", state["verification"])
        state["verification"]["git_status"] = local_command(["git", "status", "--short", "--untracked-files=all"])
        write_json(METRICS / "verification.json", state["verification"])
        state["status"] = "COMPLETE" if all(state["verification"][key]["returncode"] == 0
                                           for key in ["tests", "git_diff_check", "git_status"]) else "VERIFICATION_FAILED"
        if state["status"] != "COMPLETE":
            state["blockers"].append("One or more final verification commands failed; inspect verification output.")
    except Exception:
        state["status"] = "BLOCKED"
        state["blockers"].append(traceback.format_exc())
        state["incomplete_steps"].append("See completed holdout/CV checkpoints; work after the exception did not run.")
        state["verification"] = {"tests": local_command([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]),
                                 "git_diff_check": local_command(["git", "diff", "--check"]),
                                 "git_status": local_command(["git", "status", "--short", "--untracked-files=all"])}
    state["elapsed_seconds"] = time.perf_counter() - started
    state["completed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    write_json(METRICS / "run_manifest.json", state)
    render_log(state)
    print(f"{state['status']}: {LOG_PATH}; elapsed {state['elapsed_seconds']:.1f}s", flush=True)
    if state["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
