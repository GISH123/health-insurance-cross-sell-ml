r"""Controlled four-family benchmark with predefined, aligned probability blends.

Reproduce: .\.venv\Scripts\python.exe -B -m src.benchmark_models
Only the labeled training CSV is loaded. Existing validated artifacts are read
for integrity checks and never overwritten by this benchmark.
"""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.base import clone

from .compare_models import CV_METRICS, ROOT, fit_and_score, local_command, make_folds, score_model, write_json
from .data import load_train, split_train_validation
from .train import build_pipeline
from .train_catboost import build_catboost_pipeline, index_digest


METRICS = ROOT / "outputs" / "metrics"
LOG_PATH = ROOT / "outputs" / "logs" / "model_family_benchmark.log"
BOOSTING_MODELS = ["catboost", "lightgbm", "xgboost"]


def ensemble_definitions(available):
    """Predeclare weights/members without examining model scores."""
    boosting = [name for name in BOOSTING_MODELS if name in available]
    definitions = {}
    if len(boosting) >= 2:
        definitions["ensemble_equal_boost"] = boosting
    if "catboost" in available and "lightgbm" in available:
        definitions["ensemble_cat_light_50_50"] = ["catboost", "lightgbm"]
    elif "catboost" in available and "xgboost" in available:
        definitions["ensemble_cat_xgb_50_50"] = ["catboost", "xgboost"]
    return definitions


def blend_predictions(predictions, members):
    """Require ordered index alignment; do not silently reorder missing/mixed rows."""
    if len(members) < 2 or len(set(members)) != len(members):
        raise ValueError("A blend needs at least two distinct models")
    first = predictions[members[0]]
    if not first.index.is_unique:
        raise ValueError("Prediction indices must be unique")
    arrays = []
    for member in members:
        scores = predictions[member]
        if not scores.index.equals(first.index):
            raise ValueError("Prediction arrays have different indices or row order")
        values = scores.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Blend inputs must be finite probabilities in [0,1]")
        arrays.append(values)
    return pd.Series(np.column_stack(arrays).mean(axis=1), index=first.index)


def metric_row(name, metrics, evaluation):
    row = {"model": name, "evaluation": evaluation}
    for key in ["roc_auc", "average_precision", "log_loss", "brier_score", "validation_count", "responders",
                "overall_response_rate", "fit_and_score_seconds", "training_count", "training_index_sha256",
                "validation_index_sha256", "member_fit_and_score_seconds", "blend_and_score_seconds"]:
        if key in metrics:
            row[key] = metrics[key]
    for segment, values in metrics["ranking"].items():
        prefix = segment.replace("_pct", "")
        for key, value in values.items():
            row[f"{prefix}_{key}"] = value
    row["warnings"] = " | ".join(metrics.get("warnings", []))
    return row


def fit_metadata(model, expected_indices):
    if hasattr(model, "named_steps") and "prepare" in model.named_steps:
        prepare = model.named_steps["prepare"]
        if prepare.fit_index_sha256_ != index_digest(expected_indices):
            raise AssertionError("Preprocessing fit indices differ from training indices")
        metadata = {"preprocessing_fit_rows": prepare.fit_rows_, "preprocessing_fit_index_sha256": prepare.fit_index_sha256_}
        if hasattr(prepare, "encoder_"):
            metadata["training_category_cardinalities"] = json.dumps([len(v) for v in prepare.encoder_.categories_])
        return metadata
    return {}


def blend_metrics(name, members, predictions, labels, member_metrics, training_indices):
    started = time.perf_counter()
    scores = blend_predictions(predictions, members)
    metrics = score_model(labels, scores.to_numpy())
    metrics.update({
        "training_count": len(training_indices),
        "training_index_sha256": index_digest(training_indices),
        "validation_index_sha256": index_digest(scores.index),
        "member_fit_and_score_seconds": sum(member_metrics[m]["fit_and_score_seconds"] for m in members),
        "blend_and_score_seconds": time.perf_counter() - started,
        "warnings": [],
    })
    return metrics, scores


def run_experiment(frame, templates, on_checkpoint=None):
    """One holdout and one shared fold list; OOF storage uses original row positions."""
    if not frame.index.is_unique:
        raise ValueError("Original row indices must be unique")
    x_train, x_valid, y_train, y_valid = split_train_validation(frame)
    definitions = ensemble_definitions(templates)
    holdout, holdout_predictions, rows = {}, {}, []
    for name, template in templates.items():
        model = clone(template)
        metrics, scores = fit_and_score(model, x_train, y_train, x_valid, y_valid)
        metrics.update(fit_metadata(model, x_train.index))
        holdout[name] = metrics
        holdout_predictions[name] = pd.Series(scores, index=x_valid.index)
        if on_checkpoint:
            on_checkpoint("holdout", name, holdout, rows)
    for name, members in definitions.items():
        holdout[name], holdout_predictions[name] = blend_metrics(
            name, members, holdout_predictions, y_valid, holdout, x_train.index)
    if on_checkpoint:
        on_checkpoint("holdout_complete", "ensembles", holdout, rows)

    x, y = frame.drop(columns=["id", "Response"]), frame["Response"]
    folds = make_folds(y)
    oof_arrays = {name: np.full(len(frame), np.nan) for name in templates}
    fold_ids = np.zeros(len(frame), dtype=np.int8)
    for number, (train, valid) in enumerate(folds, start=1):
        if (fold_ids[valid] != 0).any():
            raise AssertionError("Repeated validation membership")
        fold_ids[valid] = number
        train_x, valid_x, train_y, valid_y = x.iloc[train], x.iloc[valid], y.iloc[train], y.iloc[valid]
        fold_predictions, fold_metrics = {}, {}
        for name, template in templates.items():
            model = clone(template)
            metrics, scores = fit_and_score(model, train_x, train_y, valid_x, valid_y)
            metrics.update(fit_metadata(model, train_x.index))
            oof_arrays[name][valid] = scores
            fold_predictions[name] = pd.Series(scores, index=valid_x.index)
            fold_metrics[name] = metrics
            row = metric_row(name, metrics, "cv")
            row.update({"fold": number, **fit_metadata(model, train_x.index)})
            rows.append(row)
            if on_checkpoint:
                on_checkpoint("cv", f"fold {number} {name}", holdout, rows)
        for name, members in definitions.items():
            metrics, _ = blend_metrics(name, members, fold_predictions, valid_y, fold_metrics, train_x.index)
            row = metric_row(name, metrics, "cv")
            row["fold"] = number
            rows.append(row)
        if on_checkpoint:
            on_checkpoint("cv_fold_complete", f"fold {number} ensembles", holdout, rows)
    if (fold_ids == 0).any() or any(not np.isfinite(values).all() for values in oof_arrays.values()):
        raise AssertionError("Incomplete OOF predictions")
    oof_predictions = {name: pd.Series(scores, index=x.index) for name, scores in oof_arrays.items()}
    for name, members in definitions.items():
        oof_predictions[name] = blend_predictions(oof_predictions, members)
    oof_metrics = {name: score_model(y, scores.to_numpy()) for name, scores in oof_predictions.items()}
    return {"holdout": holdout, "holdout_predictions": holdout_predictions, "cv_results": rows,
            "oof_predictions": oof_predictions, "oof_metrics": oof_metrics, "folds": folds,
            "fold_ids": fold_ids, "ensemble_definitions": definitions, "labels": y}


def run_from_training_data(templates, on_checkpoint=None):
    """Use the existing loader, with no alternate data source or CLI dataset option."""
    return run_experiment(load_train(), templates, on_checkpoint)


def summarize_cv(rows):
    table = pd.DataFrame(rows)
    summary = []
    columns = ["roc_auc", "average_precision", "log_loss", "brier_score", "top_10_lift", "top_10_responder_capture_rate",
               "fit_and_score_seconds", "member_fit_and_score_seconds", "blend_and_score_seconds"]
    for name, group in table.groupby("model"):
        for metric in columns:
            if metric not in group:
                continue
            values = group[metric].dropna()
            if values.empty:
                continue
            summary.append({"model": name, "metric": metric, "completed_folds": len(group),
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=1)) if len(values) > 1 else None})
    return summary


def paired_deltas(rows, candidate, reference):
    table = pd.DataFrame(rows)
    left = table[table["model"] == candidate].set_index("fold")
    right = table[table["model"] == reference].set_index("fold")
    output = {}
    for metric in CV_METRICS:
        difference = left[metric] - right[metric]
        output[metric] = {"mean_delta": float(difference.mean()), "std_delta": float(difference.std(ddof=1)),
                          "reference_std": float(right[metric].std(ddof=1)),
                          "per_fold_deltas": difference.tolist()}
    return output


def comparative_decision(result, available):
    """Use predeclared AP priority and a conservative descriptive gain rule."""
    summary = pd.DataFrame(summarize_cv(result["cv_results"]))
    pivot = summary.pivot(index="model", columns="metric", values="mean")
    ranked_singles = pivot.loc[list(available)].sort_values(["average_precision", "top_10_lift", "roc_auc"], ascending=False)
    best = ranked_singles.index[0]
    improvement = paired_deltas(result["cv_results"], best, "catboost")
    ap, lift = improvement["average_precision"], improvement["top_10_lift"]
    meaningful = ap["mean_delta"] > ap["reference_std"] and lift["mean_delta"] > lift["reference_std"]
    if best == "catboost":
        classification = "CatBoost remains the strongest single under mean CV AP; no improvement over itself."
    elif meaningful:
        classification = "Meaningful under the predeclared descriptive rule: AP and targeting lift gains exceed CatBoost fold dispersion."
    elif abs(ap["mean_delta"]) < .1 * ap["reference_std"] and abs(lift["mean_delta"]) < .1 * lift["reference_std"]:
        classification = "Negligible compared with ordinary CatBoost fold dispersion."
    else:
        classification = "Marginal compared with ordinary CatBoost fold dispersion; inspect mixed ranking tradeoffs."
    ensemble_comparisons = {}
    for name in result["ensemble_definitions"]:
        delta = paired_deltas(result["cv_results"], name, best)
        ap_delta, lift_delta = delta["average_precision"], delta["top_10_lift"]
        holdout_improves = (result["holdout"][name]["average_precision"] > result["holdout"][best]["average_precision"]
                           and result["holdout"][name]["ranking"]["top_10_pct"]["responder_capture_rate"]
                           > result["holdout"][best]["ranking"]["top_10_pct"]["responder_capture_rate"])
        worthwhile = (ap_delta["mean_delta"] > ap_delta["reference_std"]
                      and lift_delta["mean_delta"] > lift_delta["reference_std"] and holdout_improves)
        ensemble_comparisons[name] = {"versus": best, "paired_deltas": delta,
                                     "holdout_ap_and_capture_improve": holdout_improves,
                                     "complexity_justified_by_gain_rule": worthwhile}
    practical = best if meaningful else "catboost"
    worthwhile_ensembles = [name for name, comparison in ensemble_comparisons.items()
                           if comparison["complexity_justified_by_gain_rule"]]
    if worthwhile_ensembles:
        practical = max(worthwhile_ensembles, key=lambda name: pivot.loc[name, "average_precision"])
    roc_candidate = max(result["oof_metrics"], key=lambda name: result["oof_metrics"][name]["roc_auc"])
    return {"best_single_by_mean_cv_ap": best, "improvement_over_catboost": improvement,
            "classification": classification, "ensemble_comparisons": ensemble_comparisons,
            "interview_take_home": practical, "production_oriented_simplicity": practical,
            "kaggle_style_if_roc_auc_is_objective": roc_candidate,
            "rule_note": "Gain rule is conservative/descriptive, not a significance test or a business ROI threshold. Fold scores are dependent. No independent test or leaderboard was used."}


def available_templates():
    templates = {"logistic": build_pipeline(), "catboost": build_catboost_pipeline()}
    blockers = {}
    for library in ["lightgbm", "xgboost"]:
        try:
            module = importlib.import_module(f"src.train_{library}")
            templates[library] = getattr(module, f"build_{library}_pipeline")()
        except (ImportError, OSError) as error:
            blockers[library] = str(error)
    return templates, blockers


def parameter_record(estimator):
    """Strict JSON metadata: preserve nonfinite library sentinels as named strings."""
    def convert(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    return convert(estimator.get_params())


def write_report(state):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "Controlled model-family benchmark: Logistic, CatBoost, LightGBM, XGBoost; no hyperparameter or weight search.",
        "Reproduce: .\\.venv\\Scripts\\python.exe -B -m src.benchmark_models",
        "Holdout: existing stratified 80/20 seed 42. CV: existing five shared stratified shuffled folds seed 42.",
        "New tree encoding: train-local one-hot, explicit float32 zeros, unknown categories ignored, numeric values retained.",
        "Ensembles are predefined equal boosting weights and CatBoost/LightGBM 50/50 (XGBoost fallback if LightGBM unavailable).",
        "Technical metrics use Average Precision (sklearn), ROC-AUC, log loss, Brier. AP is not trapezoidal PR-AUC.",
        "Top-K uses ceil(K*n), stable descending scores; capture=selected responders/all responders, response=responders/selected, lift=response/overall rate.",
        "Threshold 0.5 is diagnostic only. Campaign capacity, costs, revenue and ROI have not been assumed.",
    ]
    for key in ["environment", "parameters", "checkpoint", "holdout_rows", "holdout_topk", "cv_summary", "oof_correlations", "ensemble_rows", "decision", "artifacts", "verification", "warnings", "blockers", "elapsed_seconds", "status"]:
        if key in state:
            value = state[key]
            if key == "oof_correlations":
                value = pd.DataFrame(value).round(6).to_string()
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                table = pd.DataFrame(value)
                if key in ["holdout_rows", "ensemble_rows"]:
                    display = ["model", "evaluation", "roc_auc", "average_precision", "log_loss", "brier_score",
                               "top_10_lift", "top_10_responder_capture_rate", "fit_and_score_seconds",
                               "member_fit_and_score_seconds", "blend_and_score_seconds"]
                    table = table[[column for column in display if column in table]]
                value = table.to_string(index=False)
            else:
                value = json.dumps(value, indent=2, allow_nan=False)
            sections.append(f"\n{key.upper()}\n{value}")
    sections.append("\nHuman review: confirm scoring-time availability of Previously_Insured and Vehicle_Damage; Response definition/window; premium 2630 meaning; hidden customer/time structure. Importance is not causality. Reused holdout/CV are internal comparisons, not independent external performance estimates. Correlation measures probability similarity, not error independence. Fixed configurations do not establish each family's tuned ceiling. Conservative complexity rule is descriptive, not a formal significance test.")
    LOG_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main():
    started = time.perf_counter()
    METRICS.mkdir(parents=True, exist_ok=True)
    templates, blockers = available_templates()
    state = {"status": "RUNNING", "warnings": [], "blockers": blockers,
             "checkpoint": local_command(["git", "rev-parse", "--short", "HEAD"])["stdout"].strip(),
             "ensemble_definitions": ensemble_definitions(templates)}
    import catboost, sklearn
    state["environment"] = {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                            "sklearn": sklearn.__version__, "catboost": catboost.__version__}
    for name in ["lightgbm", "xgboost"]:
        if name in templates:
            state["environment"][name] = importlib.import_module(name).__version__
    state["parameters"] = {name: parameter_record(model.named_steps["classifier"]) for name, model in templates.items()}
    for name in ["lightgbm", "xgboost"]:
        if name in templates:
            write_json(METRICS / f"{name}_parameters.json", {"parameters": state["parameters"][name],
                        "early_stopping": False, "encoding": "fold-local dense float32 one-hot, unknown ignored",
                        "parameter_serialization": "Nonfinite sentinels are named strings; XGBoost missing=NaN is not changed in the estimator."})
    old_artifacts = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in METRICS.iterdir() if path.is_file() and not path.name.startswith("model_family")
                     and path.name not in ["lightgbm_parameters.json", "xgboost_parameters.json", "ensemble_results.csv", "oof_prediction_correlations.csv"]}
    raw_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / "data/raw").glob("*.csv")}
    write_report(state)
    try:
        def checkpoint(phase, label, holdout, rows):
            state["holdout_rows"] = [metric_row(name, metrics, "holdout") for name, metrics in holdout.items()]
            pd.DataFrame(state["holdout_rows"]).to_csv(METRICS / "model_family_holdout.csv", index=False)
            write_json(METRICS / "model_family_holdout.json", holdout)
            for name in ["logistic", "catboost"]:
                if phase == "holdout" and label == name:
                    previous = json.loads((METRICS / f"{name}_holdout.json").read_text(encoding="utf-8"))
                    if holdout[name]["validation_index_sha256"] != previous["validation_index_sha256"]:
                        raise AssertionError("Validated holdout rows changed")
                    if not all(np.isclose(holdout[name][key], previous[key], atol=1e-10, rtol=0)
                               for key in ["roc_auc", "average_precision", "log_loss", "brier_score"]):
                        raise AssertionError(f"Validated {name} results did not reproduce")
            if rows:
                pd.DataFrame(rows).to_csv(METRICS / "model_family_cv_results.csv", index=False)
                state["cv_summary"] = summarize_cv(rows)
                pd.DataFrame(state["cv_summary"]).to_csv(METRICS / "model_family_cv_summary.csv", index=False)
            state["elapsed_seconds"] = time.perf_counter() - started
            write_report(state)
            if phase in ["holdout", "cv"]:
                current = holdout[label] if phase == "holdout" else rows[-1]
                print(f"{label}: AUC={current['roc_auc']:.6f}, AP={current['average_precision']:.6f}, "
                      f"seconds={current['fit_and_score_seconds']:.1f}", flush=True)

        result = run_from_training_data(templates, checkpoint)
        boost = [name for name in BOOSTING_MODELS if name in templates]
        correlations = pd.DataFrame({name: result["oof_predictions"][name] for name in boost}).corr(method="pearson")
        correlations.to_csv(METRICS / "oof_prediction_correlations.csv", index_label="model")
        state["oof_correlations"] = correlations.to_dict()
        state["ensemble_rows"] = [metric_row(name, result[source][name], evaluation)
                                  for name in result["ensemble_definitions"]
                                  for source, evaluation in [("holdout", "holdout"), ("oof_metrics", "pooled_oof")]]
        pd.DataFrame(state["ensemble_rows"]).to_csv(METRICS / "ensemble_results.csv", index=False)
        pd.DataFrame([metric_row(name, metrics, "pooled_oof") for name, metrics in result["oof_metrics"].items()]).to_csv(METRICS / "model_family_oof_metrics.csv", index=False)
        topk = [{"model": name, "evaluation": evaluation, "segment": segment, **values}
                for source, evaluation in [("holdout", "holdout"), ("oof_metrics", "pooled_oof")]
                for name, metrics in result[source].items() for segment, values in metrics["ranking"].items()]
        pd.DataFrame(topk).to_csv(METRICS / "model_family_topk.csv", index=False)
        state["holdout_topk"] = [row for row in topk if row["evaluation"] == "holdout"]
        prediction_bundle = {f"oof_{name}": scores.to_numpy() for name, scores in result["oof_predictions"].items()}
        prediction_bundle.update({f"holdout_{name}": scores.to_numpy() for name, scores in result["holdout_predictions"].items()})
        prediction_bundle.update({"oof_row_indices": result["labels"].index.to_numpy(), "oof_labels": result["labels"].to_numpy(),
                                  "fold_id": result["fold_ids"], "holdout_row_indices": next(iter(result["holdout_predictions"].values())).index.to_numpy()})
        np.savez_compressed(METRICS / "model_family_predictions.npz", **prediction_bundle)
        state["decision"] = comparative_decision(result, templates)
        write_json(METRICS / "model_family_decision.json", state["decision"])
        state["warnings"] = [warning for metrics in result["holdout"].values() for warning in metrics.get("warnings", [])]
        state["warnings"] += [row["warnings"] for row in result["cv_results"] if row["warnings"]]
        state["verification"] = {
            "old_artifacts_unchanged": all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in old_artifacts.items()),
            "raw_csvs_unchanged": all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in raw_hashes.items()),
            "oof_all_rows_once": bool((result["fold_ids"] > 0).all()),
            "oof_finite": all(np.isfinite(scores).all() for scores in result["oof_predictions"].values()),
            "same_fold_indices": all(group["validation_index_sha256"].nunique() == 1 and group["training_index_sha256"].nunique() == 1
                                     for _, group in pd.DataFrame(result["cv_results"]).groupby("fold")),
        }
        if not all(state["verification"].values()):
            raise AssertionError("Artifact/fold integrity check failed")
        state["verification"]["tests"] = local_command([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"])
        state["verification"]["git_diff_check"] = local_command(["git", "diff", "--check"])
        state["status"] = "COMPLETE" if not any(state["verification"][key]["returncode"] for key in ["tests", "git_diff_check"]) else "VERIFICATION_FAILED"
    except Exception:
        state["status"] = "BLOCKED"
        state["blockers"]["runtime"] = traceback.format_exc()
    state["elapsed_seconds"] = time.perf_counter() - started
    state["artifacts"] = [str(path.relative_to(ROOT)).replace("\\", "/")
                          for path in sorted(METRICS.iterdir()) if path.is_file()
                          and (path.name.startswith("model_family") or path.name in
                               ["ensemble_results.csv", "oof_prediction_correlations.csv", "lightgbm_parameters.json", "xgboost_parameters.json"])]
    if "outputs/metrics/model_family_manifest.json" not in state["artifacts"]:
        state["artifacts"].append("outputs/metrics/model_family_manifest.json")
    # Normalize numpy scalar checks for JSON while preserving precise metrics.
    for key, value in state.get("verification", {}).items():
        if isinstance(value, np.generic):
            state["verification"][key] = value.item()
    write_json(METRICS / "model_family_manifest.json", state)
    state.setdefault("verification", {})["git_status"] = local_command(["git", "status", "--short", "--untracked-files=all"])
    write_json(METRICS / "model_family_manifest.json", state)
    write_report(state)
    print(f"{state['status']}: {LOG_PATH}; elapsed={state['elapsed_seconds']:.1f}s", flush=True)
    if state["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
