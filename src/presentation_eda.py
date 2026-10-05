r"""Build a small, reproducible presentation figure set without fitting models.

Run: .\.venv\Scripts\python.exe -B -m src.presentation_eda
Sources: train.csv, validated CV/holdout metrics and saved holdout scores.
Test-set channel metadata is reused from the already-known audit; no test CSV
is loaded. Matplotlib configuration/cache stays inside this repository.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "figures"
METRICS = ROOT / "outputs" / "metrics"
os.environ["MPLCONFIGDIR"] = str(FIGURES / "matplotlib_config")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import TRAIN_PATH, load_train
from .evaluate import top_k_metrics


NAVY = "#193B56"
TEAL = "#007F7C"
SLATE = "#8193A5"
GRAY = "#5A6975"
GRID = "#E6ECEF"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "text.color": NAVY, "axes.labelcolor": NAVY,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def cumulative_gains(labels, scores):
    """Full ranked curve, including origin; stable ties match validated Top-K logic."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if (labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores)
            or not len(labels) or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or labels.sum() == 0):
        raise ValueError("Gains require aligned binary labels, valid scores and at least one responder")
    order = np.argsort(-scores, kind="stable")
    captured = np.r_[0, np.cumsum(labels[order])].astype(int)
    return {"selected_count": np.arange(len(labels) + 1),
            "responders_captured": captured,
            "capacity": np.arange(len(labels) + 1) / len(labels),
            "capture_rate": captured / int(labels.sum())}


def response_table(frame, column, values):
    rows = []
    for value in values:
        labels = frame.loc[frame[column] == value, "Response"]
        if labels.empty:
            raise ValueError(f"Empty requested group: {column}={value}")
        rows.append({"value": value, "rows": len(labels), "responders": int(labels.sum()),
                     "response_rate": float(labels.mean())})
    return rows


def canvas(title, subtitle):
    figure = plt.figure(figsize=(12, 6.75))
    figure.text(.065, .93, title, fontsize=20, fontweight="bold", va="top")
    figure.text(.065, .87, subtitle, fontsize=11, color=GRAY, va="top")
    return figure


def footer(figure, text):
    figure.text(.065, .065, text, fontsize=10.5, color=GRAY, va="bottom", linespacing=1.5)


def percent_axis(axis, direction="y"):
    getattr(axis, f"{direction}axis").set_major_formatter(PercentFormatter(1, decimals=0))
    axis.grid(axis=direction, color=GRID, linewidth=.8)
    axis.set_axisbelow(True)


def save(figure, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / filename, dpi=200)
    plt.close(figure)


def draw_rate_bars(axis, rows, labels, colors, overall, limit):
    positions = np.arange(len(rows))
    rates = [row["response_rate"] for row in rows]
    axis.bar(positions, rates, color=colors, width=.6)
    axis.axhline(overall, color=GRAY, linestyle="--", linewidth=1)
    for position, rate in zip(positions, rates):
        axis.annotate(f"{rate:.2%}", (position, rate), xytext=(0, 7), textcoords="offset points",
                      ha="center", fontsize=13, fontweight="bold")
    axis.set_xticks(positions, [f"{label}\nn={row['rows']:,}" for label, row in zip(labels, rows)])
    axis.set_ylim(0, limit)
    percent_axis(axis)


def main():
    frame = load_train()
    train_hash = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
    original_artifacts = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in METRICS.iterdir() if path.is_file()
                          and not path.name.startswith(("presentation_", "policy_channel_summary"))}
    overall = float(frame["Response"].mean())
    sources = {"command": r".\.venv\Scripts\python.exe -B -m src.presentation_eda",
               "train_rows": len(frame), "train_sha256": train_hash,
               "training_response_rate": overall, "figures": {}, "checks": {}}

    # 1: imbalance and the derived, trivial all-negative accuracy reference.
    counts = frame["Response"].value_counts().sort_index()
    target_rows = [{"response": int(label), "count": int(count), "percentage": count / len(frame) * 100}
                   for label, count in counts.items()]
    figure = canvas(f"Only {overall:.2%} of training rows have a positive response",
                    f"Target: Response | {len(frame):,} labeled training rows")
    axis = figure.subplots()
    figure.subplots_adjust(left=.19, right=.95, top=.78, bottom=.28)
    shares = counts.to_numpy() / len(frame)
    axis.barh([1, 0], shares, height=.45, color=[SLATE, TEAL])
    for position, share, count in zip([1, 0], shares, counts):
        inside = share > .5
        axis.text(share - .02 if inside else share + .02, position,
                  f"{int(count):,} rows  |  {share:.2%}", va="center",
                  ha="right" if inside else "left", color="white" if inside else NAVY,
                  fontsize=15, fontweight="bold")
    axis.set_yticks([1, 0], ["Response = 0", "Response = 1"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of training rows")
    percent_axis(axis, "x")
    figure.text(.065, .16, f"Predicting 0 for everyone gives {shares[0]:.2%} accuracy and 0% responder recall.",
                fontsize=14, fontweight="bold")
    footer(figure, "Source: original train.csv. Class imbalance makes accuracy alone a weak success criterion.")
    filename = "eda_target_distribution.png"
    save(figure, filename)
    sources["figures"][filename] = {"source": "data/raw/train.csv", "target_distribution": target_rows,
                                   "all_negative_accuracy": float(shares[0]), "message": "Evaluate ranking and responder capture, not accuracy alone."}

    # 2: unadjusted associations, with explicit availability flags.
    segments = {
        "Previously_Insured": response_table(frame, "Previously_Insured", [0, 1]),
        "Vehicle_Damage": response_table(frame, "Vehicle_Damage", ["No", "Yes"]),
        "Vehicle_Age": response_table(frame, "Vehicle_Age", ["< 1 Year", "1-2 Year", "> 2 Years"]),
    }
    figure = canvas("Response rates differ across observed customer segments",
                    f"Unadjusted training associations; the dashed line is the overall {overall:.2%} response rate")
    axes = figure.subplots(1, 3, sharey=True)
    figure.subplots_adjust(left=.085, right=.95, top=.76, bottom=.25, wspace=.23)
    draw_rate_bars(axes[0], segments["Previously_Insured"], ["0", "1"], [TEAL, SLATE], overall, .36)
    draw_rate_bars(axes[1], segments["Vehicle_Damage"], ["No", "Yes"], [SLATE, TEAL], overall, .36)
    draw_rate_bars(axes[2], segments["Vehicle_Age"], ["<1 year", "1–2 years", ">2 years"], [SLATE, SLATE, TEAL], overall, .36)
    for axis, title in zip(axes, ["Previously insured*", "Vehicle damage*", "Vehicle age"]):
        axis.set_title(title, pad=16)
    axes[0].set_ylabel("Observed response rate")
    footer(figure, "*Confirm Previously_Insured and Vehicle_Damage are available at campaign scoring time.\nThese are descriptive associations, not causal effects; other features may explain group differences.")
    filename = "eda_key_segments.png"
    save(figure, filename)
    sources["figures"][filename] = {"source": "data/raw/train.csv", "segments": segments,
                                   "message": "Strong associations warrant scoring-time availability checks, not automatic leakage claims."}

    # 3: interpretable decade bins, including the full observed upper age range.
    age_labels = ["20–29", "30–39", "40–49", "50–59", "60–69", "70–85"]
    age_frame = frame.assign(age_bin=pd.cut(frame["Age"], [20, 30, 40, 50, 60, 70, 86], right=False, labels=age_labels))
    ages = response_table(age_frame, "age_bin", age_labels)
    if sum(row["rows"] for row in ages) != len(frame):
        raise AssertionError("Age bins do not cover all training rows")
    figure = canvas("The descriptive age-response pattern is nonlinear",
                    "Response rate rises in middle age bins, then falls in older bins")
    axis = figure.subplots()
    figure.subplots_adjust(left=.095, right=.95, top=.78, bottom=.25)
    axis.plot(range(6), [row["response_rate"] for row in ages], color=TEAL, marker="o", linewidth=2.5, markersize=8)
    axis.axhline(overall, color=GRAY, linestyle="--", linewidth=1, label=f"Overall: {overall:.2%}")
    for position, row in enumerate(ages):
        axis.annotate(f"{row['response_rate']:.2%}", (position, row["response_rate"]),
                      xytext=(0, 12), textcoords="offset points", ha="center", fontweight="bold")
    axis.set_xticks(range(6), [f"{row['value']}\nn={row['rows']:,}" for row in ages])
    axis.set_ylim(0, .26)
    axis.set_ylabel("Observed response rate")
    axis.set_xlabel("Age bin")
    percent_axis(axis)
    axis.legend(frameon=False, loc="lower right")
    footer(figure, "Unadjusted rates motivate testing nonlinear relationships, not a causal claim about age.\nThe pattern may also reflect correlated features; it does not isolate a conditional age effect.")
    filename = "eda_age_response.png"
    save(figure, filename)
    sources["figures"][filename] = {"source": "data/raw/train.csv", "bins": ages,
                                   "message": "A nonlinear challenger is a defensible hypothesis; this is not proof of mechanism."}

    # 4: training concentration, with previously known test metadata only.
    frequencies = frame["Policy_Sales_Channel"].value_counts().sort_values(ascending=False)
    rare = frequencies[frequencies < 100]
    policy = {"train_categories": len(frequencies), "test_categories": 145,
              "test_only_unseen_codes": [141.0, 142.0],
              "test_metadata_source": "Already-known audit metadata explicitly supplied in this task; test.csv not loaded.",
              "top_three_codes": [{"code": float(code), "rows": int(count)} for code, count in frequencies.iloc[:3].items()],
              "top_three_row_share": float(frequencies.iloc[:3].sum() / len(frame)),
              "rare_definition": "Fewer than 100 training rows", "rare_categories": len(rare),
              "rare_rows": int(rare.sum()), "rare_row_share": float(rare.sum() / len(frame)),
              "interpretation": "Sparse unseen-code cases are an inference-robustness edge case, not evidence of major distribution shift.",
              "mitigation": {"logistic": 'OneHotEncoder(handle_unknown="ignore")',
                             "catboost": "Consistent categorical string representation; unseen categories supported",
                             "lightgbm_xgboost": "Training-local one-hot encoding, handle_unknown=ignore",
                             "production": "Monitor unseen-category rate and input schema"}}
    if policy["train_categories"] != 155:
        raise AssertionError("Unexpected training channel cardinality")
    figure = canvas("Sales-channel codes are concentrated and long-tailed",
                    f"{len(frequencies)} training categories | Top 3 cover {policy['top_three_row_share']:.2%} of rows | {len(rare)} codes have fewer than 100 rows")
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.10, right=.95, top=.75, bottom=.25, wspace=.30)
    ranks = np.arange(1, len(frequencies) + 1)
    axes[0].plot(ranks, frequencies.to_numpy(), color=NAVY, linewidth=2)
    axes[0].set_yscale("log")
    axes[0].set_ylim(.7, frequencies.max() * 1.4)
    axes[0].set_yticks([1, 10, 100, 1000, 10000, 100000])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[0].axhline(100, color=GRAY, linestyle="--", linewidth=1)
    axes[0].axvspan(len(frequencies) - len(rare) + .5, len(frequencies), color="#F4EBDC", alpha=.8)
    axes[0].set(title="Ranked frequency", xlabel="Category rank", ylabel="Training rows (log scale)")
    axes[0].grid(axis="y", color=GRID)
    axes[1].plot(np.r_[0, ranks], np.r_[0, frequencies.cumsum().to_numpy() / len(frame)], color=TEAL, linewidth=2.5)
    axes[1].scatter([3], [policy["top_three_row_share"]], color=TEAL, s=55, zorder=3)
    axes[1].annotate(f"Top 3: {policy['top_three_row_share']:.2%}", (3, policy["top_three_row_share"]),
                     xytext=(48, .56), arrowprops={"arrowstyle": "-", "color": GRAY}, fontsize=13, fontweight="bold")
    axes[1].set(title="Cumulative row share", xlabel="Number of largest categories", ylim=(0, 1.04))
    percent_axis(axes[1])
    footer(figure, "Known inference edge: test-only codes 141.0 and 142.0. Use unknown-safe encoders and monitor unseen-category rate.\nTwo unseen codes alone do not establish major distribution shift; shaded ranks have fewer than 100 training rows.")
    filename = "eda_policy_channel_long_tail.png"
    save(figure, filename)
    (METRICS / "policy_channel_summary.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    sources["figures"][filename] = {"source": "data/raw/train.csv; already-known inference audit", "summary": policy,
                                   "message": "Concentration and rare codes call for robust category handling, not numeric code-distance assumptions."}

    # 5: full training distribution; an explicit overflow bar retains the tail.
    premiums = frame["Annual_Premium"].to_numpy()
    premium_frame = frame.assign(premium_group=np.where(premiums == 2630, "2630", "Other values"))
    premium_groups = response_table(premium_frame, "premium_group", ["2630", "Other values"])
    edges = np.r_[np.arange(0, 80001, 5000), np.inf]
    histogram, _ = np.histogram(premiums, bins=edges)
    figure = canvas("Premium 2,630 is a large, unexplained mass point",
                    f"{premium_groups[0]['rows']:,} rows ({premium_groups[0]['rows']/len(frame):.2%}) have exactly Annual_Premium = 2630")
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.095, right=.95, top=.75, bottom=.25, wspace=.30)
    centers = np.arange(len(histogram)) * 5000 + 2500
    axes[0].bar(centers, histogram, width=4500, color=[TEAL] + [SLATE] * (len(histogram) - 1))
    axes[0].set_xticks([0, 20000, 40000, 60000, 82500], ["0", "20,000", "40,000", "60,000", "80,000+"])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[0].set(xlabel="Annual premium (source units)", ylabel="Training rows", ylim=(0, histogram.max() * 1.4))
    axes[0].annotate(f"{premium_groups[0]['rows']:,} exactly 2,630", (2500, histogram[0]),
                     xytext=(16000, histogram.max() * 1.18), arrowprops={"arrowstyle": "-", "color": GRAY}, fontweight="bold")
    axes[0].grid(axis="y", color=GRID)
    axes[0].set_axisbelow(True)
    draw_rate_bars(axes[1], premium_groups, ["Exactly 2,630", "Other values"], [TEAL, SLATE], overall, .18)
    axes[1].set_title("Observed response rate", pad=16)
    footer(figure, "The last histogram bar combines all values ≥80,000; no values were removed. Dashed line: overall response rate.\nSource files do not define whether 2,630 is a real premium, floor or code. Retain it without arbitrary recoding.")
    filename = "eda_annual_premium.png"
    save(figure, filename)
    sources["figures"][filename] = {"source": "data/raw/train.csv", "groups": premium_groups,
                                   "mass_point_share": premium_groups[0]["rows"] / len(frame),
                                   "histogram_counts": histogram.tolist(), "histogram_edges": list(range(0, 80001, 5000)) + ["infinity"],
                                   "overflow_count_ge_80000": int(histogram[-1]), "maximum_retained": float(premiums.max()),
                                   "message": "Retain an unexplained mass point; displaying the tail separately is not deleting it."}

    # 6: full score axes avoid exaggerating the small differences among boosters.
    cv = pd.read_csv(METRICS / "model_family_cv_summary.csv", float_precision="round_trip")
    model_names = ["logistic", "catboost", "lightgbm", "xgboost"]
    model_labels = ["Logistic Regression", "CatBoost", "LightGBM", "XGBoost"]
    cv_rows = cv[cv["model"].isin(model_names) & cv["metric"].isin(["roc_auc", "average_precision"])]
    figure = canvas("CatBoost has the highest mean scores among single models",
                    "Fixed configurations and identical five folds; points show mean ± sample standard deviation")
    axes = figure.subplots(1, 2, sharey=True)
    figure.subplots_adjust(left=.19, right=.96, top=.76, bottom=.23, wspace=.28)
    for axis, metric, title in zip(axes, ["roc_auc", "average_precision"], ["ROC-AUC", "Average Precision (AP)"]):
        for position, model in enumerate(model_names):
            row = cv_rows[(cv_rows["model"] == model) & (cv_rows["metric"] == metric)].iloc[0]
            color = TEAL if model == "catboost" else SLATE
            axis.errorbar(row["mean"], position, xerr=row["std"], fmt="o", color=color, markersize=8, capsize=4, linewidth=2)
            axis.text(.04 if metric == "roc_auc" else .53, position,
                      f"{row['mean']:.4f} ± {row['std']:.4f}", va="center", fontsize=11.5, color=color,
                      fontweight="bold" if model == "catboost" else "normal")
        axis.set(xlim=(0, 1), title=title, xlabel="Score (0–1)")
        axis.set_yticks(range(4), model_labels)
        axis.grid(axis="x", color=GRID)
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    footer(figure, "Full 0–1 axes keep small differences in perspective. Fold SD is descriptive dispersion, not a confidence interval.\nBoosted-model SD ranges overlap; this comparison does not establish each family's tuned ceiling.")
    filename = "model_family_cv_comparison.png"
    save(figure, filename)
    sources["figures"][filename] = {"source": "outputs/metrics/model_family_cv_summary.csv", "values": cv_rows.to_dict(orient="records"),
                                   "message": "CatBoost is the strongest single model in this fixed comparison, with small gaps among boosting families."}

    # 7 and 8: original validated holdout scores and original row order only.
    holdout = json.loads((METRICS / "catboost_holdout.json").read_text(encoding="utf-8"))
    with np.load(METRICS / "model_family_predictions.npz", allow_pickle=False) as archive:
        indices = archive["holdout_row_indices"]
        scores = archive["holdout_catboost"]
    digest = hashlib.sha256(np.asarray(indices, dtype="int64").tobytes()).hexdigest()
    if digest != holdout["validation_index_sha256"] or not pd.Index(indices).is_unique:
        raise AssertionError("Saved holdout index integrity failed")
    labels = frame.loc[indices, "Response"].to_numpy()
    ranked = top_k_metrics(labels, scores)
    for segment, metrics in ranked.items():
        for key, value in metrics.items():
            if not np.isclose(value, holdout["ranking"][segment][key], atol=1e-12, rtol=0):
                raise AssertionError(f"Saved Top-K result changed: {segment}/{key}")
    if not (np.isclose(roc_auc_score(labels, scores), holdout["roc_auc"], atol=1e-12, rtol=0)
            and np.isclose(average_precision_score(labels, scores), holdout["average_precision"], atol=1e-12, rtol=0)):
        raise AssertionError("Saved holdout scores do not reproduce validated metrics")
    source_holdout = {"validation_rows": len(labels), "responders": int(labels.sum()),
                      "overall_response_rate": float(labels.mean()), "roc_auc": holdout["roc_auc"],
                      "average_precision": holdout["average_precision"], "ranking": ranked,
                      "validation_index_sha256": digest}
    figure = canvas("Ranked targeting concentrates validation responders",
                    f"CatBoost holdout | {len(labels):,} rows | Overall response rate {labels.mean():.2%} | Top 10% highlighted")
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=.08, right=.96, top=.76, bottom=.27, wspace=.35)
    for axis, key, title, limit in zip(axes, ["responder_capture_rate", "response_rate", "lift"],
                                     ["Responder capture", "Segment response rate", "Lift vs overall"], [.7, .5, 4]):
        values = [ranked[f"top_{k}_pct"][key] for k in [5, 10, 20]]
        axis.bar(range(3), values, color=[SLATE, TEAL, SLATE], width=.6)
        for position, value in enumerate(values):
            axis.annotate(f"{value:.2f}x" if key == "lift" else f"{value:.2%}", (position, value),
                          xytext=(0, 7), textcoords="offset points", ha="center", fontweight="bold", fontsize=13)
        axis.set_xticks(range(3), ["Top 5%", "Top 10%", "Top 20%"])
        axis.get_xticklabels()[1].set_fontweight("bold")
        axis.set(title=title, ylim=(0, limit))
        if key != "lift":
            percent_axis(axis)
        else:
            axis.set_yticks([0, 1, 2, 3, 4])
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}x"))
            axis.grid(axis="y", color=GRID)
            axis.set_axisbelow(True)
        if key in ["response_rate", "lift"]:
            axis.axhline(labels.mean() if key == "response_rate" else 1, color=GRAY, linestyle="--", linewidth=1)
    top10 = ranked["top_10_pct"]
    figure.text(.065, .16, f"Top 10%: {top10['selected_count']:,} selected rows → {top10['responders_captured']:,} responders",
                fontsize=15, fontweight="bold", color=TEAL)
    footer(figure, "Capacity-based ranking; threshold 0.5 is not the operational rule. Dashed lines show overall rate and 1x lift.\nCapture = selected responders / all responders; segment rate = responders / selected; lift = segment rate / overall rate.")
    filename = "catboost_topk_targeting.png"
    save(figure, filename)
    sources["figures"][filename] = {"source": "validated CatBoost holdout metrics and saved scores", **source_holdout,
                                   "message": "Capacity-based prioritization concentrates responders without assuming monetary benefit."}

    curve = cumulative_gains(labels, scores)
    figure = canvas("How responder capture grows with targeting capacity",
                    "Validated CatBoost holdout ranking compared with the expected random-targeting reference")
    axis = figure.subplots()
    figure.subplots_adjust(left=.095, right=.95, top=.78, bottom=.23)
    axis.plot(curve["capacity"], curve["capture_rate"], color=TEAL, linewidth=2.5, label="CatBoost ranking")
    axis.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1.5, label="Random targeting (expected)")
    count = top10["selected_count"]
    axis.scatter([curve["capacity"][count]], [curve["capture_rate"][count]], color=TEAL, s=65, zorder=3)
    axis.annotate(f"Top 10% captures {top10['responder_capture_rate']:.2%}\n{top10['responders_captured']:,} of {int(labels.sum()):,} responders",
                  (curve["capacity"][count], curve["capture_rate"][count]), xytext=(.33, .19),
                  arrowprops={"arrowstyle": "-", "color": GRAY}, fontweight="bold", fontsize=13,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 3})
    axis.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Share of validation rows selected", ylabel="Share of validation responders captured")
    percent_axis(axis, "x")
    percent_axis(axis)
    axis.legend(frameon=False, loc="lower right")
    footer(figure, "Campaign capacity is a choice, not a fixed 0.5 probability cutoff. No campaign economics are assumed.\nStable score ties follow the validated input order; the random diagonal is a mathematical reference, not a fitted model.")
    filename = "catboost_cumulative_gains.png"
    save(figure, filename)
    sample_counts = np.ceil(np.arange(101) / 100 * len(labels)).astype(int)
    points = pd.DataFrame({"target_capacity_fraction": np.arange(101) / 100,
                           "selected_count": sample_counts,
                           "actual_capacity_fraction": curve["capacity"][sample_counts],
                           "responders_captured": curve["responders_captured"][sample_counts],
                           "capture_rate": curve["capture_rate"][sample_counts]})
    points.to_csv(METRICS / "presentation_gains_points.csv", index=False)
    sources["figures"][filename] = {"source": "saved CatBoost holdout scores; Response from original aligned train rows", **source_holdout,
                                   "curve_points_source": "outputs/metrics/presentation_gains_points.csv",
                                   "random_reference": "Expected capture fraction equals the selected row fraction",
                                   "message": "The gains curve shows how capture changes as campaign capacity expands."}

    # Optional native importance: useful for the appendix, not a causal narrative.
    importance = pd.read_csv(METRICS / "catboost_feature_importance.csv", float_precision="round_trip").head(5)
    figure = canvas("Native predictive importance is not causal influence",
                    "Top five CatBoost features from the validated holdout-trained model")
    axis = figure.subplots()
    figure.subplots_adjust(left=.27, right=.94, top=.78, bottom=.23)
    axis.barh(range(5), importance["importance"], color=[TEAL, TEAL, SLATE, SLATE, SLATE], height=.6)
    axis.set_yticks(range(5), ["Previously insured*", "Vehicle damage*", "Age", "Sales channel", "Vehicle age"])
    axis.invert_yaxis()
    for position, value in enumerate(importance["importance"]):
        axis.text(value + .8, position, f"{value:.2f}", va="center", fontweight="bold")
    axis.set_xlim(0, 65)
    axis.set_xlabel("Native importance (normalized total across all features = 100)")
    axis.grid(axis="x", color=GRID)
    axis.set_axisbelow(True)
    footer(figure, "Predictive importance ≠ causality. *Confirm availability at campaign scoring time.\nPredictionValuesChange importance; correlated predictors can share importance. This is not an ablation or causal estimate.")
    filename = "catboost_feature_importance.png"
    save(figure, filename)
    sources["figures"][filename] = {"source": "outputs/metrics/catboost_feature_importance.csv", "top_five": importance.to_dict(orient="records"),
                                   "message": "Native importance guides explanation, with availability flags and no causal claim."}

    sources["recommended_main"] = ["eda_target_distribution.png", "eda_key_segments.png", "model_family_cv_comparison.png", "catboost_topk_targeting.png", "catboost_cumulative_gains.png"]
    sources["recommended_appendix"] = ["eda_age_response.png", "eda_policy_channel_long_tail.png", "eda_annual_premium.png", "catboost_feature_importance.png", "calibration.png"]
    sources["checks"] = {
        "all_nine_figures_exist": all((FIGURES / name).is_file() for name in sources["figures"]),
        "train_bytes_unchanged": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest() == train_hash,
        "validated_artifacts_unchanged": all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest for path, digest in original_artifacts.items()),
        "target_count_matches": int(counts.sum()) == 381109 and int(counts.loc[1]) == 46710,
        "premium_histogram_retains_every_row": int(histogram.sum()) == len(frame),
        "first_histogram_bin_is_mass_point_only": int(histogram[0]) == premium_groups[0]["rows"],
        "holdout_metrics_and_topk_reproduced": True,
        "age_bins_cover_every_row": sum(row["rows"] for row in ages) == len(frame),
        "segment_groups_cover_every_row": all(sum(row["rows"] for row in rows) == len(frame) for rows in segments.values()),
        "policy_channel_cardinality_matches": len(frequencies) == 155,
        "catboost_highest_cv_means": all(float(cv_rows[(cv_rows["model"] == "catboost") & (cv_rows["metric"] == metric)]["mean"].iloc[0])
                                        == float(cv_rows[cv_rows["metric"] == metric]["mean"].max())
                                        for metric in ["roc_auc", "average_precision"]),
    }
    (METRICS / "presentation_figure_sources.json").write_text(json.dumps(sources, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"figures": list(sources["figures"]), "checks": sources["checks"], "source_manifest": "outputs/metrics/presentation_figure_sources.json"}, indent=2))
    if not all(sources["checks"].values()):
        raise SystemExit("Figure source verification failed")


if __name__ == "__main__":
    main()
