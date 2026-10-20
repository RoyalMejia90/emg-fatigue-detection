"""
Label Analisis

Author: RT, NM, NI
"""

import os

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, wilcoxon

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score


# Configuration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "RESULTS")
DATASET_FILE = os.path.join(RESULTS_DIR, "fatigue_dataset.csv")

RANDOM_STATE = 42
SET_KEYS = ["subject_id", "file"]


# Helpers

def loso_logistic(df, label_col, features=("rms", "mdf")):
    """LOSO evaluation of the Baseline 2 pipeline. Scaling is fitted on
    training subjects only (inside the Pipeline)."""
    rows = []
    features = list(features)
    for train_idx, test_idx in LeaveOneGroupOut().split(
            df, df[label_col], df["subject_id"]):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000,
                                         random_state=RANDOM_STATE)),
        ])
        model.fit(train[features], train[label_col])
        pred = model.predict(test[features])
        prob = model.predict_proba(test[features])[:, 1]
        y = test[label_col]
        rows.append({
            "subject_id": test["subject_id"].iloc[0],
            "f1": f1_score(y, pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y, prob) if y.nunique() > 1 else np.nan,
            "predicted_fatigued_rate": pred.mean(),
            "true_fatigued_rate": y.mean(),
        })
    return pd.DataFrame(rows)


def shifted_labels(group, k):
    """Move the non-fatigued/fatigued boundary of one set by k
    repetitions (k < 0: earlier, k > 0: later). At least one repetition
    is kept in each class."""
    group = group.sort_values("repetition")
    n = len(group)
    first_fatigued = int(np.argmax(group["fatigue_label"].values == 1))
    boundary = min(max(first_fatigued + k, 1), n - 1)
    labels = np.zeros(n, dtype=int)
    labels[boundary:] = 1
    return pd.Series(labels, index=group.index)


# Load

df = pd.read_csv(DATASET_FILE)
df = df.sort_values(SET_KEYS + ["repetition"]).reset_index(drop=True)
df["series"] = df["file"].str.extract(r"_(R\d)_")[0]

n_per_set = df.groupby(SET_KEYS)["repetition"].transform("count")
df["reps_in_reserve"] = n_per_set - df.groupby(SET_KEYS).cumcount() - 1


# 1. Class distribution (Section 3.2)

print("\n=== Class distribution ===")
counts = df["fatigue_label"].value_counts().sort_index()
for cls, n in counts.items():
    print(f"Class {cls}: {n} windows ({100 * n / len(df):.2f} %)")
print(f"Total: {len(df)} windows, {df.groupby(SET_KEYS).ngroups} sets, "
      f"{df['subject_id'].nunique()} subjects")
print(f"Imbalance ratio (minority/majority): {counts.min() / counts.max():.2f}")


# 2. Label sensitivity (Section 3.4)

schemes = {"primary_50pct": df["fatigue_label"]}
for thr in (30, 40, 60, 70):
    schemes[f"threshold_{thr}pct"] = (df["progress_percent"] >= thr).astype(int)
for k in (-2, -1, 1, 2):
    schemes[f"shift_{k:+d}_rep"] = df.groupby(
        SET_KEYS, group_keys=False).apply(lambda g: shifted_labels(g, k))
for r in (3, 4, 5):
    schemes[f"rir_le_{r}"] = (df["reps_in_reserve"] <= r).astype(int)

rows = []
for name, labels in schemes.items():
    d = df.copy()
    d["label_alt"] = labels.astype(int)
    folds = loso_logistic(d, "label_alt")
    rows.append({
        "scheme": name,
        "pct_fatigued": 100 * d["label_alt"].mean(),
        "agreement_pct": 100 * (d["label_alt"] == df["fatigue_label"]).mean(),
        "kappa_vs_primary": cohen_kappa_score(df["fatigue_label"], d["label_alt"]),
        "f1_mean": folds["f1"].mean(),
        "f1_std": folds["f1"].std(),
        "auc_mean": folds["roc_auc"].mean(),
        "auc_std": folds["roc_auc"].std(),
    })

sens = pd.DataFrame(rows)
sens.to_csv(os.path.join(RESULTS_DIR, "label_sensitivity.csv"), index=False)
print("\n=== Label sensitivity (Logistic RMS + MDF, LOSO) ===")
print(sens.round(3).to_string(index=False))


# 3. Interpretation (Section 7.1)

trend_rows = []
for feature in ["rms", "mav", "variance", "waveform_length",
                "mdf", "mnf", "peak_frequency", "samples"]:
    rhos, changes = [], []
    for _, g in df.groupby(SET_KEYS):
        g = g.sort_values("repetition")
        if len(g) >= 4:
            rho = spearmanr(g["progress_percent"], g[feature])[0]
            if not np.isnan(rho):
                rhos.append(rho)
        third = max(1, len(g) // 3)
        first = g[feature].iloc[:third].mean()
        last = g[feature].iloc[-third:].mean()
        if first != 0:
            changes.append(100 * (last / first - 1))
    rhos = np.array(rhos)
    trend_rows.append({
        "feature": feature,
        "median_rho": np.median(rhos),
        "q1_rho": np.percentile(rhos, 25),
        "q3_rho": np.percentile(rhos, 75),
        "pct_sets_increasing": 100 * np.mean(rhos > 0),
        "pct_sets_decreasing": 100 * np.mean(rhos < 0),
        "wilcoxon_p": wilcoxon(rhos).pvalue,
        "median_change_last_vs_first_third_pct": np.median(changes),
        "n_sets": len(rhos),
    })

trends = pd.DataFrame(trend_rows)
trends.to_csv(os.path.join(RESULTS_DIR, "within_set_trends.csv"), index=False)
print("\n=== Within-set trends (Spearman rho vs set progression) ===")
print(trends.round(3).to_string(index=False))

print("\n=== Best model: predicted vs true fatigued rate per held-out subject ===")
best = loso_logistic(df, "fatigue_label")
print(best.round(3).to_string(index=False))
print(f"Macro-F1 {best['f1'].mean():.3f} +/- {best['f1'].std():.3f}; "
      f"ROC-AUC {best['roc_auc'].mean():.3f} +/- {best['roc_auc'].std():.3f}")

print("\n=== Control: repetition duration only ===")
dur = loso_logistic(df, "fatigue_label", ["samples"])
print(f"Macro-F1 {dur['f1'].mean():.3f} +/- {dur['f1'].std():.3f}; "
      f"ROC-AUC {dur['roc_auc'].mean():.3f}")

print("\n=== Control: first and last segment of every set removed ===")
inner = df[(df["repetition"] > 1) & (df["reps_in_reserve"] > 0)].reset_index(drop=True)
inn = loso_logistic(inner, "fatigue_label")
print(f"Macro-F1 {inn['f1'].mean():.3f} +/- {inn['f1'].std():.3f}; "
      f"ROC-AUC {inn['roc_auc'].mean():.3f} +/- {inn['roc_auc'].std():.3f}")

print("\n=== Exploratory: features relative to the first two repetitions of each set ===")
rel = df.copy()
for f in ("rms", "mdf"):
    rel[f + "_rel"] = rel.groupby(SET_KEYS)[f].transform(
        lambda x: x / x.iloc[:2].mean())
rel = rel[rel["repetition"] > 2].reset_index(drop=True)
r = loso_logistic(rel, "fatigue_label", ["rms_rel", "mdf_rel"])
print(f"(evaluated on repetitions 3..n) Macro-F1 {r['f1'].mean():.3f} +/- "
      f"{r['f1'].std():.3f}; ROC-AUC {r['roc_auc'].mean():.3f} +/- {r['roc_auc'].std():.3f}")

print("\n=== Feature medians by series (load increases R1 -> R5) ===")
print(df.groupby("series")[["rms", "mdf", "mnf"]].median().round(1))

print("\nLabel analysis completed.")
