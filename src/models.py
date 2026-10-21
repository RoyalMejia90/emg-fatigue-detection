"""
Extended Feature Set, Nested Model Selection and Final Report Figures
(Report Sections 4.2, 4.5, 5.1-5.3 and 6.1-6.5)

Pipeline position: run after validation.py.

Outputs (RESULTS/)
  results_extended.csv, summary_extended.csv, predictions_extended.csv,
  per_class_best.csv, importance.csv, ablation.csv, run_info.txt
  within_set_trends_extended.csv

Deployable model (MODELS/)
  best_model_<name>.pkl      joblib payload: model, features, metrics, metadata
  best_model_metadata.json   human-readable metadata of the exported model

Figures (FIGURES_RESULTS/)
  Figure3_LOSO_Nested_Scheme.png
  Figure4_Confusion_Matrix_Best.png
  Figure5_Per_Subject_Best.png
  Figure6_Importance_Ablation.png

Usage
  python src/models.py
"""

import os
import json
import time
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import antropy as ant
import scipy
import sklearn
from scipy.stats import spearmanr

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    confusion_matrix,
)

# Configuration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "RESULTS")
FIGURES_DIR = os.path.join(ROOT, "FIGURES_RESULTS")
os.makedirs(FIGURES_DIR, exist_ok=True)
MODELS_DIR = os.path.join(ROOT, "MODELS")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
INNER_SPLITS = 5

np.random.seed(RANDOM_STATE)

# Feature Groups
AMPLITUDE = ["rms", "mav", "variance", "waveform_length"]
SPECTRAL_BASE = ["mdf", "mnf", "peak_frequency"]
SPECTRAL_NEW = ["log_fins5", "low_high_ratio"]
NONLINEAR = ["sample_entropy", "higuchi_fd"]
ALL_BASE = AMPLITUDE + SPECTRAL_BASE
ALL_EXT = AMPLITUDE + SPECTRAL_BASE + SPECTRAL_NEW + NONLINEAR

EXPLAIN_MODEL = "Logistic_Ext_tuned"

FAMILIES = {
    "Amplitude": AMPLITUDE,
    "Spectral": SPECTRAL_BASE + SPECTRAL_NEW,
    "Nonlinear": NONLINEAR,
}

# 1. Within-set trends (Section 4.3)

def within_set_trends(df):
    rows = []
    for feat in ALL_EXT:
        rhos, changes = [], []
        for _, g in df.groupby(["subject_id", "file"]):
            g = g.sort_values("repetition")
            rho = spearmanr(g["progress_percent"], g[feat])[0]
            if not np.isnan(rho):
                rhos.append(rho)
            k = max(1, len(g) // 3)
            first, last = g[feat].iloc[:k].mean(), g[feat].iloc[-k:].mean()
            if feat == "log_fins5":
                changes.append(100 * (10 ** (last - first) - 1))
            else:
                changes.append(100 * (last - first) / abs(first) if first != 0 else np.nan)
        rhos = np.array(rhos)
        rows.append({
            "feature": feat,
            "family": next(f for f, v in FAMILIES.items() if feat in v),
            "median_rho": np.median(rhos),
            "q1_rho": np.percentile(rhos, 25),
            "q3_rho": np.percentile(rhos, 75),
            "pct_sets_increasing": 100 * np.mean(rhos > 0),
            "pct_sets_decreasing": 100 * np.mean(rhos < 0),
            "median_change_last_vs_first_third_pct": np.median(changes)
        })
    return pd.DataFrame(rows)

# 2. Models

def pipe(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

def tuned(estimator, grid):
    return ("nested", pipe(estimator), {f"model__{k}": v for k, v in grid.items()})

C_GRID = {"C": [0.01, 0.1, 1, 10, 100]}

MODELS = {
    "Majority_Class": (("plain", DummyClassifier(strategy="most_frequent"), None), ["rms"]),
    "Logistic_RMS_MDF": (("plain", pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)), None), ["rms", "mdf"]),
    "Logistic_RMS_MDF_tuned": (tuned(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), C_GRID), ["rms", "mdf"]),
    "Logistic_All": (("plain", pipe(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)), None), ALL_BASE),
    "LDA_All": (("plain", pipe(LinearDiscriminantAnalysis()), None), ALL_BASE),
    "Logistic_Ext_tuned": (tuned(LogisticRegression(max_iter=5000, random_state=RANDOM_STATE), C_GRID), ALL_EXT),
    "LDA_Ext_shrinkage": (("plain", pipe(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")), None), ALL_EXT),
    "SVM_RBF_Ext_tuned": (tuned(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
                                {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]}), ALL_EXT),
    "RF_Ext_tuned": (tuned(RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
                           {"max_depth": [3, 6, None], "min_samples_leaf": [5, 20]}), ALL_EXT),
}

def build(spec):
    kind, estimator, grid = spec
    if kind == "plain":
        return clone(estimator)
    return GridSearchCV(clone(estimator), grid, scoring="f1_macro",
                        cv=GroupKFold(n_splits=INNER_SPLITS), n_jobs=-1, refit=True)

def fit(model, X, y, groups):
    if isinstance(model, GridSearchCV):
        model.fit(X, y, groups=groups)
    else:
        model.fit(X, y)
    return model

def fold_metrics(y, pred, prob):
    both = y.nunique() > 1
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="macro", zero_division=0),
        "recall": recall_score(y, pred, average="macro", zero_division=0),
        "f1": f1_score(y, pred, average="macro", zero_division=0),
        "f1_non_fatigued": f1_score(y, pred, pos_label=0, zero_division=0),
        "f1_fatigued": f1_score(y, pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, prob) if both else np.nan,
        "pr_auc": average_precision_score(y, prob) if both else np.nan,
        "kappa": cohen_kappa_score(y, pred),
    }

def run_loso(df, spec, features, keep_models=False):
    rows, preds, fitted = [], [], []
    logo = LeaveOneGroupOut()
    y_all, g_all = df["fatigue_label"], df["subject_id"]
    for fold, (tr, te) in enumerate(logo.split(df, y_all, g_all), start=1):
        train, test = df.iloc[tr], df.iloc[te]
        model = build(spec)
        fit(model, train[features], train["fatigue_label"], train["subject_id"])
        pred = model.predict(test[features])
        prob = model.predict_proba(test[features])[:, 1]
        m = fold_metrics(test["fatigue_label"], pred, prob)
        m.update({
            "fold": fold,
            "subject_id": test["subject_id"].iloc[0],
            "best_params": json.dumps(model.best_params_) if isinstance(model, GridSearchCV) else ""
        })
        rows.append(m)
        preds.append(pd.DataFrame({
            "index": test.index,
            "subject_id": test["subject_id"].values,
            "y_true": test["fatigue_label"].values,
            "y_pred": pred,
            "p_fatigued": prob
        }))
        if keep_models:
            fitted.append((model, test))
    return pd.DataFrame(rows), pd.concat(preds), fitted

# 3. Held-out permutation importance

def heldout_permutation(fitted, features, n_repeats=20, metric="auc"):
    rng = np.random.default_rng(RANDOM_STATE)
    def score(model, X, y):
        if metric == "auc":
            return roc_auc_score(y, model.predict_proba(X)[:, 1])
        return f1_score(y, model.predict(X), average="macro", zero_division=0)

    rows = []
    for model, test in fitted:
        X, y = test[features].copy(), test["fatigue_label"]
        base = score(model, X, y)
        for feat in features:
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[feat] = rng.permutation(Xp[feat].values)
                drops.append(base - score(model, Xp, y))
            rows.append({"subject_id": test["subject_id"].iloc[0], "feature": feat,
                         "importance": np.mean(drops)})
    imp = pd.DataFrame(rows)
    return (imp.groupby("feature")["importance"].agg(["mean", "std"])
            .sort_values("mean", ascending=False).reset_index())

# 4. Figures

COLORS = {"train": "#2F5D8A", "test": "#E07B39", "inner": "#9DB9D5"}

def figure3(subjects, path):
    n = len(subjects)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for i in range(n):
        for j in range(n):
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 0.92, 0.85,
                         color=COLORS["test"] if i == j else COLORS["train"]))
    ax.set_xlim(0, n)
    ax.set_ylim(-0.3, n)
    ax.set_xticks(np.arange(n) + 0.46)
    ax.set_xticklabels(subjects, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(n) + 0.42)
    ax.set_yticklabels([f"Fold {k}" for k in range(n, 0, -1)], fontsize=8)
    ax.set_xlabel("Participant (all windows of both legs and all sets)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS["train"]),
               plt.Rectangle((0, 0), 1, 1, color=COLORS["test"])]
    ax.legend(handles, ["Training subjects (14): scaler fit + inner 5-fold GroupKFold grid search",
                        "Held-out test subject (1): used only for the final evaluation"],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False, fontsize=9)
    ax.set_title("Outer loop: leave-one-subject-out (15 folds); inner loop: grouped 5-fold CV on the 14 training subjects",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def figure4(cm, title, path):
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    labels = ["Non-fatigued", "Fatigued"]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f} %)", ha="center", va="center",
                    fontsize=13, color="white" if pct[i, j] > 55 else "#1b1b1b")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def figure5(best_folds, comp_folds, best_name, comp_name, path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(best_folds))
    ax.bar(x, best_folds["f1"], color="#2F5D8A", label=f"Best model ({best_name})")
    ax.scatter(x, comp_folds["f1"], color="#E07B39", zorder=3, s=28,
               label=f"Comparison: {comp_name}")
    ax.axhline(best_folds["f1"].mean(), color="#2F5D8A", ls="--", lw=1.2,
               label=f"Best model mean = {best_folds['f1'].mean():.3f}")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="Chance-level macro-F1 for balanced classes (0.5)")
    for xi, v in zip(x, best_folds["f1"]):
        ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(best_folds["subject_id"], rotation=0, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 (held-out subject)")
    ax.set_xlabel("Held-out subject (LOSO fold)")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

FAMILY_COLOR = {"Amplitude": "#2F5D8A", "Spectral": "#E07B39", "Nonlinear": "#5B9E57"}

def family_of(feat):
    for fam, feats in FAMILIES.items():
        if feat in feats:
            return fam
    return "Other"

def figure6(imp, abl, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    imp = imp.iloc[::-1]
    ax.barh(imp["feature"], imp["mean"], xerr=imp["std"],
            color=[FAMILY_COLOR[family_of(f)] for f in imp["feature"]], ecolor="#555", capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Drop in held-out ROC-AUC when the feature is permuted")
    ax.set_title("(a) Logistic regression, 3 families: permutation importance\non the held-out subject (mean $\\pm$ SD, 15 LOSO folds)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    ax.legend(handles, FAMILY_COLOR.keys(), fontsize=8, loc="lower right")
    ax = axes[1]
    order = abl.reset_index(drop=True)
    y = np.arange(len(order))[::-1]
    ax.barh(y, order["auc_mean"], xerr=order["auc_std"], color="#9DB9D5", ecolor="#555", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s} ({n})" for s, n in zip(order["feature_set"], order["n_features"])], fontsize=8)
    full = order.loc[order["feature_set"] == "All families", "auc_mean"].iloc[0]
    ax.axvline(full, color="#2F5D8A", ls="--", lw=1)
    ax.axvline(0.5, color="grey", ls=":", lw=1)
    for yi, v, f1 in zip(y, order["auc_mean"], order["f1_mean"]):
        ax.text(0.02, yi, f"AUC {v:.3f} | F1 {f1:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("LOSO ROC-AUC (mean $\\pm$ SD); macro-F1 printed in each bar")
    ax.set_title("(b) Family ablation, same model and nested tuning", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

# Final deployable model

def train_final_model(df, spec, features):
    """
    Train deployable model on entire dataset.
    """
    model = build(spec)

    fit(
        model,
        df[features],
        df["fatigue_label"],
        df["subject_id"]
    )

    return model

# Main

if __name__ == "__main__":
    t0 = time.time()
    # Load dataset (already contains 11 features from feature_extraction.py)
    df = pd.read_csv(os.path.join(RESULTS_DIR, "fatigue_dataset.csv"))
    print(f"Windows: {len(df)} | subjects: {df.subject_id.nunique()}")

    # Within-set trends
    trends = within_set_trends(df)
    trends.to_csv(os.path.join(RESULTS_DIR, "within_set_trends_extended.csv"), index=False)
    print(trends.round(3).to_string(index=False))

    all_rows, all_preds, kept = [], [], {}
    for name, (spec, feats) in MODELS.items():
        t = time.time()
        folds, preds, fitted = run_loso(df, spec, feats, keep_models=True)
        folds.insert(0, "model", name)
        preds.insert(0, "model", name)
        all_rows.append(folds)
        all_preds.append(preds)
        kept[name] = fitted
        print(f"{name:26s} F1 {folds.f1.mean():.3f} +/- {folds.f1.std():.3f} | "
              f"AUC {folds.roc_auc.mean():.3f} | {time.time() - t:.0f} s")

    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(os.path.join(RESULTS_DIR, "results_extended.csv"), index=False)
    pd.concat(all_preds).to_csv(os.path.join(RESULTS_DIR, "predictions_extended.csv"), index=False)
    metrics = ["accuracy", "precision", "recall", "f1", "f1_non_fatigued", "f1_fatigued",
               "roc_auc", "pr_auc", "kappa"]
    summary = results.groupby("model", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_extended.csv"), index=False)

    best = summary.loc[summary["f1_mean"].idxmax(), "model"]
    print(f"\nBest model by mean macro-F1: {best}")
    best_spec, best_feats = MODELS[best]
    preds = pd.concat(all_preds)
    bp = preds[preds["model"] == best]

    # Per-class (pooled over folds)
    rows = []
    for cls, name in [(0, "Non-fatigued"), (1, "Fatigued")]:
        rows.append({
            "class": name,
            "precision": precision_score(bp.y_true, bp.y_pred, pos_label=cls),
            "recall": recall_score(bp.y_true, bp.y_pred, pos_label=cls),
            "f1": f1_score(bp.y_true, bp.y_pred, pos_label=cls),
            "support": int((bp.y_true == cls).sum()),
        })
    per_fold = []
    for _, g in bp.groupby("subject_id"):
        for cls in (0, 1):
            per_fold.append({"class": cls,
                             "precision": precision_score(g.y_true, g.y_pred, pos_label=cls, zero_division=0),
                             "recall": recall_score(g.y_true, g.y_pred, pos_label=cls, zero_division=0),
                             "f1": f1_score(g.y_true, g.y_pred, pos_label=cls, zero_division=0)})
    pf = pd.DataFrame(per_fold).groupby("class").agg(["mean", "std"])
    per_class = pd.DataFrame(rows)
    for k, cls in enumerate((0, 1)):
        for m in ("precision", "recall", "f1"):
            per_class.loc[k, f"{m}_fold_mean"] = pf.loc[cls, (m, "mean")]
            per_class.loc[k, f"{m}_fold_std"] = pf.loc[cls, (m, "std")]
    per_class.to_csv(os.path.join(RESULTS_DIR, "per_class_best.csv"), index=False)

    cm = confusion_matrix(bp.y_true, bp.y_pred, labels=[0, 1])
    print("Confusion matrix (pooled over folds):\n", cm)

    # Importance
    imp_best = heldout_permutation(kept[best], best_feats, metric="auc")
    imp_best.insert(0, "model", best)
    exp_spec, exp_feats = MODELS[EXPLAIN_MODEL]
    imp = heldout_permutation(kept[EXPLAIN_MODEL], exp_feats, metric="auc")
    imp.insert(0, "model", EXPLAIN_MODEL)
    pd.concat([imp_best, imp]).to_csv(os.path.join(RESULTS_DIR, "importance.csv"), index=False)
    print(imp_best.round(3).to_string(index=False))
    print(imp.round(3).to_string(index=False))

    # Family ablation
    sets = {"All families": ALL_EXT}
    for fam, feats in FAMILIES.items():
        sets[f"Without {fam.lower()}"] = [f for f in ALL_EXT if f not in feats]
    for fam, feats in FAMILIES.items():
        sets[f"{fam} only"] = feats
    abl_rows = []
    for label, feats in sets.items():
        folds, _, _ = run_loso(df, exp_spec, feats)
        abl_rows.append({"feature_set": label, "n_features": len(feats),
                         "f1_mean": folds.f1.mean(), "f1_std": folds.f1.std(),
                         "auc_mean": folds.roc_auc.mean(), "auc_std": folds.roc_auc.std()})
        print(f"Ablation {label:22s} F1 {folds.f1.mean():.3f} +/- {folds.f1.std():.3f} "
              f"AUC {folds.roc_auc.mean():.3f} +/- {folds.roc_auc.std():.3f}")
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)

    # Figures
    subjects = sorted(df.subject_id.unique())
    figure3(subjects, os.path.join(FIGURES_DIR, "Figure3_LOSO_Nested_Scheme.png"))
    figure4(cm, f"{best}: confusion matrix pooled over 15 LOSO folds\n(counts and row-normalised %)",
            os.path.join(FIGURES_DIR, "Figure4_Confusion_Matrix_Best.png"))
    bf = results[results.model == best].reset_index(drop=True)
    comp = EXPLAIN_MODEL if best != EXPLAIN_MODEL else "Logistic_RMS_MDF"
    cf = results[results.model == comp].reset_index(drop=True)
    figure5(bf, cf, best, comp, os.path.join(FIGURES_DIR, "Figure5_Per_Subject_Best.png"))
    if len(abl):
        figure6(imp, abl, os.path.join(FIGURES_DIR, "Figure6_Importance_Ablation.png"))

    # Deployable model: refit best model on the ENTIRE dataset and serialise it
    final_model = train_final_model(df, best_spec, best_feats)
    best_row = summary.loc[summary["model"] == best].iloc[0]
    train_subjects = [int(s) if isinstance(s, (int, np.integer)) else str(s)
                      for s in sorted(df["subject_id"].unique())]
    saved_at = datetime.now().isoformat(timespec="seconds")
    best_params = (json.dumps(final_model.best_params_)
                   if isinstance(final_model, GridSearchCV) else "")

    payload = {
        "model": final_model,
        "model_name": best,
        "features": list(best_feats),
        "f1_mean": float(best_row["f1_mean"]),
        "f1_std": float(best_row["f1_std"]),
        "roc_auc_mean": float(best_row["roc_auc_mean"]),
        "train_subjects": train_subjects,
        "train_samples": int(len(df)),
        "best_params": best_params,
        "sklearn_version": sklearn.__version__,
        "saved_at": saved_at,
    }
    model_path = os.path.join(MODELS_DIR, f"best_model_{best}.pkl")
    joblib.dump(payload, model_path)

    metadata = {
        "best_model": best,
        "model_file": os.path.basename(model_path),
        "features": list(best_feats),
        "train_subjects": train_subjects,
        "n_train_subjects": len(train_subjects),
        "train_samples": int(len(df)),
        "f1_mean": float(best_row["f1_mean"]),
        "f1_std": float(best_row["f1_std"]),
        "roc_auc_mean": float(best_row["roc_auc_mean"]),
        "best_params": best_params,
        "sklearn_version": sklearn.__version__,
        "saved_at": saved_at,
    }
    with open(os.path.join(MODELS_DIR, "best_model_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=4, ensure_ascii=False)
    print(f"Deployable model saved: {model_path}")

    elapsed = time.time() - t0
    with open(os.path.join(RESULTS_DIR, "run_info.txt"), "w") as fh:
        fh.write(f"random_state={RANDOM_STATE}\n")
        fh.write(f"python={platform.python_version()} numpy={np.__version__} pandas={pd.__version__} "
                 f"scipy={scipy.__version__} scikit-learn={sklearn.__version__} "
                 f"antropy={ant.__version__} matplotlib={matplotlib.__version__}\n")
        fh.write(f"machine={platform.machine()} processor={platform.processor() or 'n/a'} "
                 f"cpus={os.cpu_count()} system={platform.system()}\n")
        fh.write(f"total_runtime_s={elapsed:.0f}\n")
    print(f"\nTotal runtime: {elapsed:.0f} s")
    print("Extended analysis completed.")