"""
Evaluation report for an external raw sEMG signal (single CSV, any file name).

Runs the same pipeline as predict_raw_signal.py and writes a descriptive
report. No ground-truth labels are assumed, so NO classification metrics
(F1, AUC...) are computed for the new signal; the LOSO figures shown in the
report are the training-time reference for the model.

Usage
    python src/evaluate_external_signal.py --csv path/to/any_name.csv
    python src/evaluate_external_signal.py --csv signal.csv --outdir out/ --model MODELS/x.pkl

Outputs (default: RESULTS/EXTERNAL/<csv stem>/)
    predictions.csv
    report.txt
    plot.png
"""

import os
import re
import sys
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import predict_raw_signal as prs  # noqa: E402

FEATURE_UNITS = {
    "rms": "mV", "mav": "mV", "variance": "mV²", "waveform_length": "mV",
    "mdf": "Hz", "mnf": "Hz", "peak_frequency": "Hz",
    "log_fins5": "log10", "low_high_ratio": "", "sample_entropy": "", "higuchi_fd": "",
}
MIN_REPS_FOR_TREND = 5
DECISION_THRESHOLD = 0.5

# ============================================================
# ANALYSIS
# ============================================================

def trend_statistics(table):
    """Descriptive trend of P(fatigued) across repetitions."""
    p = table["probability_fatigued"].values
    x = np.arange(len(p))
    out = {"n": len(p), "reliable": len(p) >= MIN_REPS_FOR_TREND}
    if len(p) >= 3:
        out["slope_per_rep"] = float(np.polyfit(x, p, 1)[0])
        rho, pval = spearmanr(x, p)
        out["spearman_rho"] = float(rho) if not np.isnan(rho) else np.nan
        out["spearman_p"] = float(pval) if not np.isnan(pval) else np.nan
    else:
        out["slope_per_rep"] = out["spearman_rho"] = out["spearman_p"] = np.nan
    k = max(1, len(p) // 3)
    out["mean_first_third"] = float(p[:k].mean())
    out["mean_last_third"] = float(p[-k:].mean())
    return out


def describe_trend(t):
    if not t["reliable"]:
        return (f"Only {t['n']} repetitions: too few (< {MIN_REPS_FOR_TREND}) to "
                f"describe a trend reliably.")
    delta = t["mean_last_third"] - t["mean_first_third"]
    if abs(delta) < 0.10:
        shape = "roughly flat"
    elif delta > 0:
        shape = "increasing"
    else:
        shape = "decreasing"
    return (f"P(fatigued) is {shape} across the set (last third minus first "
            f"third = {delta:+.3f}).")


def build_report(csv_file, table, ctx, trend, caught):
    feats = ctx["features"]
    n = len(table)
    n_fat = int((table["predicted_label"] == 1).sum())
    info = ctx["load_info"]
    L = []
    add = L.append

    add("EXTERNAL sEMG SIGNAL - FATIGUE EVALUATION REPORT")
    add("=" * 60)
    add(f"Generated   : {datetime.now().isoformat(timespec='seconds')}")
    add(f"Input file  : {os.path.basename(csv_file)}")
    add("")
    add("MODEL")
    add("-" * 60)
    add(f"Model       : {ctx['model_name']}")
    add(f"Features    : {', '.join(feats)}")
    add(f"Model file  : {os.path.basename(ctx['model_file'])}")
    if ctx["cv_f1_mean"] is not None:
        add(f"Reference (LOSO, 15 training subjects): macro-F1 = "
            f"{ctx['cv_f1_mean']:.3f}, ROC-AUC = {ctx['cv_roc_auc_mean']:.3f}")
    add("")
    add("SIGNAL")
    add("-" * 60)
    fs = info["fs_estimated"]
    add(f"Estimated sampling rate : {f'{fs:.1f} Hz' if fs else 'n/a (assumed 1000 Hz)'}")
    add(f"Resampled to 1000 Hz    : {'yes' if info['resampled'] else 'no'}")
    add(f"Interpolated samples    : {info['n_interpolated']}")
    add(f"Repetitions detected    : {len(ctx['peaks'])}")
    add(f"Repetitions analysed    : {n} (discarded: {ctx['discarded']})")
    add("")
    add("RESULTS")
    add("-" * 60)
    add(f"Classified as fatigued  : {n_fat}/{n} ({100 * n_fat / n:.1f}%)")
    add(f"Mean P(fatigued)        : {table['probability_fatigued'].mean():.3f}")
    add(f"Min / max P(fatigued)   : {table['probability_fatigued'].min():.3f} / "
        f"{table['probability_fatigued'].max():.3f}")
    add(f"Decision threshold      : {DECISION_THRESHOLD}")
    add("")
    add("Per-repetition feature summary (mean +/- SD):")
    for f in feats:
        add(f"  {f:16s}: {table[f].mean():.4g} +/- {table[f].std(ddof=1) if n > 1 else float('nan'):.4g}")
    add("")
    add("TREND OF P(fatigued) ACROSS THE SET")
    add("-" * 60)
    add(describe_trend(trend))
    if not np.isnan(trend["slope_per_rep"]):
        add(f"Linear slope            : {trend['slope_per_rep']:+.4f} per repetition")
        add(f"Spearman rho (p-value)  : {trend['spearman_rho']:+.3f} ({trend['spearman_p']:.3g})")
    add(f"Mean first third / last third : {trend['mean_first_third']:.3f} / "
        f"{trend['mean_last_third']:.3f}")
    add("")
    add("QUALITY FLAGS")
    add("-" * 60)
    flags = []
    if ctx["used_fallback_prominence"]:
        flags.append("Peak detection used a relative prominence (the training "
                     "threshold found no peaks). Check the repetition markers in plot.png.")
    if info["resampled"]:
        flags.append("Signal was resampled to 1000 Hz by linear interpolation.")
    if ctx["discarded"]:
        flags.append(f"{ctx['discarded']} segment(s) discarded (too short or NaN features).")
    if n < MIN_REPS_FOR_TREND:
        flags.append("Few repetitions analysed; results are weakly informative.")
    handled = ("No peaks with the training prominence", "Sampling rate ~")
    for w in caught:
        if not w.startswith(handled):
            flags.append(f"Warning: {w}")
    if flags:
        for f in flags:
            add(f"  * {f}")
    else:
        add("  None.")
    add("")
    add("LIMITATIONS")
    add("-" * 60)
    add("  * No ground-truth labels were provided: no accuracy/F1/AUC is computed "
        "for this signal.")
    add("  * The training reference is moderate and varies across subjects "
        "(see macro-F1 above); the per-repetition probability and its trend are "
        "more informative than the single label.")
    add("  * 'rms' enters the model in absolute units (mV). Signals acquired with a "
        "different gain/electrode setup than the training data may shift the "
        "probabilities regardless of muscle state.")
    add("  * Output describes the sEMG pattern relative to the training population; "
        "it is not a clinical diagnosis.")
    return "\n".join(L) + "\n"


def save_plot(table, ctx, trend, out_png):
    feats = ctx["features"][:2]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5))

    ax = axes[0]
    t = np.arange(len(ctx["envelope"])) / prs.FS
    ax.plot(t, ctx["envelope"], lw=0.8, color="#2F5D8A")
    ax.plot(ctx["peaks"] / prs.FS, ctx["envelope"][ctx["peaks"]], "v", color="#E07B39")
    for b in ctx["boundaries"][1:-1]:
        ax.axvline(b / prs.FS, color="grey", ls=":", lw=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS envelope (mV)")
    ax.set_title("(a) Detected repetitions")
    ax.legend(handles=[
        plt.Line2D([], [], color="#2F5D8A", lw=1, label="RMS envelope"),
        plt.Line2D([], [], color="#E07B39", marker="v", ls="", label="Detected peak (one per repetition)"),
        plt.Line2D([], [], color="grey", ls=":", label="Segment boundary between repetitions"),
    ], fontsize=8, loc="upper right")

    ax = axes[1]
    fat = table["predicted_label"] == 1
    rep = table["repetition"].values
    p = table["probability_fatigued"].values
    ax.plot(rep, p, color="grey", lw=0.8)
    ax.scatter(rep[~fat.values], p[~fat.values], color="#5B9E57", label="Non-fatigued", zorder=3)
    ax.scatter(rep[fat.values], p[fat.values], color="#C0392B", label="Fatigued", zorder=3)
    if trend["reliable"] and not np.isnan(trend["slope_per_rep"]):
        coef = np.polyfit(np.arange(len(p)), p, 1)
        ax.plot(rep, np.polyval(coef, np.arange(len(p))), "--", color="#2F5D8A",
                lw=1.2, label="Linear trend")
    ax.axhline(DECISION_THRESHOLD, ls=":", color="black", lw=0.8,
               label=f"Decision threshold ({DECISION_THRESHOLD})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Repetition")
    ax.set_ylabel("P(fatigued)")
    ax.set_title("(b) Fatigue probability per repetition")
    ax.legend(fontsize=8)

    ax = axes[2]
    colors = ["#2F5D8A", "#E07B39"]
    markers = ["o", "s"]
    styles = ["-", "--"]
    handles = []
    twin = None
    for i, f in enumerate(feats):
        a = ax if i == 0 else ax.twinx()
        if i == 1:
            twin = a
        label = f"{f} ({FEATURE_UNITS[f]})" if FEATURE_UNITS.get(f) else f
        (line,) = a.plot(rep, table[f], marker=markers[i], ls=styles[i],
                         color=colors[i], lw=1.4, label=label)
        a.set_ylabel(label, color=colors[i])
        a.tick_params(axis="y", colors=colors[i])
        handles.append(line)
    ax.set_xticks(rep)
    ax.set_xlabel("Repetition")
    title = "(c) Model input features per repetition"
    if len(ctx["features"]) > 2:
        title += f" (first 2 of {len(ctx['features'])} shown)"
    ax.set_title(title)
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Report for an external raw sEMG CSV.")
    parser.add_argument("--csv", required=True, help="Path to the raw EMG CSV (any file name).")
    parser.add_argument("--model", default=None, help="Optional .pkl path (default: from metadata).")
    parser.add_argument("--outdir", default=None,
                        help="Output folder (default: RESULTS/EXTERNAL/<csv stem>).")
    parser.add_argument("--peak-prominence", type=float, default=prs.PEAK_PROMINENCE)
    args = parser.parse_args()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        table, ctx = prs.predict_signal(args.csv, args.model, args.peak_prominence)
    caught = sorted({str(x.message) for x in w})

    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(args.csv))[0])
    outdir = args.outdir or os.path.join(prs.ROOT, "RESULTS", "EXTERNAL", stem)
    os.makedirs(outdir, exist_ok=True)

    trend = trend_statistics(table)
    table.to_csv(os.path.join(outdir, "predictions.csv"), index=False)
    with open(os.path.join(outdir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(build_report(args.csv, table, ctx, trend, caught))
    save_plot(table, ctx, trend, os.path.join(outdir, "plot.png"))

    n_fat = int((table["predicted_label"] == 1).sum())
    print("\n=== EVALUATION COMPLETED ===")
    print(f"Repetitions analysed : {len(table)}")
    print(f"Fatigued repetitions : {n_fat}/{len(table)} ({100 * n_fat / len(table):.1f}%)")
    print(f"Output folder        : {outdir}")
    print("Files                : predictions.csv, report.txt, plot.png")


if __name__ == "__main__":
    main()