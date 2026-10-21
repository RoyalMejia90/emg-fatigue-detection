"""
Local inference for a raw sEMG signal (single CSV, any file name).

Pipeline (mirrors preprocessing.py + feature_extraction.py):
    raw CSV
        -> DC removal, 1 Hz high-pass, 20-450 Hz band-pass, 60 Hz notch
        -> RMS envelope (250 ms window + 200-sample smoothing)
        -> repetition detection (peaks of the envelope)
        -> segmentation at midpoints between consecutive peaks
        -> features required by the saved model (computed with the SAME
           functions used in training: src/feature_extraction.py)
        -> saved model (MODELS/*.pkl) -> fatigue prediction

The file name is never used to infer anything; only its content matters.
Required content: a voltage column ("Voltage_mV"); "Relative_Time_s" is used
to check the sampling rate when present.

Usage
    python src/predict_raw_signal.py --csv path/to/any_name.csv
    python src/predict_raw_signal.py --csv signal.csv --outdir out/ --model MODELS/x.pkl

Outputs (next to the CSV, or in --outdir)
    <stem>_predictions.csv
    <stem>_prediction_plot.png
"""

import os
import re
import sys
import json
import glob
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, iirnotch, find_peaks

# Reuse the exact feature functions used in training.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import feature_extraction as fe  # noqa: E402

# CONFIGURATION (must match preprocessing.py / feature_extraction.py)

ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ROOT, "MODELS")
METADATA_FILE = os.path.join(MODELS_DIR, "best_model_metadata.json")

FS = 1000
POWERLINE_FREQ = 60
NOTCH_Q = 30
RMS_WINDOW_MS = 250
RMS_SMOOTHING_WINDOW = 200
MIN_SEGMENT_SAMPLES = fe.MIN_SEGMENT          # 500, same as training

# Repetition detection: identical to src/repetition_detection.py (the script
# that produced RESULTS/repetition_summary.csv during training), applied to
# the same RMS envelope.
PEAK_MIN_DISTANCE_S = 0.8
PEAK_PROMINENCE = 10
PEAK_WIDTH = 50
FALLBACK_PROMINENCE_FRACTION = 0.25   # of the envelope range, used only if no peaks

FS_TOLERANCE = 0.02                    # >2 % deviation -> resample to FS
MIN_ACCEPTABLE_FS = 900                # need content up to 450 Hz

VOLTAGE_COLUMN = "Voltage_mV"
TIME_COLUMN = "Relative_Time_s"

FEATURE_FUNCTIONS = {
    "rms": fe.compute_rms,
    "mav": fe.compute_mav,
    "variance": fe.compute_variance,
    "waveform_length": fe.compute_waveform_length,
    "mdf": fe.compute_mdf,
    "mnf": fe.compute_mnf,
    "peak_frequency": fe.compute_peak_frequency,
    "log_fins5": fe.compute_log_fins5,
    "low_high_ratio": fe.compute_low_high_ratio,
    "sample_entropy": fe.compute_sample_entropy,
    "higuchi_fd": fe.compute_higuchi_fd,
}

# PREPROCESSING (identical filters to preprocessing.py)

def preprocess_signal(signal):
    nyq = FS / 2.0
    signal = signal - np.mean(signal)
    b, a = butter(4, 1 / nyq, btype="highpass")
    signal = filtfilt(b, a, signal)
    b, a = butter(4, [20 / nyq, 450 / nyq], btype="bandpass")
    signal = filtfilt(b, a, signal)
    b, a = iirnotch(POWERLINE_FREQ, NOTCH_Q, FS)
    return filtfilt(b, a, signal)


def rms_envelope(signal):
    w = int(RMS_WINDOW_MS * FS / 1000)
    rms = np.sqrt(np.convolve(signal ** 2, np.ones(w) / w, mode="same"))
    return (pd.Series(rms)
            .rolling(window=RMS_SMOOTHING_WINDOW, center=True, min_periods=1)
            .mean().values)

# INPUT HANDLING

def find_column(df, wanted):
    """Case/format-insensitive column lookup."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    for c in df.columns:
        if norm(c) == norm(wanted):
            return c
    return None


def load_signal(csv_file):
    df = pd.read_csv(csv_file)

    vcol = find_column(df, VOLTAGE_COLUMN)
    if vcol is None:
        raise ValueError(
            f"Column '{VOLTAGE_COLUMN}' not found. Columns in file: {list(df.columns)}"
        )
    tcol = find_column(df, TIME_COLUMN)

    cols = [vcol] + ([tcol] if tcol else [])
    df = df[cols].apply(pd.to_numeric, errors="coerce")

    n_bad = int(df[vcol].isna().sum())
    if n_bad:
        warnings.warn(f"{n_bad} non-numeric/NaN voltage samples were interpolated.")
        df[vcol] = df[vcol].interpolate(limit_direction="both")
    if df[vcol].isna().all():
        raise ValueError("Voltage column contains no numeric data.")

    signal = df[vcol].values.astype(float)
    info = {"fs_estimated": None, "resampled": False, "n_interpolated": n_bad}

    if tcol is not None and df[tcol].notna().all() and len(df) > 2:
        t = df[tcol].values.astype(float)
        dt = np.median(np.diff(t))
        if dt > 0:
            fs_est = 1.0 / dt
            info["fs_estimated"] = float(fs_est)
            if fs_est < MIN_ACCEPTABLE_FS:
                raise ValueError(
                    f"Estimated sampling rate is {fs_est:.0f} Hz; at least "
                    f"{MIN_ACCEPTABLE_FS} Hz are needed for the 20-450 Hz analysis."
                )
            if abs(fs_est - FS) / FS > FS_TOLERANCE:
                warnings.warn(
                    f"Sampling rate ~{fs_est:.0f} Hz differs from {FS} Hz; "
                    f"resampling by linear interpolation."
                )
                t_new = np.arange(t[0], t[-1], 1.0 / FS)
                signal = np.interp(t_new, t, signal)
                info["resampled"] = True
    else:
        warnings.warn(f"No usable time column; assuming {FS} Hz.")

    if len(signal) < 2 * MIN_SEGMENT_SAMPLES:
        raise ValueError("Signal is too short to contain a repetition.")
    return signal, info

# REPETITION DETECTION / SEGMENTATION

def detect_repetitions(envelope, prominence=PEAK_PROMINENCE):
    kwargs = dict(distance=int(PEAK_MIN_DISTANCE_S * FS), width=PEAK_WIDTH)
    peaks, _ = find_peaks(envelope, prominence=prominence, **kwargs)
    used_fallback = False
    if len(peaks) == 0:
        rel = FALLBACK_PROMINENCE_FRACTION * (envelope.max() - envelope.min())
        peaks, _ = find_peaks(envelope, prominence=rel, **kwargs)
        used_fallback = len(peaks) > 0
        if used_fallback:
            warnings.warn(
                "No peaks with the training prominence; used a relative "
                f"prominence ({rel:.4g}). Check the plot."
            )
    return peaks, used_fallback


def build_boundaries(peaks, n):
    b = [0]
    for i in range(len(peaks) - 1):
        b.append(int((peaks[i] + peaks[i + 1]) / 2))
    b.append(n)
    return b

# MODEL

def resolve_model_path(model_arg=None):
    if model_arg:
        return model_arg
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, encoding="utf-8") as fh:
            name = json.load(fh).get("model_file")
        if name and os.path.exists(os.path.join(MODELS_DIR, name)):
            return os.path.join(MODELS_DIR, name)
    found = sorted(glob.glob(os.path.join(MODELS_DIR, "best_model_*.pkl")))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"Could not determine the model in {MODELS_DIR}. Use --model."
    )


def load_payload(model_file):
    import sklearn
    payload = joblib.load(model_file)
    saved = payload.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        warnings.warn(
            f"Model trained with scikit-learn {saved}, installed {sklearn.__version__}."
        )
    unknown = [f for f in payload["features"] if f not in FEATURE_FUNCTIONS]
    if unknown:
        raise ValueError(f"Model needs unsupported features: {unknown}")
    return payload

# CORE

def predict_signal(csv_file, model_file=None, prominence=PEAK_PROMINENCE):
    """Return (predictions_df, context_dict). Reusable by other scripts."""
    model_file = resolve_model_path(model_file)
    payload = load_payload(model_file)
    model, feats = payload["model"], list(payload["features"])

    signal, load_info = load_signal(csv_file)
    filtered = preprocess_signal(signal)
    envelope = rms_envelope(filtered)

    peaks, used_fallback = detect_repetitions(envelope, prominence)
    if len(peaks) == 0:
        raise ValueError("No repetitions detected in the signal.")
    if len(peaks) == 1:
        warnings.warn("Only one repetition detected; the result is not informative.")

    boundaries = build_boundaries(peaks, len(filtered))
    rows, discarded = [], 0
    for rep in range(len(boundaries) - 1):
        start, end = boundaries[rep], boundaries[rep + 1]
        seg = filtered[start:end]
        if len(seg) < MIN_SEGMENT_SAMPLES:
            discarded += 1
            continue
        row = {"repetition": rep + 1,
               "start_s": start / FS, "end_s": end / FS,
               "peak_s": float(peaks[min(rep, len(peaks) - 1)] / FS),
               "samples": len(seg)}
        row.update({f: FEATURE_FUNCTIONS[f](seg) for f in feats})
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No valid repetitions (all segments too short).")

    nan_mask = table[feats].isna().any(axis=1)
    if nan_mask.any():
        discarded += int(nan_mask.sum())
        warnings.warn(f"{int(nan_mask.sum())} repetitions dropped (NaN features).")
        table = table.loc[~nan_mask].reset_index(drop=True)
        if table.empty:
            raise ValueError("All repetitions had NaN features.")

    X = table[feats]
    table["predicted_label"] = model.predict(X)
    table["probability_fatigued"] = model.predict_proba(X)[:, 1]

    context = {
        "model_file": model_file, "model_name": payload.get("model_name"),
        "features": feats, "envelope": envelope, "peaks": peaks,
        "boundaries": boundaries, "discarded": discarded,
        "used_fallback_prominence": used_fallback,
        "load_info": load_info,
        "cv_f1_mean": payload.get("f1_mean"), "cv_roc_auc_mean": payload.get("roc_auc_mean"),
    }
    return table, context


def save_plot(table, ctx, out_png):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))
    t = np.arange(len(ctx["envelope"])) / FS
    ax1.plot(t, ctx["envelope"], lw=0.8, color="#2F5D8A")
    ax1.plot(ctx["peaks"] / FS, ctx["envelope"][ctx["peaks"]], "v", color="#E07B39")
    for b in ctx["boundaries"][1:-1]:
        ax1.axvline(b / FS, color="grey", ls=":", lw=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMS envelope (mV)")
    ax1.set_title("Detected repetitions")

    fat = table["predicted_label"] == 1
    ax2.plot(table["repetition"], table["probability_fatigued"], color="grey", lw=0.8)
    ax2.scatter(table.loc[~fat, "repetition"], table.loc[~fat, "probability_fatigued"],
                color="#5B9E57", label="Non-fatigued", zorder=3)
    ax2.scatter(table.loc[fat, "repetition"], table.loc[fat, "probability_fatigued"],
                color="#C0392B", label="Fatigued", zorder=3)
    ax2.axhline(0.5, ls="--", color="black", lw=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Repetition")
    ax2.set_ylabel("P(fatigued)")
    ax2.set_title("Fatigue prediction per repetition")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# CLI

def main():
    parser = argparse.ArgumentParser(description="Fatigue prediction from a raw sEMG CSV.")
    parser.add_argument("--csv", required=True, help="Path to the raw EMG CSV (any file name).")
    parser.add_argument("--model", default=None, help="Optional .pkl path (default: from metadata).")
    parser.add_argument("--outdir", default=None, help="Output folder (default: next to the CSV).")
    parser.add_argument("--peak-prominence", type=float, default=PEAK_PROMINENCE)
    args = parser.parse_args()

    table, ctx = predict_signal(args.csv, args.model, args.peak_prominence)

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(outdir, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(args.csv))[0])
    out_csv = os.path.join(outdir, f"{stem}_predictions.csv")
    out_png = os.path.join(outdir, f"{stem}_prediction_plot.png")
    table.to_csv(out_csv, index=False)
    save_plot(table, ctx, out_png)

    pct = 100 * (table["predicted_label"] == 1).mean()
    print("\n=== RESULTS ===")
    print(f"Model                : {ctx['model_name']} (features: {', '.join(ctx['features'])})")
    if ctx["cv_f1_mean"] is not None:
        print(f"LOSO reference       : macro-F1 {ctx['cv_f1_mean']:.3f} | ROC-AUC {ctx['cv_roc_auc_mean']:.3f}")
    print(f"Repetitions analysed : {len(table)} (discarded: {ctx['discarded']})")
    print(f"Fatigued repetitions : {pct:.1f}%")
    print(f"Mean P(fatigued)     : {table['probability_fatigued'].mean():.3f}")
    print(f"Predictions file     : {out_csv}")
    print(f"Plot file            : {out_png}")


if __name__ == "__main__":
    main()
