"""
EMG Feature Extraction (Extended - 11 features, 3 families)

Author: RT, NM, NI
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import welch
import antropy as ant

# Configuration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "PROCESSED")
RESULTS_DIR = os.path.join(ROOT, "RESULTS")
REPETITION_FILE = os.path.join(RESULTS_DIR, "repetition_summary.csv")

SAMPLING_RATE = 1000
MIN_SEGMENT = 500
BAND = (20, 450)
LOW_BAND = (20, 45)
HIGH_BAND = (95, 450)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Time-Domain Features

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def compute_mav(signal):
    return np.mean(np.abs(signal))

def compute_variance(signal):
    return np.var(signal)

def compute_waveform_length(signal):
    return np.sum(np.abs(np.diff(signal)))

# Frequency-Domain Features

def compute_psd(signal):
    frequencies, power = welch(
        signal, fs=SAMPLING_RATE, nperseg=min(len(signal), 1024)
    )
    return frequencies, power

def compute_mdf(signal):
    frequencies, power = compute_psd(signal)
    cumulative = np.cumsum(power)
    threshold = cumulative[-1] / 2
    index = np.where(cumulative >= threshold)[0][0]
    return frequencies[index]

def compute_mnf(signal):
    frequencies, power = compute_psd(signal)
    return np.sum(frequencies * power) / np.sum(power)

def compute_peak_frequency(signal):
    frequencies, power = compute_psd(signal)
    return frequencies[np.argmax(power)]

def compute_log_fins5(signal):
    frequencies, power = compute_psd(signal)
    m = (frequencies >= BAND[0]) & (frequencies <= BAND[1])
    if not np.any(m) or np.sum(power[m] * frequencies[m] ** 5) == 0:
        return np.nan
    fins5 = np.sum(power[m] / frequencies[m]) / np.sum(power[m] * frequencies[m] ** 5)
    return np.log10(fins5) if fins5 > 0 else np.nan

def compute_low_high_ratio(signal):
    frequencies, power = compute_psd(signal)
    low = power[(frequencies >= LOW_BAND[0]) & (frequencies <= LOW_BAND[1])].sum()
    high = power[(frequencies >= HIGH_BAND[0]) & (frequencies <= HIGH_BAND[1])].sum()
    return low / high if high > 0 else np.nan

# Nonlinear Features

def compute_sample_entropy(signal):
    try:
        return ant.sample_entropy(signal, order=2)
    except Exception:
        return np.nan

def compute_higuchi_fd(signal):
    try:
        return ant.higuchi_fd(signal, kmax=10)
    except Exception:
        return np.nan

# Main Pipeline

def main():
    repetition_df = pd.read_csv(REPETITION_FILE)
    feature_rows = []

    for subject_id in sorted(os.listdir(PROCESSED_DIR)):
        subject_path = os.path.join(PROCESSED_DIR, subject_id)
        if not os.path.isdir(subject_path):
            continue

        print(f"\nProcessing {subject_id}")
        subject_summary = repetition_df[repetition_df["subject_id"] == subject_id]

        for file in sorted(os.listdir(subject_path)):
            if not file.endswith(".csv"):
                continue

            trial_summary = subject_summary[subject_summary["file"] == file]
            if len(trial_summary) < 2:
                continue

            file_path = os.path.join(subject_path, file)
            df = pd.read_csv(file_path)
            signal = df["Filtered_mV"].values
            peak_times = trial_summary["peak_time_s"].values
            peak_indices = (peak_times * SAMPLING_RATE).astype(int)

            boundaries = [0]
            for i in range(len(peak_indices) - 1):
                midpoint = int((peak_indices[i] + peak_indices[i + 1]) / 2)
                boundaries.append(midpoint)
            boundaries.append(len(signal))

            for repetition in range(len(boundaries) - 1):
                start_idx = boundaries[repetition]
                end_idx = boundaries[repetition + 1]
                segment = signal[start_idx:end_idx]

                if len(segment) < MIN_SEGMENT:
                    continue

                feature_rows.append({
                    "subject_id": subject_id,
                    "file": file,
                    "repetition": repetition + 1,
                    "samples": len(segment),
                    "rms": compute_rms(segment),
                    "mav": compute_mav(segment),
                    "variance": compute_variance(segment),
                    "waveform_length": compute_waveform_length(segment),
                    "mdf": compute_mdf(segment),
                    "mnf": compute_mnf(segment),
                    "peak_frequency": compute_peak_frequency(segment),
                    "log_fins5": compute_log_fins5(segment),
                    "low_high_ratio": compute_low_high_ratio(segment),
                    "sample_entropy": compute_sample_entropy(segment),
                    "higuchi_fd": compute_higuchi_fd(segment),
                })

    features_df = pd.DataFrame(feature_rows)
    features_file = os.path.join(RESULTS_DIR, "features.csv")
    features_df.to_csv(features_file, index=False)

    print(f"\nFeatures saved to:\n{features_file}")
    print(f"Total windows: {len(features_df)}")
    print("Feature extraction completed.")

if __name__ == "__main__":
    main()