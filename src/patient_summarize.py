"""
Patient Signal Summary V4

Creates one diagnostic dashboard per trial.

Output:

FIGURES_PAT_SUMMARIZE/
    Pat01/
        Pat01_R1_dr_SUMMARIZE.png
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import welch
from scipy.signal import find_peaks


# Configuration


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROCESSED_DIR = os.path.join(
    ROOT,
    "PROCESSED"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "FIGURES_PAT_SUMMARIZE"
)

SAMPLING_RATE = 1000

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# Spectral Features


def compute_psd(signal):

    frequencies, power = welch(
        signal,
        fs=SAMPLING_RATE,
        nperseg=min(
            len(signal),
            2048
        )
    )

    return frequencies, power


def compute_mdf(frequencies, power):

    cumulative = np.cumsum(
        power
    )

    threshold = (
        cumulative[-1] / 2
    )

    idx = np.where(
        cumulative >= threshold
    )[0][0]

    return frequencies[idx]


def compute_mnf(frequencies, power):

    return np.sum(
        frequencies * power
    ) / np.sum(power)


# Main Pipeline


for patient_id in sorted(
        os.listdir(PROCESSED_DIR)):

    patient_path = os.path.join(
        PROCESSED_DIR,
        patient_id
    )

    if not os.path.isdir(patient_path):
        continue

    patient_output = os.path.join(
        OUTPUT_DIR,
        patient_id
    )

    os.makedirs(
        patient_output,
        exist_ok=True
    )

    print(
        f"\nProcessing {patient_id}"
    )

    for file in sorted(
            os.listdir(patient_path)):

        if not file.endswith(".csv"):
            continue

        try:

            file_path = os.path.join(
                patient_path,
                file
            )

            df = pd.read_csv(
                file_path
            )

            required_columns = [
                "Relative_Time_s",
                "Voltage_mV",
                "Filtered_mV",
                "RMS_mV"
            ]

            if not all(
                column in df.columns
                for column in required_columns
            ):
                continue

            time_vector = df[
                "Relative_Time_s"
            ].values

            raw_signal = df[
                "Voltage_mV"
            ].values

            filtered_signal = df[
                "Filtered_mV"
            ].values

            rms_signal = df[
                "RMS_mV"
            ].values

            peaks, _ = find_peaks(
                rms_signal,
                distance=800,
                prominence=10,
                width=50
            )

            frequencies, power = compute_psd(
                filtered_signal
            )

            mdf = compute_mdf(
                frequencies,
                power
            )

            mnf = compute_mnf(
                frequencies,
                power
            )

            repetition_count = len(
                peaks
            )

            figure_name = (
                file.replace(
                    "_filtered.csv",
                    "_SUMMARIZE.png"
                )
            )

            output_file = os.path.join(
                patient_output,
                figure_name
            )

            parts = figure_name.replace(
                "_SUMMARIZE.png",
                ""
            ).split("_")

            subject = parts[0]

            trial = parts[1]

            leg = (
                "Right Leg"
                if parts[2].lower() == "dr"
                else "Left Leg"
            )

            fig, axes = plt.subplots(
                4,
                1,
                figsize=(14, 12)
            )

            fig.suptitle(
                f"{subject} | {trial} | {leg}",
                fontsize=15,
                fontweight="bold"
            )

            # Raw Signal

            axes[0].plot(
                time_vector,
                raw_signal,
                color="tab:blue",
                linewidth=0.6
            )

            axes[0].set_title(
                "Raw Signal"
            )

            axes[0].set_ylabel(
                "Amplitude (mV)"
            )

            axes[0].grid(
                alpha=0.3
            )

            # Filtered Signal

            axes[1].plot(
                time_vector,
                filtered_signal,
                color="tab:green",
                linewidth=0.6
            )

            axes[1].set_title(
                "Filtered Signal"
            )

            axes[1].set_ylabel(
                "Amplitude (mV)"
            )

            axes[1].grid(
                alpha=0.3
            )

            # RMS

            axes[2].plot(
                time_vector,
                rms_signal,
                color="tab:orange",
                linewidth=1.4
            )

            axes[2].scatter(
                time_vector[peaks],
                rms_signal[peaks],
                color="red",
                s=30,
                zorder=5
            )

            for repetition_id, peak in enumerate(
                    peaks,
                    start=1):

                axes[2].annotate(
                    str(repetition_id),
                    (
                        time_vector[peak],
                        rms_signal[peak]
                    ),
                    fontsize=7
                )

            axes[2].text(
                0.98,
                0.92,
                (
                    f"Detected Repetitions: "
                    f"{repetition_count}"
                ),
                transform=axes[2].transAxes,
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round",
                    fc="white",
                    alpha=0.8
                )
            )

            axes[2].set_title(
                "RMS Envelope + Peaks"
            )

            axes[2].set_ylabel(
                "RMS (mV)"
            )

            axes[2].grid(
                alpha=0.3
            )

            # PSD

            axes[3].axvspan(
                20,
                150,
                color="lightgray",
                alpha=0.15
            )

            axes[3].plot(
                frequencies,
                power,
                color="tab:purple",
                linewidth=1.2
            )

            axes[3].axvline(
                mdf,
                color="red",
                linestyle="--",
                linewidth=1.5
            )

            axes[3].axvline(
                mnf,
                color="green",
                linestyle="--",
                linewidth=1.5
            )

            axes[3].text(
                0.98,
                0.92,
                (
                    f"MDF = {mdf:.1f} Hz\n"
                    f"MNF = {mnf:.1f} Hz"
                ),
                transform=axes[3].transAxes,
                ha="right",
                va="top",
                bbox=dict(
                    boxstyle="round",
                    fc="white",
                    alpha=0.85
                )
            )

            axes[3].set_xlim(
                0,
                150
            )

            axes[3].set_title(
                "Power Spectral Density (Welch)"
            )

            axes[3].set_xlabel(
                "Frequency (Hz)"
            )

            axes[3].set_ylabel(
                "Power"
            )

            axes[3].grid(
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"  OK -> {figure_name}"
            )

        except Exception as error:

            print(
                f"  ERROR -> {file}"
            )

            print(error)

print(
    "\nPatient summaries completed."
)