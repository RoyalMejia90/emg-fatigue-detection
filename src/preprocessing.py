"""
EMG Preprocessing Pipeline

Author: RT, NM, NI
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import iirnotch

warnings.filterwarnings("ignore")


# Configuration


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(ROOT, "DATA")
PROCESSED_DIR = os.path.join(ROOT, "PROCESSED")
FIGURES_DIR = os.path.join(ROOT, "FIGURES")
META_DIR = os.path.join(ROOT, "META")

FS = 1000
POWERLINE_FREQ = 60
RMS_WINDOW_MS = 250

for directory in [
    PROCESSED_DIR,
    FIGURES_DIR,
    META_DIR
]:
    os.makedirs(directory, exist_ok=True)


# Subject Mapping


PATIENT_MAP = {
    "Carolina_Cordoba": "Pat01",
    "Chelsy_Quevedo": "Pat02",
    "Daniela_Fernandez": "Pat03",
    "Dennis_Benavides": "Pat04",
    "Emilia_Lucas": "Pat05",
    "Erick_Yamberla": "Pat06",
    "Estefania_Paucar": "Pat07",
    "Ethan_Guevara": "Pat08",
    "Kennedy_Rodriguez": "Pat09",
    "Lady_Salazar": "Pat10",
    "Mateo_Gavilanes": "Pat11",
    "Mayerlin_Solorzano": "Pat12",
    "Natasha_Ipiales": "Pat13",
    "Nathalia_Marquez": "Pat14",
    "Rommel_Tuquerres": "Pat15"
}


# Filters


def remove_dc(signal):
    return signal - np.mean(signal)


def highpass_filter(signal, cutoff=1):

    nyquist = FS / 2

    b, a = butter(
        4,
        cutoff / nyquist,
        btype="highpass"
    )

    return filtfilt(b, a, signal)


def bandpass_filter(signal,
                    lowcut=20,
                    highcut=450):

    nyquist = FS / 2

    b, a = butter(
        4,
        [lowcut / nyquist,
         highcut / nyquist],
        btype="bandpass"
    )

    return filtfilt(b, a, signal)


def notch_filter(signal,
                 freq=POWERLINE_FREQ,
                 q=30):

    b, a = iirnotch(
        freq,
        q,
        FS
    )

    return filtfilt(b, a, signal)


def preprocess_signal(signal):

    filtered_signal = remove_dc(
        signal
    )

    filtered_signal = highpass_filter(
        filtered_signal
    )

    filtered_signal = bandpass_filter(
        filtered_signal
    )

    filtered_signal = notch_filter(
        filtered_signal
    )

    return filtered_signal


# RMS Envelope


def moving_rms(signal,
               window_ms=RMS_WINDOW_MS):

    window_samples = int(
        window_ms * FS / 1000
    )

    squared_signal = signal ** 2

    rms_signal = np.sqrt(
        np.convolve(
            squared_signal,
            np.ones(window_samples)
            / window_samples,
            mode="same"
        )
    )

    return rms_signal


# Visualization


def save_signal_plot(time_vector,
                     signal,
                     output_path,
                     title,
                     ylabel):

    plt.figure(
        figsize=(14, 4)
    )

    plt.plot(
        time_vector,
        signal,
        linewidth=0.5
    )

    plt.title(title)

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# Metadata


mapping_rows = []
quality_rows = []


# Main Pipeline


for subject_name in sorted(os.listdir(DATA_DIR)):

    subject_path = os.path.join(
        DATA_DIR,
        subject_name
    )

    if not os.path.isdir(subject_path):
        continue

    if subject_name not in PATIENT_MAP:
        continue

    subject_id = PATIENT_MAP[
        subject_name
    ]

    csv_files = [
        file
        for file in os.listdir(subject_path)
        if file.endswith(".csv")
    ]

    status = "OK"

    if len(csv_files) < 10:
        status = "PARTIAL"

    if subject_id == "Pat09":
        status = "SPECIAL_R0"

    mapping_rows.append({
        "patient_id": subject_id,
        "original_name": subject_name
    })

    quality_rows.append({
        "patient_id": subject_id,
        "file_count": len(csv_files),
        "status": status
    })

    processed_subject_dir = os.path.join(
        PROCESSED_DIR,
        subject_id
    )

    figures_subject_dir = os.path.join(
        FIGURES_DIR,
        subject_id
    )

    os.makedirs(
        processed_subject_dir,
        exist_ok=True
    )

    os.makedirs(
        figures_subject_dir,
        exist_ok=True
    )

    print(f"\nProcessing {subject_id}")

    for file in sorted(csv_files):

        if file.upper().startswith("R0"):
            continue

        try:

            file_path = os.path.join(
                subject_path,
                file
            )

            df = pd.read_csv(
                file_path
            )

            required_columns = [
                "Relative_Time_s",
                "ADC_Value",
                "Voltage_mV"
            ]

            if not all(
                column in df.columns
                for column in required_columns
            ):
                continue

            time_vector = df[
                "Relative_Time_s"
            ].values

            signal = df[
                "Voltage_mV"
            ].values

            filtered_signal = preprocess_signal(
                signal
            )

            rms_signal = moving_rms(
                filtered_signal
            )

            rms_signal = pd.Series(
                rms_signal
            ).rolling(
                window=200,
                center=True,
                min_periods=1
            ).mean().values

            match = re.search(
                r"(R\d+)_(dr|iz)",
                file,
                re.IGNORECASE
            )

            if match:

                repetition = match.group(1)
                leg = match.group(2)

                base_name = (
                    f"{subject_id}_"
                    f"{repetition}_"
                    f"{leg}"
                )

            else:

                base_name = file.replace(
                    ".csv",
                    ""
                )

            output_df = pd.DataFrame(
                {
                    "Relative_Time_s":
                        time_vector,

                    "ADC_Value":
                        df["ADC_Value"],

                    "Voltage_mV":
                        signal,

                    "Filtered_mV":
                        filtered_signal,

                    "RMS_mV":
                        rms_signal
                }
            )

            output_csv = os.path.join(
                processed_subject_dir,
                f"{base_name}_filtered.csv"
            )

            output_df.to_csv(
                output_csv,
                index=False
            )

            raw_figure = os.path.join(
                figures_subject_dir,
                f"{base_name}_raw.png"
            )

            filtered_figure = os.path.join(
                figures_subject_dir,
                f"{base_name}_filtered.png"
            )

            rms_figure = os.path.join(
                figures_subject_dir,
                f"{base_name}_rms.png"
            )

            save_signal_plot(
                time_vector,
                signal,
                raw_figure,
                f"{base_name} Raw Signal",
                "Amplitude (mV)"
            )

            save_signal_plot(
                time_vector,
                filtered_signal,
                filtered_figure,
                f"{base_name} Filtered Signal",
                "Amplitude (mV)"
            )

            save_signal_plot(
                time_vector,
                rms_signal,
                rms_figure,
                f"{base_name} RMS Envelope",
                "RMS (mV)"
            )

            print(
                f"  OK -> {base_name}"
            )

        except Exception as error:

            print(
                f"  ERROR -> {file}"
            )

            print(error)


# Export Metadata


mapping_df = pd.DataFrame(
    mapping_rows
)

mapping_df.to_csv(
    os.path.join(
        META_DIR,
        "metadata_mapping.csv"
    ),
    index=False
)

quality_df = pd.DataFrame(
    quality_rows
)

quality_df.to_csv(
    os.path.join(
        META_DIR,
        "data_quality.csv"
    ),
    index=False
)

print("\nPreprocessing completed.")