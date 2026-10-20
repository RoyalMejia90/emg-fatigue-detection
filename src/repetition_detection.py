"""
EMG Repetition Detection

Author: RT, NM, NI
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

RESULTS_DIR = os.path.join(
    ROOT,
    "RESULTS"
)

FIGURES_DIR = os.path.join(
    ROOT,
    "FIGURES_REPETITIONS"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# Peak Detection Parameters


SAMPLING_RATE = 1000

MIN_DISTANCE_SECONDS = 0.8

MIN_DISTANCE = int(
    MIN_DISTANCE_SECONDS *
    SAMPLING_RATE
)

PROMINENCE = 10

WIDTH = 50


# Results


summary_rows = []


# Main Pipeline


for subject_id in sorted(
        os.listdir(PROCESSED_DIR)):

    subject_path = os.path.join(
        PROCESSED_DIR,
        subject_id
    )

    if not os.path.isdir(
            subject_path):
        continue

    print(
        f"\nProcessing {subject_id}"
    )

    subject_figure_dir = os.path.join(
        FIGURES_DIR,
        subject_id
    )

    os.makedirs(
        subject_figure_dir,
        exist_ok=True
    )

    for file in sorted(
            os.listdir(subject_path)):

        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(
            subject_path,
            file
        )

        try:

            df = pd.read_csv(
                file_path
            )

            if "RMS_mV" not in df.columns:
                continue

            time_vector = (
                df["Relative_Time_s"]
                .values
            )

            rms_signal = (
                df["RMS_mV"]
                .values
            )

            peaks, properties = find_peaks(
                rms_signal,
                distance=MIN_DISTANCE,
                prominence=PROMINENCE,
                width=WIDTH
            )

            for repetition_id, peak_index in enumerate(
                    peaks,
                    start=1):

                summary_rows.append(
                    {
                        "subject_id":
                            subject_id,

                        "file":
                            file,

                        "repetition":
                            repetition_id,

                        "peak_time_s":
                            float(
                                time_vector[
                                    peak_index
                                ]
                            ),

                        "peak_rms_mV":
                            float(
                                rms_signal[
                                    peak_index
                                ]
                            )
                    }
                )

            plt.figure(
                figsize=(14, 5)
            )

            plt.plot(
                time_vector,
                rms_signal,
                color="steelblue",
                linewidth=1.5
            )

            plt.scatter(
                time_vector[peaks],
                rms_signal[peaks],
                color="red",
                s=50,
                zorder=10
            )

            for peak_number, peak_index in enumerate(
                    peaks,
                    start=1):

                plt.annotate(
                    str(peak_number),
                    (
                        time_vector[
                            peak_index
                        ],
                        rms_signal[
                            peak_index
                        ]
                    ),
                    fontsize=8,
                    color="black"
                )

            plt.xlabel(
                "Time (s)"
            )

            plt.ylabel(
                "RMS (mV)"
            )

            plt.title(
                file.replace(
                    ".csv",
                    ""
                )
            )

            plt.grid(
                alpha=0.3
            )

            plt.tight_layout()

            output_figure = os.path.join(
                subject_figure_dir,
                file.replace(
                    ".csv",
                    "_peaks.png"
                )
            )

            plt.savefig(
                output_figure,
                dpi=300
            )

            plt.close()

            print(
                f"  Peaks detected: {len(peaks)}"
            )

        except Exception as error:

            print(
                f"  ERROR -> {file}"
            )

            print(error)


# Export Results


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "repetition_summary.csv"
    ),
    index=False
)

print(
    "\nRepetition detection completed."
)