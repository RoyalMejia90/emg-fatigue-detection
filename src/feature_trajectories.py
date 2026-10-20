"""
Feature Trajectories

Author: RT, NM, NI
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# Configuration


ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    ROOT,
    "RESULTS"
)

FIGURES_DIR = os.path.join(
    ROOT,
    "FIGURES_RESULTS"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)

FEATURES_FILE = os.path.join(
    RESULTS_DIR,
    "features.csv"
)


# Load Data


df = pd.read_csv(
    FEATURES_FILE
)


# Normalize Repetition Position


normalized_rows = []

group_columns = [
    "subject_id",
    "file"
]

for _, group in df.groupby(
        group_columns):

    group = group.sort_values(
        "repetition"
    )

    total_repetitions = len(
        group
    )

    if total_repetitions < 2:
        continue

    progress = np.linspace(
        0,
        100,
        total_repetitions
    )

    group = group.copy()

    group[
        "progress_percent"
    ] = progress

    normalized_rows.append(
        group
    )

normalized_df = pd.concat(
    normalized_rows,
    ignore_index=True
)


# Common Scale


progress_bins = np.arange(
    0,
    101,
    5
)

interpolated_rows = []

for (
        subject_id,
        file_name
) , group in normalized_df.groupby(
    ["subject_id", "file"]
):

    group = group.sort_values(
        "progress_percent"
    )

    interp_rms = np.interp(
        progress_bins,
        group[
            "progress_percent"
        ],
        group["rms"]
    )

    interp_mdf = np.interp(
        progress_bins,
        group[
            "progress_percent"
        ],
        group["mdf"]
    )

    interp_mnf = np.interp(
        progress_bins,
        group[
            "progress_percent"
        ],
        group["mnf"]
    )

    temp_df = pd.DataFrame(
        {
            "progress_percent":
                progress_bins,

            "rms":
                interp_rms,

            "mdf":
                interp_mdf,

            "mnf":
                interp_mnf
        }
    )

    interpolated_rows.append(
        temp_df
    )

interpolated_df = pd.concat(
    interpolated_rows,
    ignore_index=True
)


# Summary Statistics


summary = (
    interpolated_df
    .groupby(
        "progress_percent"
    )
    .agg(
        {
            "rms":
                ["mean", "std"],

            "mdf":
                ["mean", "std"],

            "mnf":
                ["mean", "std"]
        }
    )
)

summary.columns = [
    "_".join(col)
    for col in summary.columns
]

summary = summary.reset_index()


# Figure 2


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 12)
)


# RMS


axes[0].plot(
    summary[
        "progress_percent"
    ],
    summary[
        "rms_mean"
    ],
    linewidth=2
)

axes[0].fill_between(
    summary[
        "progress_percent"
    ],
    summary[
        "rms_mean"
    ]
    -
    summary[
        "rms_std"
    ],
    summary[
        "rms_mean"
    ]
    +
    summary[
        "rms_std"
    ],
    alpha=0.3
)

axes[0].set_title(
    "RMS Across Normalized Set Progress"
)

axes[0].set_xlabel(
    "Set Progress (%)"
)

axes[0].set_ylabel(
    "RMS (mV)"
)

axes[0].grid(
    alpha=0.3
)


# MDF


axes[1].plot(
    summary[
        "progress_percent"
    ],
    summary[
        "mdf_mean"
    ],
    linewidth=2
)

axes[1].fill_between(
    summary[
        "progress_percent"
    ],
    summary[
        "mdf_mean"
    ]
    -
    summary[
        "mdf_std"
    ],
    summary[
        "mdf_mean"
    ]
    +
    summary[
        "mdf_std"
    ],
    alpha=0.3
)

axes[1].set_title(
    "MDF Across Normalized Set Progress"
)

axes[1].set_xlabel(
    "Set Progress (%)"
)

axes[1].set_ylabel(
    "Frequency (Hz)"
)

axes[1].grid(
    alpha=0.3
)


# MNF


axes[2].plot(
    summary[
        "progress_percent"
    ],
    summary[
        "mnf_mean"
    ],
    linewidth=2
)

axes[2].fill_between(
    summary[
        "progress_percent"
    ],
    summary[
        "mnf_mean"
    ]
    -
    summary[
        "mnf_std"
    ],
    summary[
        "mnf_mean"
    ]
    +
    summary[
        "mnf_std"
    ],
    alpha=0.3
)

axes[2].set_title(
    "MNF Across Normalized Set Progress"
)

axes[2].set_xlabel(
    "Set Progress (%)"
)

axes[2].set_ylabel(
    "Frequency (Hz)"
)

axes[2].grid(
    alpha=0.3
)


plt.tight_layout()

output_file = os.path.join(
    FIGURES_DIR,
    "Figure2_Feature_Trajectories_Normalized.png"
)

plt.savefig(
    output_file,
    dpi=300
)

plt.close()

print(
    f"\nFigure saved:\n{output_file}"
)