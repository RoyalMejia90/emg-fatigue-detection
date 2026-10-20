"""
Validation Dataset Generator

Author: RT, NM, NI
"""

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import LeaveOneGroupOut


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

FEATURES_FILE = os.path.join(
    RESULTS_DIR,
    "features.csv"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "fatigue_dataset.csv"
)


# Load Features


df = pd.read_csv(
    FEATURES_FILE
)


# Fatigue Labeling


labeled_trials = []

for (
        subject_id,
        file_name
) , group in df.groupby(
    ["subject_id", "file"]
):

    group = group.sort_values(
        "repetition"
    ).reset_index(drop=True)

    n_repetitions = len(
        group
    )

    if n_repetitions < 3:
        continue

    progress = np.linspace(
        0,
        100,
        n_repetitions
    )

    labels = []

    for p in progress:

        if p < 50:

            labels.append(
                0
            )

        else:

            labels.append(
                1
            )

    group[
        "progress_percent"
    ] = progress

    group[
        "fatigue_label"
    ] = labels

    labeled_trials.append(
        group
    )


fatigue_df = pd.concat(
    labeled_trials,
    ignore_index=True
)


# Class Distribution


distribution = (
    fatigue_df[
        "fatigue_label"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nClass Distribution"
)

print(
    distribution
)


# LOSO Validation


logo = LeaveOneGroupOut()

# Updated feature list including the 11 extended features
X = fatigue_df[
    [
        "rms",
        "mav",
        "variance",
        "waveform_length",
        "mdf",
        "mnf",
        "peak_frequency",
        "log_fins5",
        "low_high_ratio",
        "sample_entropy",
        "higuchi_fd"
    ]
]

y = fatigue_df[
    "fatigue_label"
]

groups = fatigue_df[
    "subject_id"
]


fold_rows = []

for fold_id, (
        train_idx,
        test_idx
) in enumerate(

    logo.split(
        X,
        y,
        groups
    ),

    start=1
):

    test_subject = (
        fatigue_df.iloc[
            test_idx
        ][
            "subject_id"
        ]
        .unique()[0]
    )

    fold_rows.append(
        {
            "fold":
                fold_id,

            "test_subject":
                test_subject,

            "train_samples":
                len(train_idx),

            "test_samples":
                len(test_idx)
        }
    )


folds_df = pd.DataFrame(
    fold_rows
)

folds_file = os.path.join(
    RESULTS_DIR,
    "loso_folds.csv"
)

folds_df.to_csv(
    folds_file,
    index=False
)


# Export Dataset


fatigue_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nDataset saved:"
)

print(
    OUTPUT_FILE
)

print(
    "\nLOSO folds saved:"
)

print(
    folds_file
)

print(
    f"\nSubjects: {groups.nunique()}"
)

print(
    f"LOSO folds: {len(folds_df)}"
)

print(
    "\nValidation dataset completed."
)