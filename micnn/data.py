"""Loading, windowing, and scaling for the multimodal VR postural-state data.

The expected input is one wide CSV of synchronised 60 Hz samples. Each row is a
single time sample with 1003 sensor columns (Xsens full-body kinematics,
Shimmer inertial/physiological channels, EMG) plus the metadata columns listed
in :data:`micnn.config.META_COLUMNS`.

Two windowing functions are provided:

* :func:`create_windows` — the standard pipeline used for the LOPO results.
* :func:`create_scenario_windows` — adds a perturbation-category tag per window
  and a purity filter, used for the leave-one-scenario-out analysis.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DROP_COLUMNS, META_COLUMNS
from .scenarios import LABEL_TO_CATEGORY

CONTROL_CATEGORY = "CONTROL"


def time_to_seconds(time_str) -> float:
    """Convert an ``"HH:MM:SS.mmm"`` timestamp to seconds since midnight.

    Returns ``NaN`` for missing or unparseable values so that malformed rows can
    be detected rather than silently reordered.
    """
    if pd.isna(time_str):
        return np.nan
    try:
        hours, minutes, seconds = str(time_str).split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return np.nan


def load_dataset(csv_path: str, drop_columns: Sequence[str] = DROP_COLUMNS) -> pd.DataFrame:
    """Load the prepared CSV and add the derived ``Timestamp_seconds`` column.

    The CSV is expected to be *already class-balanced* by participant-wise
    downsampling (all imbalanced samples kept, an equal number of balanced
    samples retained per participant). See :func:`downsample_by_participant`.

    Args:
        csv_path: Path to the prepared dataset CSV.
        drop_columns: Sensor columns to remove before modelling.

    Returns:
        The loaded DataFrame, with ``Timestamp_seconds`` added.
    """
    df = pd.read_csv(csv_path)
    df.drop(columns=list(drop_columns), inplace=True, errors="ignore")
    df["Timestamp_seconds"] = df["Timestamp_Xsens"].apply(time_to_seconds)
    return df


def downsample_by_participant(
    df: pd.DataFrame, random_state: int = 42, participant_col: str = "Participant"
) -> pd.DataFrame:
    """Balance the classes within each participant.

    The raw recordings are heavily skewed toward the non-perturbed class (roughly
    5:1 overall, and between 3.3:1 and 8:1 for individual participants). For each
    participant the smaller class is kept in full and an equal number of samples
    is drawn without replacement from the larger one; in this dataset class 1
    (imbalanced) is always the smaller class, so every class-1 sample is retained.

    Balancing *per participant* rather than globally preserves each individual's
    share of the dataset, so inter-individual variability is not distorted by the
    resampling.

    Caveat, because it affects how the windows should be interpreted: this drops
    individual *samples*, and windowing then runs over what remains. Perturbed
    windows stay contiguous (almost no class-1 samples are dropped), but control
    windows become temporally dilated — 100 retained rows can span far more than
    100 frames of wall clock. The published pipeline balances before windowing, so
    this function is applied in that order here too; windowing first and balancing
    at the window level would avoid the dilation. See the README's reproducibility
    notes.
    """
    balanced = []
    for participant, group in df.groupby(participant_col, sort=True):
        positives = group[group["class"] == 1]
        negatives = group[group["class"] == 0]
        n = min(len(positives), len(negatives))
        balanced.append(positives.sample(n=n, random_state=random_state))
        balanced.append(negatives.sample(n=n, random_state=random_state))
    return pd.concat(balanced, ignore_index=True)


def feature_columns(df: pd.DataFrame, meta_columns: Sequence[str] = META_COLUMNS) -> List[str]:
    """Return the sensor feature columns, i.e. everything that is not metadata."""
    return [c for c in df.columns if c not in set(meta_columns)]


def create_windows(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
    participant_col: str = "Participant",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Cut fixed-length sliding windows, independently per participant.

    Windows are generated *within* each participant only, so no window ever spans
    two people. Combined with participant-wise splitting at evaluation time, this
    means overlapping windows can never straddle the train/test boundary.

    Each window inherits a single label by majority vote over its samples.

    Args:
        df: DataFrame from :func:`load_dataset`.
        window_size: Number of consecutive samples per window (100 = ~1.67 s).
        step_size: Stride between window starts (50 = 50% overlap;
            set equal to ``window_size`` for non-overlapping windows).
        participant_col: Name of the participant identifier column.
        verbose: Print per-participant window counts.

    Returns:
        ``(X, y, participant_ids, feature_names)`` where ``X`` has shape
        ``(n_windows, window_size, n_features)`` as float32.
    """
    features = feature_columns(df)
    if verbose:
        print(f"Feature columns: {len(features)} | window={window_size} step={step_size}")

    windows, labels, participants = [], [], []

    for participant in sorted(df[participant_col].unique()):
        # Sort by time so that consecutive rows really are consecutive samples.
        subset = (
            df[df[participant_col] == participant][["Timestamp_seconds", "class"] + features]
            .sort_values("Timestamp_seconds")
            .reset_index(drop=True)
        )
        values = subset[features].values.astype(np.float32)
        classes = subset["class"].values.astype(np.int64)

        count = 0
        for start in range(0, len(subset) - window_size + 1, step_size):
            end = start + window_size
            windows.append(values[start:end])
            labels.append(int(np.bincount(classes[start:end]).argmax()))
            participants.append(participant)
            count += 1

        if verbose:
            print(f"  {participant}: {count} windows from {len(subset)} samples")
        del subset, values, classes

    if not windows:
        raise ValueError(
            f"No windows produced: no participant has at least window_size={window_size} "
            "samples. Check that the CSV is not empty and that window_size is not larger "
            "than the shortest recording."
        )

    X = np.asarray(windows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    participant_ids = np.asarray(participants)

    if verbose:
        print(f"\nTotal windows: {len(X):,}  shape={X.shape}  ({X.nbytes / 1024 ** 3:.2f} GB)")
        print(f"Class 0: {int((y == 0).sum()):,} | Class 1: {int((y == 1).sum()):,}")

    return X, y, participant_ids, features


def create_scenario_windows(
    df: pd.DataFrame,
    window_size: int,
    step_size: int,
    participant_col: str = "Participant",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Window the data and tag each window with its perturbation category.

    This is the windowing used for leave-one-scenario-out evaluation, and it adds
    a **purity filter** on top of :func:`create_windows`:

    * *positive* window — every non-zero label inside it maps to a single
      perturbation category **and** the majority class is 1;
    * *negative* window — every label inside it is 0 (pure walking / control);
    * anything else — a window straddling a perturbation onset/offset, or mixing
      two categories — is **dropped**.

    Dropping mixed windows is what makes the held-out-scenario test honest: a
    window that contains part of a training category and part of the held-out
    category would otherwise leak the held-out scenario into training.

    Each positive window also records its dominant original perturbation label,
    which allows a per-sub-scenario breakdown afterwards with no extra training.

    Returns:
        ``(X, y, participant_ids, categories, sublabels)``. ``categories`` holds
        the category name per window (``"CONTROL"`` for negatives) and
        ``sublabels`` holds the dominant perturbation id (0 for controls).
    """
    features = feature_columns(df)
    if verbose:
        print(f"Feature columns: {len(features)} | window={window_size} step={step_size}")

    windows, labels, participants, categories, sublabels = [], [], [], [], []
    n_dropped = 0

    for participant in sorted(df[participant_col].unique()):
        subset = (
            df[df[participant_col] == participant][
                ["Timestamp_seconds", "class", "Label"] + features
            ]
            .sort_values("Timestamp_seconds")
            .reset_index(drop=True)
        )
        values = subset[features].values.astype(np.float32)
        classes = subset["class"].values.astype(np.int64)
        raw_labels = subset["Label"].values.astype(np.int64)

        n_pos = n_neg = 0
        for start in range(0, len(subset) - window_size + 1, step_size):
            end = start + window_size
            window_labels = raw_labels[start:end]
            perturbed = window_labels[window_labels > 0]

            if perturbed.size == 0:
                # Pure control window: no perturbation frame anywhere inside it.
                category, window_class, dominant = CONTROL_CATEGORY, 0, 0
                n_neg += 1
            else:
                present = {LABEL_TO_CATEGORY[int(v)] for v in perturbed}
                window_class = int(np.bincount(classes[start:end]).argmax())
                if len(present) == 1 and window_class == 1:
                    category = next(iter(present))
                    dominant = int(np.bincount(perturbed).argmax())
                    n_pos += 1
                else:
                    n_dropped += 1  # transition or mixed-category window
                    continue

            windows.append(values[start:end])
            labels.append(window_class)
            participants.append(participant)
            categories.append(category)
            sublabels.append(dominant)

        if verbose:
            print(f"  {participant}: +{n_pos} positive / {n_neg} control")
        del subset, values, classes, raw_labels

    if not windows:
        raise ValueError(
            f"No windows survived the purity filter with window_size={window_size}. "
            "Check the Label column and that recordings are longer than one window."
        )

    X = np.asarray(windows, dtype=np.float32)
    if verbose:
        print(f"\nKept {len(X):,} windows (dropped {n_dropped:,} transition/mixed windows)")
        print(f"X shape: {X.shape}  ({X.nbytes / 1024 ** 3:.2f} GB)")

    return (
        X,
        np.asarray(labels, dtype=np.int32),
        np.asarray(participants),
        np.asarray(categories),
        np.asarray(sublabels, dtype=np.int64),
    )


def scale_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Standardise features using statistics from the training fold only.

    The scaler is fit on the training windows and merely applied to the test
    windows: fitting on the pooled data would let test-fold statistics inform
    training and inflate the reported scores.

    Standardisation is per feature channel, computed over all (window, timestep)
    pairs, so the temporal structure within a window is preserved.
    """
    from sklearn.preprocessing import StandardScaler

    n_features = X_train.shape[-1]
    n_timesteps = X_train.shape[1]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
    test_scaled = scaler.transform(X_test.reshape(-1, n_features))

    return (
        train_scaled.reshape(-1, n_timesteps, n_features).astype(np.float32),
        test_scaled.reshape(-1, n_timesteps, n_features).astype(np.float32),
    )
