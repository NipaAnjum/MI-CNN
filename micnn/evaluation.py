"""Subject-independent and scenario-independent evaluation protocols.

Two protocols are implemented:

``run_lopo``
    Leave-One-Participant-Out. Each fold holds out one participant entirely and
    trains on the rest, so every score is measured on a person the model has
    never seen. This is the headline protocol in the paper. Standard k-fold is
    not used because windows from the same participant would land in both the
    training and test sets, which inflates accuracy.

``run_loso``
    Leave-One-Scenario-Out. Each fold holds out one whole perturbation category,
    testing whether the model recognises a *kind of instability it was never
    trained on* rather than memorising scenario-specific structure.

Both protocols fit the feature scaler on the training fold only and rebuild the
model from scratch each fold, so no state leaks between folds.
"""

from __future__ import annotations

import gc
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config import Config, DEFAULT_CONFIG
from .data import CONTROL_CATEGORY, scale_train_test
from .model import build_micnn, compile_micnn

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "auc"]


def _callbacks(config: Config) -> List[tf.keras.callbacks.Callback]:
    """Early stopping on validation loss plus LR decay, as used for all results."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=0,
        ),
    ]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Accuracy, weighted precision/recall/F1, and ROC-AUC for one fold.

    Precision, recall, and F1 are weighted by class support. ``auc`` is NaN when
    a fold's test set contains only one class, which ROC-AUC is undefined for.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
    }


def summarize(results: pd.DataFrame, metric_columns: Sequence[str] = METRIC_COLUMNS) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds."""
    return pd.DataFrame(
        {
            "mean": results[list(metric_columns)].mean(),
            "std": results[list(metric_columns)].std(),
        }
    )


def append_summary_rows(
    results: pd.DataFrame,
    fold_column: str,
    metric_columns: Sequence[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Append MEAN and STD rows so a single CSV carries both folds and summary."""
    stats = summarize(results, metric_columns)
    mean_row = {fold_column: "MEAN"}
    std_row = {fold_column: "STD"}
    for metric in metric_columns:
        mean_row[metric] = stats.loc[metric, "mean"]
        std_row[metric] = stats.loc[metric, "std"]
    return pd.concat([results, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def _fit_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Config,
    class_weight: Optional[Dict[int, float]] = None,
    verbose: int = 0,
) -> tf.keras.Model:
    """Build, compile, and fit a fresh MI-CNN on one fold's training data.

    Note on ``config.val_split``: Keras carves the validation set from the **last**
    rows of the arrays *before* shuffling, and the fold arrays here are ordered
    rather than shuffled. Under LOPO that makes the validation set the tail of the
    last training participant; under leave-one-scenario-out, where rows are ordered
    positives-then-controls, it is entirely control windows. This matches the
    published runs and is kept for that reason — passing an explicit stratified
    ``validation_data`` here would change the reported numbers. See the README's
    reproducibility notes.
    """
    # Clearing the session and reseeding keeps folds independent and repeatable.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.random_seed)

    model = build_micnn(input_shape=X_train.shape[1:], **config.model_kwargs())
    compile_micnn(model, learning_rate=config.learning_rate)
    model.fit(
        X_train,
        y_train,
        validation_split=config.val_split,
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weight,
        callbacks=_callbacks(config),
        verbose=verbose,
    )
    return model


def run_lopo(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = True,
    on_fold_end: Optional[Callable[[str, tf.keras.Model], None]] = None,
) -> pd.DataFrame:
    """Leave-One-Participant-Out cross-validation.

    Args:
        X: Unscaled windows, ``(n_windows, timesteps, features)``.
        y: Window labels, ``(n_windows,)``.
        participant_ids: Participant identifier per window, ``(n_windows,)``.
        config: Hyperparameters.
        verbose: Print per-fold progress.
        on_fold_end: Optional callback ``(participant, fitted_model)``, e.g. to
            save a per-fold checkpoint.

    Returns:
        One row per held-out participant with fold sizes and metrics.
    """
    participants = sorted(np.unique(participant_ids))
    results = []

    for participant in participants:
        if verbose:
            print(f"\n--- LOPO fold: held-out participant = {participant} ---")

        test_mask = participant_ids == participant
        X_train_raw, y_train = X[~test_mask], y[~test_mask].astype(np.int32)
        X_test_raw, y_test = X[test_mask], y[test_mask].astype(np.int32)

        X_train, X_test = scale_train_test(X_train_raw, X_test_raw)
        model = _fit_fold(X_train, y_train, config)

        proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(proba, axis=1)

        metrics = compute_metrics(y_test, y_pred, proba[:, 1])
        results.append(
            {
                "Participant": participant,
                "n_train_windows": int(len(y_train)),
                "n_test_windows": int(len(y_test)),
                **metrics,
            }
        )

        if verbose:
            print(
                f"  accuracy={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f} "
                f"| auc={metrics['auc']:.4f}"
            )
        if on_fold_end is not None:
            on_fold_end(participant, model)

        del model, X_train, X_test, X_train_raw, X_test_raw
        gc.collect()

    return pd.DataFrame(results)


def split_control_windows(
    participant_ids: np.ndarray,
    categories: np.ndarray,
    train_fraction: float = 0.8,
    guard: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split the control (no-perturbation) windows once, for reuse in every LOSO fold.

    The split is *contiguous in time* within each participant rather than random:
    windows overlap by 50%, so a random split would place two halves of the same
    stretch of signal on opposite sides of the train/test boundary. ``guard``
    windows are dropped at the cut for the same reason.

    Holding this split fixed across folds means the only thing that changes
    between folds is which perturbation category is held out.

    Args:
        participant_ids: Participant per window.
        categories: Category per window; control windows are ``"CONTROL"``.
        train_fraction: Fraction of each participant's control windows for training.
        guard: Number of windows discarded at the cut point.

    Returns:
        ``(train_indices, test_indices)`` into the full window arrays.
    """
    train_idx, test_idx = [], []
    for participant in sorted(np.unique(participant_ids)):
        # Windows were appended in time order, so this index array is time-ordered.
        idx = np.where((categories == CONTROL_CATEGORY) & (participant_ids == participant))[0]
        cut = int(len(idx) * train_fraction)
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut + guard :].tolist())
    return np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def run_loso(
    X: np.ndarray,
    y: np.ndarray,
    participant_ids: np.ndarray,
    categories: np.ndarray,
    sublabels: Optional[np.ndarray] = None,
    config: Config = DEFAULT_CONFIG,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-One-Scenario-Out cross-validation.

    In each fold, one perturbation category supplies the positive test windows and
    is removed from training entirely; the remaining categories supply the positive
    training windows. Control windows come from the fixed split produced by
    :func:`split_control_windows`.

    The headline number is ``scenario_recall``: the fraction of windows from the
    never-seen perturbation type that the model still flags as imbalanced.

    Class weights are applied because holding out a large category (the roll
    perturbations, for instance) leaves a heavily skewed training fold.

    Be aware that the early-stopping validation set in this protocol is entirely
    control windows (see :func:`_fit_fold`), which biases ``scenario_recall``
    downward — the reported detection rates are conservative.

    Returns:
        ``(fold_results, sublabel_records)``. The second frame has one row per
        held-out positive window with its fine perturbation label and prediction,
        enabling a per-sub-scenario breakdown without retraining. It is empty when
        ``sublabels`` is not supplied.
    """
    positive_categories = sorted(c for c in np.unique(categories) if c != CONTROL_CATEGORY)
    control_train_idx, control_test_idx = split_control_windows(participant_ids, categories)

    if verbose:
        print(
            f"Control windows -> train: {len(control_train_idx):,} "
            f"test: {len(control_test_idx):,}"
        )
        print(f"{len(positive_categories)} folds: {positive_categories}")

    results, sublabel_records = [], []

    for category in positive_categories:
        if verbose:
            print(f"\n=== Held-out scenario: {category} ===")

        train_pos = np.where((y == 1) & (categories != CONTROL_CATEGORY) & (categories != category))[0]
        test_pos = np.where((y == 1) & (categories == category))[0]

        train_idx = np.concatenate([train_pos, control_train_idx])
        # Held-out positives come first, so the first len(test_pos) rows of the
        # test set can be traced back to their fine perturbation labels below.
        test_idx = np.concatenate([test_pos, control_test_idx])

        X_train_raw, y_train = X[train_idx], y[train_idx].astype(np.int32)
        X_test_raw, y_test = X[test_idx], y[test_idx].astype(np.int32)

        if verbose:
            print(
                f"  train: {len(y_train):,} (pos {int((y_train == 1).sum()):,} / "
                f"neg {int((y_train == 0).sum()):,}) | "
                f"test: {len(y_test):,} (pos {int((y_test == 1).sum()):,} / "
                f"neg {int((y_test == 0).sum()):,})"
            )

        X_train, X_test = scale_train_test(X_train_raw, X_test_raw)

        present_classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=present_classes, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(present_classes, weights)}

        model = _fit_fold(X_train, y_train, config, class_weight=class_weight)

        proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(proba, axis=1)

        if sublabels is not None:
            n_pos = len(test_pos)
            for label, pred, score in zip(
                sublabels[test_idx][:n_pos], y_pred[:n_pos], proba[:n_pos, 1]
            ):
                sublabel_records.append(
                    {
                        "Scenario": category,
                        "Label": int(label),
                        "y_true": 1,
                        "y_pred": int(pred),
                        "proba_pos": float(score),
                    }
                )

        # Per-class recall: index 1 is the detection rate on the unseen scenario,
        # index 0 is the false-alarm complement on control windows.
        per_class_recall = recall_score(y_test, y_pred, average=None, labels=[0, 1], zero_division=0)
        metrics = compute_metrics(y_test, y_pred, proba[:, 1])
        results.append(
            {
                "Scenario": category,
                "n_test_pos": int((y_test == 1).sum()),
                "n_test_neg": int((y_test == 0).sum()),
                **metrics,
                "scenario_recall": float(per_class_recall[1]),
                "control_recall": float(per_class_recall[0]),
            }
        )

        if verbose:
            print(
                f"  scenario_recall={per_class_recall[1]:.4f} | "
                f"accuracy={metrics['accuracy']:.4f} | auc={metrics['auc']:.4f}"
            )

        del model, X_train, X_test, X_train_raw, X_test_raw
        gc.collect()

    return pd.DataFrame(results), pd.DataFrame(sublabel_records)


def sublabel_breakdown(sublabel_records: pd.DataFrame) -> pd.DataFrame:
    """Detection rate per fine perturbation label inside each held-out category.

    Every row in ``sublabel_records`` is a true positive, so the mean prediction
    is exactly the recall for that sub-scenario. This is what shows *which*
    disturbances inside a weak category the model actually misses.
    """
    from .scenarios import LABEL_NAMES

    if sublabel_records.empty:
        return sublabel_records

    records = sublabel_records.copy()
    # Fill unknown ids rather than leaving NaN: groupby would silently drop those
    # rows, losing windows from the breakdown without any warning.
    records["Label_name"] = (
        records["Label"].map(LABEL_NAMES).fillna("Unknown label " + records["Label"].astype(str))
    )
    return (
        records.groupby(["Scenario", "Label", "Label_name"])
        .agg(
            n_windows=("y_pred", "size"),
            recall=("y_pred", "mean"),
            mean_proba=("proba_pos", "mean"),
        )
        .reset_index()
        .sort_values(["Scenario", "recall"])
    )
