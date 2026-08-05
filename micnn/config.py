"""Default configuration — these are the exact values used for the reported results.

Every experiment in the paper (LOPO, leave-one-scenario-out, the non-overlapping
robustness check) uses this single configuration; only the field noted below
changes between them.
"""

from dataclasses import dataclass, field
from typing import Tuple


# Metadata columns in the source CSV that are not model inputs. Everything else
# is treated as a sensor feature.
META_COLUMNS: Tuple[str, ...] = (
    "Participant",       # participant identifier, e.g. "Participant1"
    "Timestamp_Xsens",   # wall-clock timestamp string "HH:MM:SS.mmm"
    "Timestamp_seconds", # derived: timestamp in seconds (added during loading)
    "class",             # binary target: 0 = balanced, 1 = imbalanced
    "Label",             # original perturbation id 0-35 (0 = no perturbation)
    "Frame",             # Xsens frame counter
)

# Columns dropped before modelling. These two Shimmer accelerometer axes are
# missing for one participant in the public dataset, so they are removed for
# everyone to keep the feature space identical across participants. This is what
# takes the input width to the 1001 features reported in the paper.
DROP_COLUMNS: Tuple[str, ...] = ("Shimmer_Acc_y", "Shimmer_Acc_z")


@dataclass
class Config:
    """Windowing, architecture, and training hyperparameters."""

    # --- Windowing -----------------------------------------------------------
    # 100 samples at the dataset's 60 Hz sampling rate is ~1.67 s of signal.
    window_size: int = 100
    # 50 samples (~0.83 s) gives 50% overlap between adjacent windows. Set this
    # equal to window_size to reproduce the non-overlapping robustness analysis.
    step_size: int = 50

    # --- Architecture --------------------------------------------------------
    d_model: int = 64
    n_layers: int = 4
    d_state: int = 16
    expand_factor: int = 2
    dropout: float = 0.3
    num_classes: int = 2

    # --- Training ------------------------------------------------------------
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    # Fraction of each training fold held out for early stopping / LR scheduling.
    # Taken from the training folds only, never from the held-out test fold.
    val_split: float = 0.1
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    random_seed: int = 42

    def model_kwargs(self) -> dict:
        """Architecture arguments for :func:`micnn.model.build_micnn`."""
        return dict(
            num_classes=self.num_classes,
            d_model=self.d_model,
            n_layers=self.n_layers,
            d_state=self.d_state,
            expand_factor=self.expand_factor,
            dropout=self.dropout,
        )


DEFAULT_CONFIG = Config()
