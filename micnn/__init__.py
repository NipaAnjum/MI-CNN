"""MI-CNN: Mamba-Inspired CNN for postural state classification in immersive VR.

Reference implementation for:

    Toward Postural State Classification in Immersive VR with Multimodal Data
    and Explainability Analysis.
    IEEE International Symposium on Mixed and Augmented Reality (ISMAR), 2026.

Released under the MIT License.

Typical use::

    from micnn import Config, build_micnn, load_dataset, create_windows, run_lopo

    config = Config()
    df = load_dataset("final_dataset.csv")
    X, y, participants, features = create_windows(df, config.window_size, config.step_size)
    results = run_lopo(X, y, participants, config)
"""

from .config import DEFAULT_CONFIG, DROP_COLUMNS, META_COLUMNS, Config
from .data import (
    create_scenario_windows,
    create_windows,
    downsample_by_participant,
    feature_columns,
    load_dataset,
    scale_train_test,
    time_to_seconds,
)
from .evaluation import (
    append_summary_rows,
    compute_metrics,
    run_lopo,
    run_loso,
    split_control_windows,
    sublabel_breakdown,
    summarize,
)
from .model import MambaInspiredBlock, build_micnn, compile_micnn
from .scenarios import CATEGORIES, LABEL_NAMES, LABEL_TO_CATEGORY

__version__ = "1.0.0"

__all__ = [
    "Config",
    "DEFAULT_CONFIG",
    "META_COLUMNS",
    "DROP_COLUMNS",
    "MambaInspiredBlock",
    "build_micnn",
    "compile_micnn",
    "load_dataset",
    "downsample_by_participant",
    "create_windows",
    "create_scenario_windows",
    "feature_columns",
    "scale_train_test",
    "time_to_seconds",
    "run_lopo",
    "run_loso",
    "split_control_windows",
    "sublabel_breakdown",
    "compute_metrics",
    "summarize",
    "append_summary_rows",
    "LABEL_TO_CATEGORY",
    "LABEL_NAMES",
    "CATEGORIES",
]
