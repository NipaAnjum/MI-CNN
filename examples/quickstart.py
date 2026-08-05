"""End-to-end smoke test on synthetic data - no dataset required.

Generates a small synthetic multimodal recording in the same format as the real
CSV (participants, timestamps, perturbation labels, sensor channels), then runs
the full pipeline: windowing, both evaluation protocols, and a parameter-count
check against the architecture reported in the paper.

It exists so that anyone can confirm the code runs before obtaining the dataset,
and so the shapes and interfaces are demonstrated concretely. The numbers it
prints are meaningless - the data is random.

Usage:
    python examples/quickstart.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micnn import (  # noqa: E402
    Config,
    build_micnn,
    create_scenario_windows,
    create_windows,
    run_lopo,
    run_loso,
    summarize,
)

# Small enough to run on a laptop CPU in a couple of minutes.
N_PARTICIPANTS = 3
N_SAMPLES = 3000          # time samples per participant
N_FEATURES = 24           # real dataset has 1001
N_BURST = 120             # samples per perturbation burst
SAMPLING_RATE = 60.0      # Hz, matching the real recordings


def make_synthetic_dataset(seed: int = 0) -> pd.DataFrame:
    """Build a DataFrame shaped like the real dataset, with a learnable signal.

    Perturbation blocks are injected as bursts with a different mean and variance
    so the model has something to find; a few perturbation categories are used so
    the leave-one-scenario-out path is exercised.
    """
    rng = np.random.default_rng(seed)
    # One fine label per category, so every LOSO fold has data.
    category_labels = [1, 20, 22, 29, 24, 31, 30]
    frames = []

    for p in range(1, N_PARTICIPANTS + 1):
        features = rng.normal(0.0, 1.0, size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
        labels = np.zeros(N_SAMPLES, dtype=np.int64)

        # Insert one perturbation burst per category, spaced through the recording,
        # leaving long stretches of control (no-perturbation) signal in between.
        block = N_SAMPLES // (len(category_labels) + 1)
        for i, label in enumerate(category_labels):
            start = (i + 1) * block - N_BURST // 2
            end = start + N_BURST
            labels[start:end] = label
            features[start:end] += rng.normal(1.5, 0.5, size=(1, N_FEATURES))

        seconds = np.arange(N_SAMPLES) / SAMPLING_RATE
        timestamps = [
            f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}" for s in seconds
        ]

        frame = pd.DataFrame(features, columns=[f"sensor_{i:03d}" for i in range(N_FEATURES)])
        frame.insert(0, "Label", labels)
        frame.insert(1, "Timestamp_Xsens", timestamps)
        frame.insert(2, "Frame", np.arange(N_SAMPLES))
        frame["Participant"] = f"Participant{p}"
        frame["class"] = (labels > 0).astype(np.int64)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def main():
    # Short training run: this is a plumbing check, not a result.
    config = Config(window_size=50, step_size=25, epochs=3, batch_size=32)

    print("=" * 72)
    print("MI-CNN quickstart (synthetic data - results are meaningless by design)")
    print("=" * 72)

    df = make_synthetic_dataset()
    df["Timestamp_seconds"] = np.arange(len(df)) % N_SAMPLES / SAMPLING_RATE
    print(f"\nSynthetic dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # --- 1) Parameter count of the published architecture --------------------
    print("\n[1/3] Architecture check")
    reference = build_micnn(input_shape=(100, 1001), **Config().model_kwargs())
    print(f"  MI-CNN with input (100, 1001): {reference.count_params():,} trainable parameters")
    print("  (paper reports 256,066)")

    # --- 2) Leave-One-Participant-Out ----------------------------------------
    print("\n[2/3] Leave-One-Participant-Out")
    X, y, participant_ids, features = create_windows(
        df, config.window_size, config.step_size, verbose=False
    )
    print(f"  windows: {X.shape} | features: {len(features)}")
    lopo = run_lopo(X, y, participant_ids, config, verbose=False)
    print(lopo.to_string(index=False))
    print("\n  Summary:")
    print(summarize(lopo).to_string())

    # --- 3) Leave-One-Scenario-Out -------------------------------------------
    print("\n[3/3] Leave-One-Scenario-Out")
    Xs, ys, parts, categories, sublabels = create_scenario_windows(
        df, config.window_size, config.step_size, verbose=False
    )
    print(f"  windows: {Xs.shape} | categories: {sorted(set(map(str, categories)))}")
    loso, records = run_loso(Xs, ys, parts, categories, sublabels, config, verbose=False)
    print(loso.to_string(index=False))

    print("\n" + "=" * 72)
    print("Pipeline ran end to end.")
    print("=" * 72)


if __name__ == "__main__":
    main()
