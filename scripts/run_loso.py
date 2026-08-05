"""Leave-One-Scenario-Out evaluation of MI-CNN.

Each fold holds out one of the seven perturbation categories entirely and tests
whether the model still detects that never-seen kind of instability. This is the
control that participant-wise validation cannot provide: every participant
experiences the same set of perturbations, so LOPO alone cannot rule out the
model latching onto scenario-specific structure.

Usage:
    python scripts/run_loso.py --data final_dataset.csv
    python scripts/run_loso.py --data final_dataset.csv --plot
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micnn import (  # noqa: E402
    Config,
    append_summary_rows,
    create_scenario_windows,
    load_dataset,
    run_loso,
    sublabel_breakdown,
    summarize,
)
from micnn.evaluation import METRIC_COLUMNS  # noqa: E402
from micnn.scenarios import check_labels  # noqa: E402

LOSO_METRICS = METRIC_COLUMNS + ["scenario_recall", "control_recall"]


def plot_results(results: pd.DataFrame, breakdown: pd.DataFrame, output_dir: str) -> None:
    """Save a per-scenario bar chart, plus a sub-scenario chart for the weakest fold."""
    import matplotlib.pyplot as plt

    ordered = results.sort_values("scenario_recall")
    x = np.arange(len(ordered))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ordered["scenario_recall"], width, label="Recall on unseen scenario")
    ax.bar(x + width / 2, ordered["auc"], width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["Scenario"], rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, ls="--", c="grey", lw=1)  # chance level
    ax.set_ylabel("Score")
    ax.set_title("MI-CNN - leave-one-scenario-out generalization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "micnn-loso.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if breakdown.empty:
        return

    # Zoom in on the hardest category to show which sub-scenarios drag it down.
    worst = results.sort_values("scenario_recall").iloc[0]["Scenario"]
    selection = breakdown[breakdown["Scenario"] == worst].sort_values("recall")
    if selection.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(selection["Label_name"], selection["recall"], color="#c44e52")
    for i, (recall, n) in enumerate(zip(selection["recall"], selection["n_windows"])):
        ax.text(min(recall + 0.01, 0.98), i, f"{recall:.2f} (n={n})", va="center", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Detection rate (recall)")
    ax.set_title(f"Sub-scenario recall within held-out '{worst}'")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "micnn-loso-worst-sublabels.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="MI-CNN leave-one-scenario-out evaluation")
    parser.add_argument("--data", required=True, help="Path to the dataset CSV.")
    parser.add_argument("--output-dir", default="results", help="Directory for CSVs and figures.")
    parser.add_argument("--window-size", type=int, default=Config.window_size)
    parser.add_argument("--step-size", type=int, default=Config.step_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seed", type=int, default=Config.random_seed)
    parser.add_argument("--plot", action="store_true", help="Also save summary figures.")
    args = parser.parse_args()

    config = Config(
        window_size=args.window_size,
        step_size=args.step_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    print(f"Loading {args.data} ...")
    df = load_dataset(args.data)
    check_labels(df["Label"].unique())

    X, y, participant_ids, categories, sublabels = create_scenario_windows(
        df, config.window_size, config.step_size
    )

    unique, counts = np.unique(categories, return_counts=True)
    print("\nWindows per group:")
    for name, count in sorted(zip(unique, counts), key=lambda t: -t[1]):
        print(f"  {name:32s} {count:6d}")

    results, sublabel_records = run_loso(
        X, y, participant_ids, categories, sublabels, config
    )

    print("\nLeave-one-scenario-out results")
    print("=" * 90)
    print(results.to_string(index=False))

    print("\nSummary (mean +/- std across held-out scenarios)")
    print("=" * 90)
    print(summarize(results, LOSO_METRICS).to_string())

    os.makedirs(args.output_dir, exist_ok=True)
    loso_path = os.path.join(args.output_dir, "micnn-loso.csv")
    append_summary_rows(results, fold_column="Scenario", metric_columns=LOSO_METRICS).to_csv(
        loso_path, index=False
    )
    print(f"\nSaved: {loso_path}")

    breakdown = sublabel_breakdown(sublabel_records)
    if not breakdown.empty:
        print("\nDetection rate per sub-scenario within each held-out category")
        print("=" * 90)
        print(
            breakdown.to_string(
                index=False,
                formatters={"recall": "{:.3f}".format, "mean_proba": "{:.3f}".format},
            )
        )
        breakdown_path = os.path.join(args.output_dir, "micnn-loso-by-sublabel.csv")
        breakdown.to_csv(breakdown_path, index=False)
        print(f"\nSaved: {breakdown_path}")

    if args.plot:
        plot_results(results, breakdown, args.output_dir)
        print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
