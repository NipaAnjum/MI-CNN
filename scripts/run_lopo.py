"""Leave-One-Participant-Out evaluation of MI-CNN.

Reproduces the headline result: one fold per participant, each fold trained on
everyone else and tested on a person the model has never seen.

Usage:
    python scripts/run_lopo.py --data final_dataset.csv
    python scripts/run_lopo.py --data final_dataset.csv --step-size 100   # non-overlapping
    python scripts/run_lopo.py --data raw_dataset.csv --downsample        # balance first
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micnn import (  # noqa: E402
    Config,
    append_summary_rows,
    create_windows,
    downsample_by_participant,
    load_dataset,
    run_lopo,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description="MI-CNN leave-one-participant-out evaluation")
    parser.add_argument("--data", required=True, help="Path to the dataset CSV.")
    parser.add_argument("--output", default="results/micnn-lopo.csv", help="Output CSV path.")
    parser.add_argument(
        "--downsample",
        action="store_true",
        help="Apply participant-wise class balancing (skip if the CSV is already balanced).",
    )
    parser.add_argument("--window-size", type=int, default=Config.window_size)
    parser.add_argument(
        "--step-size",
        type=int,
        default=Config.step_size,
        help="Window stride; pass the same value as --window-size for non-overlapping windows.",
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--seed", type=int, default=Config.random_seed)
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
    if args.downsample:
        print("Applying participant-wise downsampling ...")
        df = downsample_by_participant(df, random_state=config.random_seed)
    print(f"Dataset shape: {df.shape}")

    X, y, participant_ids, feature_names = create_windows(
        df, config.window_size, config.step_size
    )
    print(f"\n{len(feature_names)} features per time step")

    results = run_lopo(X, y, participant_ids, config)

    print("\nPer-participant results")
    print("=" * 78)
    print(results.to_string(index=False))

    print("\nSummary (mean +/- std across participants)")
    print("=" * 78)
    print(summarize(results).to_string())

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    append_summary_rows(results, fold_column="Participant").to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
