"""Single-window inference latency and model-size benchmark for MI-CNN.

Reproduces the deployment-feasibility numbers reported in the paper: trainable
parameter count, on-disk size, and per-window inference time on CPU and GPU.
Latency is measured on a batch of one, because an online VR system would classify
one window at a time rather than in batches.

By default the script builds a fresh (untrained) MI-CNN, since timing depends on
the architecture and not on the learned weights. Pass ``--model`` to benchmark a
saved checkpoint instead.

Usage:
    python scripts/measure_latency.py
    python scripts/measure_latency.py --features 1001 --runs 2000
    python scripts/measure_latency.py --model path/to/model.keras --device cpu
"""

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micnn import Config, MambaInspiredBlock, build_micnn  # noqa: E402


def time_predictions(predict_fn, x, n_warmup, n_runs):
    """Return per-call latency in milliseconds over ``n_runs`` timed calls.

    Warm-up calls are discarded: the first calls pay for graph tracing, kernel
    autotuning, and GPU memory allocation, which would otherwise dominate.
    """
    for _ in range(n_warmup):
        predict_fn(x)

    latencies = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        start = time.perf_counter()
        predict_fn(x)
        latencies[i] = (time.perf_counter() - start) * 1000.0
    return latencies


def report(label, latencies):
    print(f"\n=== {label} ===")
    print(f"  mean   = {latencies.mean():.3f} ms")
    print(f"  std    = {latencies.std(ddof=1):.3f} ms")
    print(f"  median = {np.median(latencies):.3f} ms")
    print(f"  p95    = {np.percentile(latencies, 95):.3f} ms")
    print(f"  p99    = {np.percentile(latencies, 99):.3f} ms")


def gpu_name():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    try:
        return tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
    except Exception:
        return "GPU (name unavailable)"


def load_or_build(args, config):
    """Load a saved checkpoint, or build a fresh model with the paper's shape."""
    if args.model:
        print(f"Loading model: {args.model}")
        # `MambaBlock` is the name the layer had in the original training
        # notebooks; it is aliased here so older checkpoints still deserialise.
        return keras.models.load_model(
            args.model,
            custom_objects={
                "MambaInspiredBlock": MambaInspiredBlock,
                "MambaBlock": MambaInspiredBlock,
            },
            compile=False,
        )
    print(f"Building a fresh MI-CNN with input shape ({args.timesteps}, {args.features})")
    return build_micnn(input_shape=(args.timesteps, args.features), **config.model_kwargs())


def main():
    parser = argparse.ArgumentParser(description="MI-CNN latency benchmark")
    parser.add_argument("--model", default=None, help="Optional saved .keras checkpoint.")
    parser.add_argument("--timesteps", type=int, default=Config.window_size)
    parser.add_argument(
        "--features", type=int, default=1001, help="Feature channels per time step."
    )
    parser.add_argument("--device", choices=["both", "gpu", "cpu"], default="both")
    parser.add_argument("--warmup", type=int, default=20, help="Discarded warm-up calls.")
    parser.add_argument("--runs", type=int, default=1000, help="Timed single-window calls.")
    args = parser.parse_args()

    config = Config()

    print("=" * 72)
    print("MI-CNN single-window inference latency benchmark")
    print("=" * 72)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Built with CUDA    : {tf.test.is_built_with_cuda()}")
    print(f"Visible GPUs       : {tf.config.list_physical_devices('GPU')}")
    if gpu_name():
        print(f"GPU device         : {gpu_name()}")

    model = load_or_build(args, config)
    print(f"\nInput shape          : {model.input_shape}")
    print(f"Trainable parameters : {model.count_params():,}")
    if args.model and os.path.exists(args.model):
        print(f"Checkpoint size      : {os.path.getsize(args.model) / 1024 ** 2:.2f} MB")

    # One random window is enough: inference cost is data-independent here.
    x = np.random.randn(1, *model.input_shape[1:]).astype(np.float32)
    print(f"\nSingle-window input  : {x.shape}")
    print(f"Warm-up: {args.warmup} | timed runs: {args.runs}")

    has_gpu = bool(tf.config.list_physical_devices("GPU"))

    if args.device in ("both", "gpu"):
        if has_gpu:
            @tf.function
            def predict_gpu(inputs):
                return model(inputs, training=False)

            report(f"GPU ({gpu_name()})", time_predictions(predict_gpu, x, args.warmup, args.runs))
        elif args.device == "gpu":
            print("\n[!] GPU requested but none is visible to TensorFlow - skipping.")

    if args.device in ("both", "cpu"):
        # Rebuild under /CPU:0 so the variables themselves live on the CPU;
        # calling a GPU-placed model from a CPU-pinned function would otherwise
        # measure cross-device transfers rather than CPU inference.
        print("\nPlacing model on CPU for the CPU measurement ...")
        with tf.device("/CPU:0"):
            cpu_model = load_or_build(args, config)
            if args.model is None:
                cpu_model.set_weights(model.get_weights())

            @tf.function
            def predict_cpu(inputs):
                return cpu_model(inputs, training=False)

            latencies = time_predictions(predict_cpu, x, args.warmup, args.runs)
        report("CPU", latencies)

    print("\n" + "=" * 72)
    print(f"Done. Model: {model.count_params():,} trainable parameters")
    print("=" * 72)


if __name__ == "__main__":
    main()
