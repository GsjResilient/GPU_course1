#!/usr/bin/env python3
"""Benchmark every exported architecture and write a reproducible CSV summary."""

import argparse
import csv
import re
import statistics
import subprocess
from pathlib import Path


def run(command):
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return result.stdout


def capture(pattern, text, cast=float):
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"cannot parse {pattern!r} from:\n{text}")
    return cast(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("weights/candidates"))
    parser.add_argument("--pattern", default="*.bin")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--executable", type=Path, default=Path("build/mnist_cuda"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/architecture_sweep.csv"))
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for weights in sorted(args.weights.glob(args.pattern)):
        print(f"benchmarking {weights} ...", flush=True)
        custom_times, end_to_end_times, library_times = [], [], []
        custom = reference = ""
        for trial in range(1, args.trials + 1):
            custom = run([
                str(args.executable), "--data", str(args.data), "--weights", str(weights),
                "--batch-size", str(args.batch_size), "--warmup", str(args.warmup),
                "--repeats", str(args.repeats),
            ])
            reference = run([
                "python", "scripts/benchmark_pytorch.py", "--data", str(args.data),
                "--weights", str(weights), "--batch-size", str(args.batch_size),
                "--warmup", str(args.warmup), "--repeats", str(args.repeats),
            ])
            custom_times.append(capture(r"CUDA kernels: ([\d.]+) ms", custom))
            end_to_end_times.append(capture(r"End-to-end: ([\d.]+) ms", custom))
            library_times.append(capture(r"NVIDIA libraries: ([\d.]+) ms", reference))
            print(f"  trial {trial}: CUDA={custom_times[-1]:.3f} ms, "
                  f"library={library_times[-1]:.3f} ms", flush=True)
        model = re.search(
            r"Model: k=(\d+), c1=(\d+), c2=(\d+), hidden=(\d+), parameters=(\d+)",
            custom,
        )
        if not model:
            raise RuntimeError(f"cannot parse model from:\n{custom}")
        custom_ms = statistics.median(custom_times)
        library_ms = statistics.median(library_times)
        row = {
            "weights": weights.name,
            "kernel": int(model.group(1)),
            "c1": int(model.group(2)),
            "c2": int(model.group(3)),
            "hidden": int(model.group(4)),
            "parameters": int(model.group(5)),
            "accuracy_percent": capture(r"Accuracy: ([\d.]+)%", custom),
            "custom_cuda_ms": custom_ms,
            "custom_cuda_min_ms": min(custom_times),
            "custom_cuda_max_ms": max(custom_times),
            "end_to_end_ms": statistics.median(end_to_end_times),
            "nvidia_library_ms": library_ms,
            "nvidia_library_min_ms": min(library_times),
            "nvidia_library_max_ms": max(library_times),
            "library_performance_percent": library_ms / custom_ms * 100.0,
        }
        rows.append(row)
        print(f"  median: accuracy={row['accuracy_percent']:.3f}%, "
              f"CUDA={custom_ms:.3f} ms, library={library_ms:.3f} ms, "
              f"ratio={row['library_performance_percent']:.2f}%", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
