#!/usr/bin/env python3
"""Evaluate the fixed classic LeNet-5 comparison weights (inference only)."""

import argparse
import json
from pathlib import Path

from compare_lenet5 import HERE, load_weights, reference
from run import find_mnist, load_mnist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-dir", type=Path)
    parser.add_argument("--weights", type=Path, default=HERE / "assets" / "lenet5_classic.npz")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--out", type=Path,
                        default=HERE.parent / "build" / "competition" / "lenet5_accuracy.json")
    args = parser.parse_args()
    weights = load_weights(args.weights)
    mnist = find_mnist(args.mnist_dir)
    correct = 0
    for offset in range(0, 10000, args.batch_size):
        count = min(args.batch_size, 10000 - offset)
        images, labels = load_mnist(mnist, offset, count)
        correct += int((reference(images, weights)["predictions"] == labels).sum())
    result = {"network": "modern-classic-lenet5", "correct": correct, "total": 10000,
              "accuracy": correct / 10000, "parameters": 44426,
              "training_or_gradient_descent": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
