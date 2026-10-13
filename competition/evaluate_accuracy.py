#!/usr/bin/env python3
"""Evaluate the fixed inference weights on all 10,000 MNIST test images."""

import argparse
import json
from pathlib import Path

from run import HERE, find_mnist, load_mnist, load_weights, reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-dir", type=Path)
    parser.add_argument("--weights", type=Path, default=HERE / "assets" / "lenet.bin")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--out", type=Path, default=HERE.parent / "build" / "competition" / "accuracy.json")
    args = parser.parse_args()
    weights = load_weights(args.weights)
    mnist = find_mnist(args.mnist_dir)
    correct = 0
    for offset in range(0, 10000, args.batch_size):
        size = min(args.batch_size, 10000 - offset)
        images, labels = load_mnist(mnist, offset, size)
        correct += int((reference(images, weights)["predictions"] == labels).sum())
    result = {"correct": correct, "total": 10000, "accuracy": correct / 10000,
              "parameters": 3722, "training_or_gradient_descent": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
