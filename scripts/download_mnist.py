#!/usr/bin/env python3
"""Download and extract the MNIST IDX files used by the CUDA executable."""

import argparse
from pathlib import Path

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    datasets.MNIST(args.data, train=True, download=True)
    datasets.MNIST(args.data, train=False, download=True)
    print(f"MNIST is ready under {args.data / 'MNIST' / 'raw'}")


if __name__ == "__main__":
    main()
