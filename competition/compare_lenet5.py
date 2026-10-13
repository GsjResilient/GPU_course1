#!/usr/bin/env python3
"""Run a modern classic LeNet-5 on NutShellGPU for a fair network comparison."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from run import HERE, ROOT, as_bytes, find_mnist, load_mnist

sys.path.insert(0, str(ROOT / "toolchain"))
from ntcc import compile_file  # noqa: E402
from ntruntime import Session  # noqa: E402


def load_weights(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        result = {name: np.ascontiguousarray(source[name], dtype="<f4") for name in source.files}
    shapes = {
        "conv1.weight": (6, 1, 5, 5), "conv1.bias": (6,),
        "conv2.weight": (16, 6, 5, 5), "conv2.bias": (16,),
        "fc1.weight": (120, 256), "fc1.bias": (120,),
        "fc2.weight": (84, 120), "fc2.bias": (84,),
        "fc3.weight": (10, 84), "fc3.bias": (10,),
    }
    if set(result) != set(shapes):
        raise ValueError("unexpected LeNet-5 weight tensors")
    for name, shape in shapes.items():
        if result[name].shape != shape:
            raise ValueError(f"{name}: expected {shape}, got {result[name].shape}")
    return result


def conv5(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    windows = sliding_window_view(x, (5, 5), axis=(-2, -1))
    return np.einsum("nihwkl,oikl->nohw", windows, weight, optimize=True) + bias[None, :, None, None]


def pool2(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], x.shape[1], x.shape[2] // 2, 2, x.shape[3] // 2, 2).max((3, 5))


def reference(images: np.ndarray, weight: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pool1 = pool2(np.maximum(conv5(images, weight["conv1.weight"], weight["conv1.bias"]), 0))
    pool2_out = pool2(np.maximum(conv5(pool1, weight["conv2.weight"], weight["conv2.bias"]), 0))
    hidden1 = np.maximum(pool2_out.reshape(len(images), 256) @ weight["fc1.weight"].T + weight["fc1.bias"], 0)
    hidden2 = np.maximum(hidden1 @ weight["fc2.weight"].T + weight["fc2.bias"], 0)
    logits = hidden2 @ weight["fc3.weight"].T + weight["fc3.bias"]
    return {"pool1": pool1, "pool2": pool2_out, "hidden1": hidden1,
            "hidden2": hidden2, "logits": logits,
            "predictions": logits.argmax(1).astype(np.int32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("func", "cycle"), default="func")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mnist-dir", type=Path)
    parser.add_argument("--weights", type=Path, default=HERE / "assets" / "lenet5_classic.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "lenet5_comparison")
    parser.add_argument("--budget", type=int, default=100_000_000)
    parser.add_argument("--block-size", type=int, choices=(16, 32, 64, 128, 256), default=32)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    weights = load_weights(args.weights)
    images, labels = load_mnist(find_mnist(args.mnist_dir), args.offset, args.samples)
    expected = reference(images, weights)
    names = ("conv5_relu", "maxpool", "dense_relu", "dense_linear", "argmax")
    modules = {name: compile_file(HERE / "kernels" / f"{name}.ntcu",
                                  args.out / "compiled" / name) for name in names}
    block = args.block_size
    blocks = lambda count: (count + block - 1) // block
    records = []
    previous = 0
    started = time.perf_counter()

    with Session(modules["conv5_relu"], args.backend) as device:
        ids = {"conv5_relu": 0}
        for name in names[1:]:
            ids[name] = device.load_module(modules[name])
        address = {name: device.upload(as_bytes(value)) for name, value in weights.items()}
        address.update({
            "input": device.upload(as_bytes(images)),
            "conv1": device.upload(bytes(args.samples * 6 * 24 * 24 * 4)),
            "pool1": device.upload(bytes(args.samples * 6 * 12 * 12 * 4)),
            "conv2": device.upload(bytes(args.samples * 16 * 8 * 8 * 4)),
            "pool2": device.upload(bytes(args.samples * 16 * 4 * 4 * 4)),
            "hidden1": device.upload(bytes(args.samples * 120 * 4)),
            "hidden2": device.upload(bytes(args.samples * 84 * 4)),
            "logits": device.upload(bytes(args.samples * 10 * 4)),
            "predictions": device.upload(bytes(args.samples * 4)),
        })
        launches = (
            ("conv1", "conv5_relu", (blocks(3456), args.samples),
             {"input": address["input"], "weight": address["conv1.weight"], "bias": address["conv1.bias"],
              "output": address["conv1"], "ci": 1, "h": 28, "out_h": 24, "output_count": 3456}),
            ("pool1", "maxpool", (blocks(864), args.samples),
             {"input": address["conv1"], "output": address["pool1"], "channels": 6,
              "h": 24, "out_h": 12, "output_count": 864}),
            ("conv2", "conv5_relu", (blocks(1024), args.samples),
             {"input": address["pool1"], "weight": address["conv2.weight"], "bias": address["conv2.bias"],
              "output": address["conv2"], "ci": 6, "h": 12, "out_h": 8, "output_count": 1024}),
            ("pool2", "maxpool", (blocks(256), args.samples),
             {"input": address["conv2"], "output": address["pool2"], "channels": 16,
              "h": 8, "out_h": 4, "output_count": 256}),
            ("hidden1", "dense_relu", (blocks(args.samples * 120),),
             {"input": address["pool2"], "weight": address["fc1.weight"], "bias": address["fc1.bias"],
              "output": address["hidden1"], "input_count": 256, "output_count": 120,
              "total_count": args.samples * 120}),
            ("hidden2", "dense_relu", (blocks(args.samples * 84),),
             {"input": address["hidden1"], "weight": address["fc2.weight"], "bias": address["fc2.bias"],
              "output": address["hidden2"], "input_count": 120, "output_count": 84,
              "total_count": args.samples * 84}),
            ("logits", "dense_linear", (blocks(args.samples * 10),),
             {"input": address["hidden2"], "weight": address["fc3.weight"], "bias": address["fc3.bias"],
              "output": address["logits"], "input_count": 84, "output_count": 10,
              "total_count": args.samples * 10}),
            ("predictions", "argmax", (blocks(args.samples),),
             {"input": address["logits"], "output": address["predictions"], "samples": args.samples}),
        )
        for layer, kernel, grid, kernel_args in launches:
            layer_start = time.perf_counter()
            device.launch(grid, block, kernel_args, module=ids[kernel])
            event = device.synchronize(args.budget)
            counter = "cycles" if args.backend == "cycle" else "instructions"
            cumulative = int(event[counter])
            records.append({"layer": layer, "kernel": kernel, "grid": list(grid), "block": block,
                            f"layer_{counter}": cumulative - previous,
                            "host_seconds": time.perf_counter() - layer_start, "execution": event})
            previous = cumulative
        shapes = {"pool1": (args.samples, 6, 12, 12), "pool2": (args.samples, 16, 4, 4),
                  "hidden1": (args.samples, 120), "hidden2": (args.samples, 84)}
        actual = {name: np.frombuffer(device.download(address[name], int(np.prod(shape)) * 4),
                                      dtype="<f4").reshape(shape) for name, shape in shapes.items()}
        actual["predictions"] = np.frombuffer(device.download(address["predictions"], args.samples * 4),
                                               dtype="<i4").copy()

    errors = {}
    for name in ("pool1", "pool2", "hidden1", "hidden2"):
        np.testing.assert_allclose(actual[name], expected[name], atol=5e-5, rtol=5e-5)
        errors[name] = float(np.max(np.abs(actual[name] - expected[name])))
    if not np.array_equal(actual["predictions"], expected["predictions"]):
        raise AssertionError("NutShellGPU predictions differ from the LeNet-5 NumPy reference")
    result = {
        "backend": args.backend, "network": "modern-classic-lenet5",
        "parameters": 44426, "samples": args.samples, "offset": args.offset,
        "block_size": block, "predictions": actual["predictions"].tolist(), "labels": labels.tolist(),
        "accuracy_on_selected_samples": float(np.mean(actual["predictions"] == labels)),
        "reference_match": True, "max_abs_error": errors, "layers": records,
        "total_cycles": previous if args.backend == "cycle" else 0,
        "total_instructions": previous if args.backend == "func" else 0,
        "simulated_seconds_1ghz": previous / 1e9 if args.backend == "cycle" else 0,
        "host_seconds": time.perf_counter() - started,
    }
    (args.out / f"result_{args.backend}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
