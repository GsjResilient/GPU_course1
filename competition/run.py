#!/usr/bin/env python3
"""Run the HW1 MNIST CNN on NutShellGPU (inference only)."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "toolchain"))
from ntcc import compile_file  # noqa: E402
from ntruntime import Session  # noqa: E402


TENSOR_NAMES = (
    "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
    "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
)


def load_weights(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    if raw[:8] != b"LENETV2\0":
        raise ValueError(f"bad weight magic in {path}")
    version, kernel, c1, c2, hidden = struct.unpack_from("<5I", raw, 8)
    if (version, kernel, c1, c2, hidden) != (2, 3, 4, 8, 16):
        raise ValueError("competition kernels require V2 k3/c1=4/c2=8/hidden=16 weights")
    shapes = (
        (4, 1, 3, 3), (4,), (8, 4, 3, 3), (8,),
        (16, 200), (16,), (10, 16), (10,),
    )
    offset = 28
    result: dict[str, np.ndarray] = {}
    for name, shape in zip(TENSOR_NAMES, shapes):
        count = int(np.prod(shape))
        value = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).copy().reshape(shape)
        offset += value.nbytes
        result[name] = value
    if offset != len(raw):
        raise ValueError(f"unexpected trailing data in {path}")
    return result


def find_mnist(path: Path | None) -> Path:
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend((HERE / "assets" / "mnist", Path("/root/GPU_course1/data/MNIST/raw")))
    for candidate in candidates:
        if (candidate / "t10k-images-idx3-ubyte").is_file() and (candidate / "t10k-labels-idx1-ubyte").is_file():
            return candidate
    raise FileNotFoundError("MNIST raw files not found; pass --mnist-dir DIR")


def load_mnist(path: Path, offset: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    image_file = path / "t10k-images-idx3-ubyte"
    label_file = path / "t10k-labels-idx1-ubyte"
    image_raw = image_file.read_bytes()
    label_raw = label_file.read_bytes()
    magic, count, rows, cols = struct.unpack_from(">4I", image_raw)
    lmagic, lcount = struct.unpack_from(">2I", label_raw)
    if (magic, lmagic, rows, cols, count, lcount) != (2051, 2049, 28, 28, 10000, 10000):
        raise ValueError("unexpected MNIST IDX header")
    if offset < 0 or samples < 1 or offset + samples > count:
        raise ValueError("requested sample range is outside the MNIST test set")
    begin = 16 + offset * 784
    images = np.frombuffer(image_raw, dtype=np.uint8, count=samples * 784, offset=begin)
    images = images.reshape(samples, 1, 28, 28).astype(np.float32)
    images = np.ascontiguousarray((images / 255.0 - 0.1307) / 0.3081, dtype="<f4")
    labels = np.frombuffer(label_raw, dtype=np.uint8, count=samples, offset=8 + offset).copy()
    return images, labels


def conv_valid(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    windows = sliding_window_view(x, (3, 3), axis=(-2, -1))
    return np.einsum("nihwkl,oikl->nohw", windows, weight, optimize=True) + bias[None, :, None, None]


def pool2(x: np.ndarray) -> np.ndarray:
    h = x.shape[-2] // 2 * 2
    w = x.shape[-1] // 2 * 2
    return x[:, :, :h, :w].reshape(x.shape[0], x.shape[1], h // 2, 2, w // 2, 2).max((3, 5))


def reference(images: np.ndarray, weight: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pool1 = pool2(np.maximum(conv_valid(images, weight["conv1.weight"], weight["conv1.bias"]), 0))
    pool2_out = pool2(np.maximum(conv_valid(pool1, weight["conv2.weight"], weight["conv2.bias"]), 0))
    hidden = np.maximum(pool2_out.reshape(len(images), 200) @ weight["fc1.weight"].T + weight["fc1.bias"], 0)
    logits = hidden @ weight["fc2.weight"].T + weight["fc2.bias"]
    return {"pool1": pool1, "pool2": pool2_out, "hidden": hidden, "logits": logits,
            "predictions": logits.argmax(1).astype(np.int32)}


def compile_kernels(out: Path, variant: str) -> dict[str, Path]:
    compiled = out / "compiled"
    names = ["dense_relu"]
    names += ["dense_argmax"] if variant != "optimized" else ["dense_linear", "argmax"]
    names += ["conv_relu_pool"] if variant == "fused" else ["conv_relu", "maxpool"]
    return {
        name: compile_file(HERE / "kernels" / f"{name}.ntcu", compiled / name)
        for name in names
    }


def as_bytes(value: np.ndarray) -> bytes:
    return np.ascontiguousarray(value, dtype="<f4").tobytes()


def run_layer(device: Session, module: int, grid, block, args: dict, budget: int) -> tuple[dict, float]:
    start = time.perf_counter()
    device.launch(grid, block, args, module=module)
    event = device.synchronize(budget)
    return event, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("func", "cycle"), default="func")
    parser.add_argument("--variant", choices=("fused", "unfused", "optimized"), default="optimized")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mnist-dir", type=Path)
    parser.add_argument("--weights", type=Path, default=HERE / "assets" / "lenet.bin")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "competition")
    parser.add_argument("--budget", type=int, default=100_000_000)
    parser.add_argument("--block-size", type=int, choices=(16, 32, 64, 128, 256), default=64)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    weights = load_weights(args.weights)
    images, labels = load_mnist(find_mnist(args.mnist_dir), args.offset, args.samples)
    expected = reference(images, weights)
    modules = compile_kernels(args.out, args.variant)
    block_size = args.block_size
    blocks = lambda count: (count + block_size - 1) // block_size
    total_start = time.perf_counter()
    records = []
    previous_counter = 0

    first_kernel = "conv_relu_pool" if args.variant == "fused" else "conv_relu"
    with Session(modules[first_kernel], args.backend) as device:
        module_ids = {first_kernel: 0}
        if args.variant in ("unfused", "optimized"):
            module_ids["maxpool"] = device.load_module(modules["maxpool"])
        module_ids["dense_relu"] = device.load_module(modules["dense_relu"])
        if args.variant == "optimized":
            module_ids["dense_linear"] = device.load_module(modules["dense_linear"])
            module_ids["argmax"] = device.load_module(modules["argmax"])
        else:
            module_ids["dense_argmax"] = device.load_module(modules["dense_argmax"])

        address = {name: device.upload(as_bytes(value)) for name, value in weights.items()}
        address["input"] = device.upload(as_bytes(images))
        if args.variant in ("unfused", "optimized"):
            address["conv1"] = device.upload(bytes(args.samples * 4 * 26 * 26 * 4))
            conv2_h = 10 if args.variant == "optimized" else 11
            address["conv2"] = device.upload(bytes(args.samples * 8 * conv2_h * conv2_h * 4))
        address["pool1"] = device.upload(bytes(args.samples * 4 * 13 * 13 * 4))
        address["pool2"] = device.upload(bytes(args.samples * 8 * 5 * 5 * 4))
        address["hidden"] = device.upload(bytes(args.samples * 16 * 4))
        if args.variant == "optimized":
            address["logits"] = device.upload(bytes(args.samples * 10 * 4))
        address["predictions"] = device.upload(bytes(args.samples * 4))

        if args.variant == "fused":
            front_launches = (
             ("pool1", "conv_relu_pool", (blocks(676), args.samples), block_size,
             {"input": address["input"], "weight": address["conv1.weight"],
              "bias": address["conv1.bias"], "output": address["pool1"],
              "ci": 1, "h": 28, "out_h": 13, "output_count": 676}),
            ("pool2", "conv_relu_pool", (blocks(200), args.samples), block_size,
             {"input": address["pool1"], "weight": address["conv2.weight"],
              "bias": address["conv2.bias"], "output": address["pool2"],
              "ci": 4, "h": 13, "out_h": 5, "output_count": 200}),
            )
        else:
            conv2_h = 10 if args.variant == "optimized" else 11
            conv2_count = 800 if args.variant == "optimized" else 968
            front_launches = (
             ("conv1", "conv_relu", (blocks(2704), args.samples), block_size,
              {"input": address["input"], "weight": address["conv1.weight"],
               "bias": address["conv1.bias"], "output": address["conv1"],
               "ci": 1, "h": 28, "out_h": 26, "output_count": 2704}),
             ("pool1", "maxpool", (blocks(676), args.samples), block_size,
              {"input": address["conv1"], "output": address["pool1"],
               "channels": 4, "h": 26, "out_h": 13, "output_count": 676}),
             ("conv2", "conv_relu", (blocks(conv2_count), args.samples), block_size,
              {"input": address["pool1"], "weight": address["conv2.weight"],
               "bias": address["conv2.bias"], "output": address["conv2"],
               "ci": 4, "h": 13, "out_h": conv2_h, "output_count": conv2_count}),
             ("pool2", "maxpool", (blocks(200), args.samples), block_size,
              {"input": address["conv2"], "output": address["pool2"],
               "channels": 8, "h": conv2_h, "out_h": 5, "output_count": 200}),
            )
        tail_launches = (
            ("hidden", "dense_relu", (blocks(args.samples * 16),), block_size,
             {"input": address["pool2"], "weight": address["fc1.weight"],
              "bias": address["fc1.bias"], "output": address["hidden"],
              "input_count": 200, "output_count": 16,
              "total_count": args.samples * 16}),
        )
        if args.variant == "optimized":
            tail_launches += (
             ("logits", "dense_linear", (blocks(args.samples * 10),), block_size,
              {"input": address["hidden"], "weight": address["fc2.weight"],
               "bias": address["fc2.bias"], "output": address["logits"],
               "input_count": 16, "output_count": 10,
               "total_count": args.samples * 10}),
             ("predictions", "argmax", (blocks(args.samples),), block_size,
              {"input": address["logits"], "output": address["predictions"],
               "samples": args.samples}),
            )
        else:
            tail_launches += (
             ("predictions", "dense_argmax", (blocks(args.samples),), block_size,
             {"input": address["hidden"], "weight": address["fc2.weight"],
              "bias": address["fc2.bias"], "output": address["predictions"],
              "samples": args.samples}),
            )
        launches = front_launches + tail_launches

        for layer, kernel, grid, block, kernel_args in launches:
            event, elapsed = run_layer(device, module_ids[kernel], grid, block, kernel_args, args.budget)
            counter_name = "cycles" if args.backend == "cycle" else "instructions"
            cumulative = int(event.get(counter_name, 0))
            row = {"layer": layer, "kernel": kernel, "grid": list(grid), "block": block,
                   "host_seconds": elapsed, "execution": event,
                   f"layer_{counter_name}": cumulative - previous_counter}
            previous_counter = cumulative
            records.append(row)
            print(json.dumps(row), flush=True)

        actual = {
            "pool1": np.frombuffer(device.download(address["pool1"], args.samples * 4 * 13 * 13 * 4), dtype="<f4").reshape(args.samples, 4, 13, 13),
            "pool2": np.frombuffer(device.download(address["pool2"], args.samples * 8 * 5 * 5 * 4), dtype="<f4").reshape(args.samples, 8, 5, 5),
            "hidden": np.frombuffer(device.download(address["hidden"], args.samples * 16 * 4), dtype="<f4").reshape(args.samples, 16),
            "predictions": np.frombuffer(device.download(address["predictions"], args.samples * 4), dtype="<i4").copy(),
        }

    total_seconds = time.perf_counter() - total_start
    errors = {}
    for layer in ("pool1", "pool2", "hidden"):
        np.testing.assert_allclose(actual[layer], expected[layer], atol=2e-5, rtol=2e-5)
        errors[layer] = float(np.max(np.abs(actual[layer] - expected[layer])))
    if not np.array_equal(actual["predictions"], expected["predictions"]):
        raise AssertionError("NutShellGPU predictions differ from NumPy reference")

    predictions = actual["predictions"]
    result = {
        "backend": args.backend,
        "variant": args.variant,
        "network": {"kernel": 3, "c1": 4, "c2": 8, "hidden": 16, "parameters": 3722},
        "offset": args.offset,
        "samples": args.samples,
        "block_size": block_size,
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
        "accuracy_on_selected_samples": float(np.mean(predictions == labels)),
        "reference_match": True,
        "max_abs_error": errors,
        "layers": records,
        "total_cycles": previous_counter if args.backend == "cycle" else 0,
        "total_instructions": previous_counter if args.backend == "func" else 0,
        "total_host_seconds": total_seconds,
    }
    (args.out / f"result_{args.variant}_{args.backend}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
