#!/usr/bin/env python3
"""cuDNN/cuBLAS reference benchmark; not used by the submitted inference path."""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_export import LeNet


def load_exported_model(path):
    names = [
        "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias",
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
    ]
    with path.open("rb") as f:
        if f.read(8) != b"LENETV2\0":
            raise ValueError("invalid weights file")
        version, kernel, c1, c2, hidden = struct.unpack("<IIIII", f.read(20))
        if version != 2:
            raise ValueError("unsupported weights version")
        model = LeNet(c1, c2, hidden, kernel)
        state = model.state_dict()
        for name in names:
            count = state[name].numel()
            values = np.frombuffer(f.read(count * 4), dtype="<f4").copy()
            if values.size != count:
                raise ValueError("truncated weights file")
            state[name] = torch.from_numpy(values.reshape(tuple(state[name].shape)))
        if f.read(1):
            raise ValueError("unexpected trailing bytes in weights file")
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--weights", type=Path, default=Path("weights/lenet.bin"))
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this reference benchmark")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model = load_exported_model(args.weights)
    model.eval().to(device)
    if args.precision == "fp16":
        model.half()
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_set = datasets.MNIST(args.data, train=False, download=False, transform=transform)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    warm_images, _ = next(iter(loader))
    warm_images = warm_images.to(device)
    if args.precision == "fp16":
        warm_images = warm_images.half()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(warm_images)
    torch.cuda.synchronize()

    total_ms = 0.0
    correct = total = 0
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if args.precision == "fp16":
                images = images.half()
            start.record()
            for _ in range(args.repeats):
                logits = model(images)
            stop.record()
            stop.synchronize()
            total_ms += start.elapsed_time(stop) / args.repeats
            predictions = logits.argmax(1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Samples: {total}, batch size: {args.batch_size}")
    print(f"Accuracy: {100 * correct / total:.3f}% ({correct}/{total})")
    print(f"NVIDIA libraries: {total_ms:.3f} ms total, "
          f"{total * 1000 / total_ms:.3f} images/s, "
          f"{total_ms * 1000 / total:.3f} us/image")


if __name__ == "__main__":
    main()
