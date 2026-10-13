#!/usr/bin/env python3
"""Train LeNet on MNIST and export weights for the standalone CUDA program."""

import argparse
import random
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet(nn.Module):
    def __init__(self, c1=16, c2=32, hidden=128, kernel=5):
        super().__init__()
        self.c1, self.c2, self.hidden, self.kernel = c1, c2, hidden, kernel
        self.conv1 = nn.Conv2d(1, c1, kernel)
        self.conv2 = nn.Conv2d(c1, c2, kernel)
        spatial = ((28 - kernel + 1) // 2 - kernel + 1) // 2
        self.spatial = spatial
        self.fc1 = nn.Linear(c2 * spatial * spatial, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.numel()
    return correct / total


def export(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = [
        model.conv1.weight, model.conv1.bias,
        model.conv2.weight, model.conv2.bias,
        model.fc1.weight, model.fc1.bias,
        model.fc2.weight, model.fc2.bias,
    ]
    with path.open("wb") as f:
        f.write(b"LENETV2\0")
        f.write(struct.pack("<IIIII", 2, model.kernel, model.c1,
                            model.c2, model.hidden))
        for tensor in tensors:
            array = tensor.detach().cpu().contiguous().numpy().astype("<f4")
            f.write(array.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("weights/lenet.bin"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--c1", type=int, default=4)
    parser.add_argument("--c2", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--kernel", type=int, choices=(3, 5), default=3)
    parser.add_argument("--augment", action="store_true",
                        help="use small affine perturbations for the training set")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_steps = []
    if args.augment:
        train_steps.append(transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)))
    train_steps.extend([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_transform = transforms.Compose(train_steps)
    train_set = datasets.MNIST(args.data, train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(args.data, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = LeNet(args.c1, args.c2, args.hidden, args.kernel).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"model: kernel={args.kernel}, c1={args.c1}, c2={args.c2}, "
          f"hidden={args.hidden}, parameters={parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
        scheduler.step()
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch {epoch:02d}: loss={loss_sum / len(train_set):.5f}, "
              f"test_accuracy={accuracy * 100:.3f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    export(model, args.output)
    print(f"exported {args.output} (best test accuracy: {best_accuracy * 100:.3f}%)")


if __name__ == "__main__":
    main()
