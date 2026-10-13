NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math -lineinfo
ARCH ?= sm_90
TARGET := build/mnist_cuda
SOURCE := src/main.cu

.PHONY: all clean data train run baseline

all: $(TARGET)

$(TARGET): $(SOURCE)
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@

data:
	python scripts/download_mnist.py

train:
	python scripts/train_export.py --epochs 8 --c1 4 --c2 8 --hidden 16 --kernel 3

run: $(TARGET)
	./$(TARGET) --data data --weights weights/lenet.bin --batch-size 10000 --warmup 10 --repeats 30

baseline:
	python scripts/benchmark_pytorch.py --data data --weights weights/lenet.bin --batch-size 10000 --warmup 10 --repeats 30

clean:
	rm -f $(TARGET)
