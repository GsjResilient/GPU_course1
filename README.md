# GPU 课程大作业（一）：CUDA CNN 推理

本项目完成题目中的“问题一（基本题）”：在 MNIST 测试集上运行 LeNet CNN
推理。卷积、ReLU、最大池化和全连接前向过程均由本项目中的 CUDA kernel
实现；推理程序只使用 CUDA Runtime，不调用 cuDNN、cuBLAS 等 NVIDIA 计算库。

## 网络结构

```text
1x28x28 -> Conv3x3(4) -> ReLU -> MaxPool2x2
         -> Conv3x3(8) -> ReLU -> MaxPool2x2
         -> FC(16) -> ReLU -> FC(10)
```

最终网络只有 3,722 个参数，MNIST 测试准确率为 97.12%，高于本作业采用的
96% 合格阈值。完整网络被融合进一个 CUDA kernel：每个 thread block 处理一张
图片，所有中间特征保存在 shared memory，只把最终 logits 写回全局显存。
权重和激活使用 FP16，乘加采用 FP32 累加。测试程序同时报告：

- MNIST 测试集准确率；
- 只含 CUDA kernels 的时间、吞吐率和单图片延迟；
- 包含主机/设备传输及预测结果统计的端到端时间。

## 快速复现

环境要求：Linux、支持 C++17 的 CUDA Toolkit、Python 3，以及
`requirements.txt` 中的训练依赖。

```bash
# 1. 下载 MNIST（仓库已提供权重，无需重新训练）
make data

# 2. 编译。H20 使用 sm_90；其他 GPU 可通过 ARCH=sm_XX 修改
make -j

# 3. 在完整 10,000 张测试集上测试
make run
```

如需从头训练并重新导出权重，执行：

```bash
make train
```

已提供的 `weights/lenet.bin` 可跳过训练步骤直接测试。`--data` 可以指向
`data`、`data/MNIST` 或包含解压后 IDX 文件的 `raw` 目录。

查看所有参数：

```bash
./build/mnist_cuda --help
```

## 文件说明

- `src/main.cu`：IDX 数据读取、权重读取、自写 CUDA kernels、准确率及计时；
- `scripts/train_export.py`：训练同一网络并以固定格式导出 FP32 权重；
- `scripts/sweep_benchmark.py`：批量执行架构及性能搜索并生成 CSV；
- `weights/lenet.bin`：训练得到的模型参数；
- `results/threshold_sweep.csv`：96% 阈值附近候选的三轮计时结果；
- `大作业（一）(1).pdf`：课程题目原文。

## 计时口径

程序先执行若干轮不计时的预热。`CUDA kernels` 使用 CUDA Event 对每个 batch
计时；`--repeats 30` 在同一进程中重复前向并报告平均值。`End-to-end` 包括
一次 H2D、平均 kernel 时间、D2H 和 CPU argmax。模型权重加载、MNIST 文件读取
及预热不计入二者。

用于回答题目“与 NVIDIA library 相比”的参考测试（不属于最终推理路径）：

```bash
make baseline
```

H20 上三轮中位数（batch=10,000，FP16）：自写 CUDA 为 6.866 ms，端到端
8.127 ms，PyTorch/cuDNN 参考为 8.510 ms。对于这个很小的网络，单 kernel 融合
消除了逐层调用和中间显存往返，因此自写实现达到参考实现吞吐率的 123.94%。

当前机器上的完整实验记录见 `REPORT.md`。
