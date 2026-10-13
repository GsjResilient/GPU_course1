# 大作业一竞赛题：NutShellGPU MNIST CNN 推理

本目录实现第一题同一网络的 **NutShellGPU 纯推理**。不包含训练、反向传播或梯度下降，也不调用 NVIDIA 数学库。

## 最终网络

```text
1x28x28
  -> Conv3x3(1->4, valid) -> ReLU -> MaxPool2
  -> Conv3x3(4->8, valid) -> ReLU -> MaxPool2
  -> FC(200->16) -> ReLU -> FC(16->10) -> Argmax
```

- 参数量：3,722
- MNIST 测试集：9,712 / 10,000，准确率 **97.12%**
- 权重：`assets/lenet.bin`，与基础题最终版本完全相同
- 输入预处理：`(pixel / 255 - 0.1307) / 0.3081`

## 构建与运行

在仓库根目录执行：

```bash
python3 toolchain/build.py

# 功能模拟；会逐层与 NumPy FP32 参考对拍
python3 competition/run.py --backend func --samples 1

# 周期模拟；默认最终 optimized 方案与 block=32
python3 competition/run.py --backend cycle --samples 1

# 全部 10,000 张的固定权重准确率（只做前向，不训练）
python3 competition/evaluate_accuracy.py
```

如果 MNIST 不在默认搜索位置，增加：

```bash
--mnist-dir /path/to/MNIST/raw
```

可复现实验变量：

```bash
# 三种实现对比
python3 competition/run.py --backend cycle --variant fused   --block-size 32
python3 competition/run.py --backend cycle --variant unfused --block-size 32
python3 competition/run.py --backend cycle --variant optimized --block-size 32

# 批量与 block size 测试
python3 competition/run.py --backend cycle --samples 32 --block-size 64
python3 competition/run.py --backend cycle --samples 1  --block-size 128
```

每次运行会把完整参数、预测、标签、逐层周期/指令数、宿主墙钟时间和误差写入 `build/competition/`。周期是 NutShellGPU CycleSim 指标；宿主秒数会随机器负载变化。

## 最终优化

最终方案没有采用表面上直观的全融合实现，因为 CycleSim 实测融合卷积会减少可并行线程，周期反而增加。保留逐输出线程并行后，做了以下安全优化：

1. Conv 与 ReLU 融合；
2. 完整保留基础题第二层卷积的 11x11 中间张量，确保逐层结构相同；
3. 最后一层的 10 个类别并行计算，再单独 Argmax；
4. 单个 Session 复用所有模块、显存和权重，避免逐层重启模拟器；
5. 实测选择 32 threads/block。

结果和实验解释见 [REPORT.md](REPORT.md)，原始汇总位于 `results/`。
