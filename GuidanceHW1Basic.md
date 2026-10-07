# 大作业一（基础题）指导：用 Trae 写 CUDA CNN 推理，用 WorkBuddy 写报告

> 适用：2026 秋 GPU 课程大作业一·基础题  
> 目标：基于 CUDA 从零实现 CNN 推理，在 MNIST 上跑通、调优、出报告  
> 截止：10 月 28 日

---

## 1. 题目要求回顾

- **任务**：基于 CUDA 语言实现 CNN 推理（推荐 LeNet），在 MNIST 数据集上测试。
- **限制**：**不能调用 cuBLAS / cuDNN / cuFFT 等英伟达数学库**，所有算子手写。
- **包含模块**：卷积、池化、激活（ReLU）、全连接、以及网络的其它必要模块。
- **考核平台**：计算中心 H20 平台。
- **考核指标**：准确率、运行时间、报告 1 份。
- **评分档位**（准确率 + 运行时间综合）：

  | 档位 | 分数 |
  |---|---|
  | 第一档 | 30 |
  | 第二档 | 25 |
  | 第三档 | 20 |

- 可多次提交，每位同学单独评分。鼓励用 AI coding 工具（如 Trae）辅助开发。

---

## 2. 整体工作流

```
环境准备 → 网络设计 → 手写各算子 Kernel → 正确性验证 → 性能调优 → 写报告 → 提交
  ↑           ↑           ↑              ↑          ↑
  H20/CUDA   LeNet     Trae 辅助编码     NumPy     共享内存/
  + Trae    结构                        参考对齐    循环展开等
```

仓库里 `examples/tiny_lenet/` 提供了一份参考（conv / relu / pool / dense / classify），可以用来对拍，但**基础题要求你自己用 CUDA 实现并调优**，不能直接交现成的。

---

## 3. 环境准备

### 3.1 H20 平台

登录计算中心 H20 节点，确认 CUDA 环境：

```bash
nvcc --version          # 确认 CUDA 版本
nvidia-smi              # 确认 GPU 可见
```

### 3.2 MNIST 数据

下载并解压 MNIST（`https://huggingface.co/datasets/ylecun/mnist`）：

```text
train-images-idx3-ubyte   train-labels-idx1-ubyte
t10k-images-idx3-ubyte    t10k-labels-idx1-ubyte
```

用一个小脚本读成 `float32` 的 `NCHW` 张量，像素归一化到 `[0,1]`（除以 255）。

### 3.3 安装并打开 Trae

- Trae 是 AI 原生 IDE，自带对话式编码能力，可直接在编辑器里让 AI 帮你写/改/调 CUDA 代码。
- 下载安装后，用 Git 账号登录即可免费使用编码模型。
- 把你的作业目录作为工作区打开，让 Trae 能看到整个项目。

---

## 4. 用 Trae 完成 CUDA CNN 推理

### 4.1 怎么跟 Trae 对话（实用提示）

- **小步迭代**：一次只让 AI 做一件事，比如"写一个 3×3 卷积 kernel，输入 NCHW，权重 OIHW"。一次性塞太多需求容易错。
- **给上下文**：把已经写好的 kernel、数据布局约定（NCHW、步长、补零）贴给它，避免 AI 自由发挥。
- **要它解释关键索引计算**：卷积里 `t -> (n, co, oy, ox)` 的解码和 `w[((co*CI+ci)*3+ky)*3+kx]` 这种索引最容易错，让 Trae 用一句话说明白，你再核一遍。
- **不要让它用库**：明确说"禁止 cudnn/cublas，纯手写 CUDA kernel"。
- **改代码用 Edit，不是让它重写整段**：Trae 的 Edit 能精确改几行，比整段重写更稳。

### 4.2 建议的项目结构

```text
hw1_basic/
├── kernels.cu           # conv / relu / pool / dense / classify 等 kernel
├── cnn.h / cnn.cu       # host 端：分配显存、传参、逐层 launch、计时
├── main.cu              # 读 MNIST → 加载权重 → 前向 → 统计准确率/时间
├── mnist_loader.py      # 读 idx 数据，导出权重/输入（可选）
└── README.md            # 运行说明
```

权重来源：可以用 PyTorch / Keras 训一个 LeNet，把权重导成 `.bin`；也可以直接用仓库 `examples/tiny_lenet/assets/weights.bin` 当参考（注意它是 Fashion-MNIST，用于 MNIST 需自己训）。

### 4.3 网络结构（推荐 LeNet 变体）

以仓库 TinyLeNet 为例（9,098 参数，够小够快，便于调优）：

```text
输入 1×28×28
  → Conv(1→8, 3×3, stride=1, pad=1)  → ReLU → MaxPool(2×2)   → 8×14×14
  → Conv(8→16, 3×3, stride=1, pad=1) → ReLU → MaxPool(2×2)   → 16×7×7
  → Flatten → Dense(16*7*7 → 10)
  → Argmax
```

你也可以用更经典的 LeNet-5（Conv 5×5、两层 FC），只要准确率能上去就行。

### 4.4 各算子 Kernel 实现要点

下面给的是**参考骨架**，对应仓库 [examples/tiny_lenet/kernels.cu](examples/tiny_lenet/kernels.cu)。基础阶段先写"一线程算一个输出元素"的朴素版本，跑通后再优化。

#### (1) 卷积 conv

```cpp
// 输入 x: N×CI×H×H, 权重 w: CO×CI×3×3 (OIHW), 偏置 b: CO
// 输出 y: N×CO×H×H  (stride=1, pad=1 时输出尺寸不变)
extern "C" __global__
void conv(const float* x, const float* w, const float* b,
          float* y, int N, int CI, int CO, int H) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * CO * H * H) return;
    int ox = t % H;
    int oy = (t / H) % H;
    int co = (t / (H * H)) % CO;
    int n  = t / (CO * H * H);

    float sum = b[co];
    for (int ci = 0; ci < CI; ++ci)
        for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
                int iy = oy + ky - 1, ix = ox + kx - 1;  // pad=1
                if (iy >= 0 && iy < H && ix >= 0 && ix < H)
                    sum += x[((n*CI + ci)*H + iy)*H + ix]
                         * w[((co*CI + ci)*3 + ky)*3 + kx];
            }
    y[t] = sum;
}
```

**Trae 提示词参考**：
> 帮我写一个 3×3 卷积的 CUDA kernel。输入 x 是 NCHW float32，权重是 OIHW，步长 1 补零 1，输出尺寸和输入一样。一一个线程算一个输出元素，禁止用 cuDNN。

#### (2) ReLU

```cpp
extern "C" __global__
void relu(const float* x, float* y, int count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) y[t] = fmaxf(x[t], 0.0f);
}
```

#### (3) 最大池化 pool（2×2, stride 2）

```cpp
extern "C" __global__
void pool(const float* x, float* y, int N, int C, int H) {
    int Q = H / 2;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * C * Q * Q) return;
    int ox = t % Q;
    int oy = (t / Q) % Q;
    int c  = (t / (Q * Q)) % C;
    int n  = t / (C * Q * Q);

    int base = ((n*C + c)*H + oy*2)*H + ox*2;
    float best = x[base];
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx)
            best = fmaxf(best, x[base + dy*H + dx]);
    y[t] = best;
}
```

#### (4) 全连接 dense

```cpp
extern "C" __global__
void dense(const float* x, const float* w, const float* b,
           float* y, int N, int CI, int CO) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N * CO) return;
    int n = t / CO, co = t % CO;
    float sum = b[co];
    for (int ci = 0; ci < CI; ++ci)
        sum += x[n*CI + ci] * w[co*CI + ci];
    y[t] = sum;
}
```

#### (5) 分类 argmax

```cpp
extern "C" __global__
void classify(const float* logits, int* pred, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int best = 0;
    for (int c = 1; c < 10; ++c)
        if (logits[n*10 + c] > logits[n*10 + best]) best = c;
    pred[n] = best;
}
```

### 4.5 Host 端编排与计时

用 `cudaMalloc` / `cudaMemcpy` 在每层之间搬运数据，逐层 launch kernel。**计时要包住 kernel 执行**，用 `cudaEventRecord`：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// 逐层 launch
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

注意：第一次 launch 会有懒加载开销，**多跑几次取稳定值**（或 warm-up 后再计时）。

### 4.6 正确性验证

最容易翻车的是索引和数据布局。建议：

1. **单张图片逐层对拍**：用 NumPy 写一份前向参考（`np.convolve` 或直接展开），把你的 CUDA 每一层输出和 NumPy 逐层对比，误差在 `1e-5` 内算过。
2. **小 batch 先跑通**：先用 N=1、N=8 验证，再上全量 10000 张测试集。
3. **让 Trae 帮你写对拍脚本**：
   > 帮我写一个 Python 脚本，用 NumPy 实现上面那个 conv kernel 的前向（OIHW 权重，stride1 pad1），输出 NCHW，用来和 CUDA 结果对比。

### 4.7 调优思路（冲击第一档）

朴素版准确率够，但运行时间大概率只能拿第三档。下面是可以让 Trae 帮你逐个尝试的优化点：

| 优化点 | 思路 | 预期收益 |
|---|---|---|
| **共享内存** | 把输入 tile 和权重块搬进 `__shared__`，减少全局内存访问 | 大 |
| **循环展开** | `#pragma unroll` 展开 `ky/kx` 小循环 | 中 |
| **常量内存** | 权重放到 `__constant__`（权重小的时候） | 中 |
| **减少分支** | pad 用边界判断改 pre-padding，或用 `__ldg` 只读缓存 | 小-中 |
| **合并访存** | 让 warp 内线程访问连续地址（按 `co` 维度相邻） | 中 |
| **向量访存** | `float4` 一次读 4 个 float | 中 |
| **block 大小调优** | 128 / 256 / 512 分别试，找 occupancy 最优 | 中 |

让 Trae 帮你改的时候，一次只改一个优化点，改完跑一次时间对比，确认确实变快了再叠下一个。典型提示词：

> 把这个 conv kernel 改成用共享内存做输入 tiling（每个 block 算一块输出，把对应的输入和权重 tile 搬到 shared memory），保持输出不变，其它不变。改完给我 diff。

---

## 5. 用 WorkBuddy 写报告

### 5.1 WorkBuddy 是什么

WorkBuddy（腾讯 WorkBuddy）是一个 AI Agent 桌面工作台，**用一句话描述需求，它会自己拆解任务、读写本地文件、生成文档**。写报告这种"有结构、要排版、要整理数据"的活非常适合它。

- 模式：默认 **Work 模式**（文档生成、数据分析、报告撰写），不要用 Coding 模式。
- 它能直接读取你授权的本地文件夹，所以可以把你的代码、运行结果、表格直接丢给它整理。

### 5.2 报告应包含的内容（评分参考）

1. **实验目的与要求**：题目复述，你实现了哪些模块。
2. **网络结构**：你用的 CNN 结构（画个图或给表格），参数量。
3. **实现细节**：
   - 各算子的实现思路（卷积索引方式、池化方式…）
   - 你做了哪些优化（共享内存、循环展开、block 大小…），每个优化的动机
4. **实验环境**：H20 平台、GPU 型号、CUDA 版本。
5. **实验结果**：
   - **准确率**（MNIST 测试集）
   - **运行时间**（前向推理总时间，单位 ms，说明是否 warm-up、batch 大小）
   - 调优前后的时间对比表（最好有）
   - 与 PyTorch / cuDNN 版本的对比（"与英伟达 library 相比达到几成功力"）
6. **分析与总结**：瓶颈在哪，还能怎么优化，遇到的坑。
7. **附录**：核心代码片段（不要全贴，挑有代表性的）。

### 5.3 具体操作步骤

**第一步：准备素材**

把这些文件放到一个文件夹里，让 WorkBuddy 能读到：

- 你的 `kernels.cu`、`main.cu`
- 运行结果：准确率数值、各优化版本的运行时间表（存成 `.txt` 或 `.csv` 最好）
- 环境信息：`nvidia-smi` 输出、`nvcc --version` 输出

**第二步：在 WorkBuddy 里下发任务**

切换到 Work 模式，输入类似这样的指令（一句话说清目标和产出）：

> 我要写一份"大作业一基础题报告"。项目是用 CUDA 手写 CNN（LeNet）在 MNIST 上做推理，没用 cuDNN。请帮我生成一份结构完整的报告（Markdown），包含：实验目的、网络结构（LeNet 变体）、各算子实现说明、调优思路、实验结果（准确率 98.7%，单张推理平均 0.32ms）、与 PyTorch cuDNN 对比、总结。我的 kernels.cu 在 ./hw1_basic/kernels.cu，运行时间数据在 ./results/timing.csv，请读取后把关键代码片段和数据嵌进去。

**第三步：验收与迭代**

- WorkBuddy 会生成报告并展示在右侧结果面板。
- 不满意的地方直接对话改，比如：
  - "把调优那部分加一个表格，列出每个优化点带来的加速比"
  - "卷积那段代码换成共享内存版本的，不要朴素版"
  - "加一个 H20 平台的环境信息章节"
- 最后导出为 Markdown / PDF。

**小贴士**：

- 数据要自己核一遍，AI 可能会把数字抄错。
- 代码片段让它从你真实文件里截取，不要让它"现编"。
- 如果报告里有图表需求（比如优化前后时间对比柱状图），可以让 WorkBuddy 先生成表格，再让它画图。

---

## 6. 提交清单

- [ ] 可编译运行的 CUDA 源码（`kernels.cu`、`main.cu` 等）
- [ ] 运行说明 `README.md`（怎么编译、怎么跑、MNIST 数据放哪）
- [ ] 报告 1 份（PDF 或 Markdown）
- [ ] 准确率和运行时间的实测数据

提交前在 H20 上**完整跑一遍测试集**，把准确率和时间记下来写进报告。

---

## 7. 常见问题

**Q: 能直接用仓库里 tiny_lenet 的 kernels.cu 吗？**  
A: 可以参考、对拍，但基础题要求你自己实现并调优。直接交现成的不加分，调过才有分。

**Q: 准确率上不去怎么办？**  
A: 先单张逐层和 NumPy 对拍，定位是哪层算错了。卷积的索引（pad、OIHW）和 dense 的权重布局最容易错。

**Q: 运行时间太慢（拿不到第一档）？**  
A: 朴素卷积是 O(N·CO·CI·H²·K²)，算力没压满。优先上**共享内存 tiling** + **循环展开**，再调 block 大小。

**Q: Trae 生成的代码编译不过？**  
A: 把报错贴回去让它修。CUDA 常见坑：`<<<>>>` 语法在 `.cpp` 里不行（要用 `.cu`）、`extern "C"` 加在 kernel 上、device 函数不能递归。

**Q: WorkBuddy 读不到我的文件？**  
A: 检查是否授权了对应文件夹，或者直接把文件拖进对话窗口上传。

---

祝你拿满分 🚀
