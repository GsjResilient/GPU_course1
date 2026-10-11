NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.4  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026  


> 教学级 GPGPU 仿真套件：从高级语言 Kernel 编写，到指令级调试，再到周期级性能分析。

NutShellGPU 提供一套完整的"软件 + 硬件"教学平台，帮助学生理解通用图形处理器（GPGPU）的编程模型、微结构时序与调试方法。本项目基于 NTAS1 指令集，包含功能模拟器、周期模拟器、SystemVerilog RTL 以及完整工具链（编译器 NTCC、汇编器 NTASM、调试器 NTDB、主机运行库 ntruntime）。

## 版本改进

- **1.1 CycleSim 周期模拟器**（作者：明同辉，26秋）：新增 T01–T07 时序机制（到期写回、指令差异延迟、依赖互锁、寄存器 Bank 冲突、内存时序等），性能从"跑对"变为"可量化"
- **1.2 调试器 + 官方工具链**（作者：王一锴，26秋）：模拟器增加调试钩子，NTDB 支持源码级断点/单步/Lane 切换/SIMT 栈观察；补齐 NTCC、NTASM、ntrun、ntruntime 完整文档
- **1.3 NutCC**（作者：崔文君，26秋）：a .cpp compatible compiler on LLVM Clang 18.1.x 和 Apple Clang 21.0.0
- **1.4 多模块会话复用与直接初始化**（作者：何贵意，26秋）：ntsim 功能后端直接初始化 FuncSim；Session 支持同一会话多次启动内核、加载多模块并按身份切换、原地写入已有分配、动态共享内存；原有构造与启动参数保持兼容

主线：**1.1 证明性能可测 → 1.2 让学生能自己写 kernel 并源码级调试 → 1.3 引入真实 C++ 语法编译路线 → 1.4 运行库能力补齐（会话复用 / 多模块 / 直接初始化）**。详细变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## 1. 项目组成

| 目录 | 说明 |
|---|---|
| `NutShellGPU_spec/` | 架构规格书（11 册），权威定义 |
| `NutShellGPU_sim/` | C++ 模拟器：功能模拟器（FuncSim）+ 周期模拟器（CycleSim） |
| `NutShellGPU_hw/` | SystemVerilog RTL 硬件模型 |
| `toolchain/` | 官方 Python 工具链：编译器（ntcc.py）、汇编器（ntasm.py）、调试器（ntdb.py）、运行器（ntrun.py）、主机库（ntruntime.py）、原生桥接（native/ntsim.cpp） |
| `compiler/` | 社区贡献的实验性 C++ Kernel 编译器 NutCC（受限 C++17 子集 → NTAS1，详见 `compiler/README.md`） |
| `docs/` | NutCC 编译器设计文档 |
| `examples/` | 示例：TinyLeNet 网络、向量计算、教学样例（SAXPY / 矩阵乘 / 直方图） |

## 2. 快速上手

### 2.1 环境要求

- **Python** 3.10 及以上（运行工具链及示例）
- **C++17 编译器**（MSVC / GCC / Clang / Zig 任一）
- **CMake** 3.16 及以上（可选，用于 CMake 构建）
- **NumPy**（运行 LeNet 示例及模型验证）

### 2.2 第一步：构建原生桥接程序 ntsim

所有工具链命令都依赖 `ntsim`（C++ 原生桥接）。在项目根目录执行：

```powershell
python toolchain/build.py
```

构建成功后，`ntsim.exe` 位于 `build/toolchain/` 目录。如需自定义编译器或输出目录：

```powershell
python toolchain/build.py --cxx "C:/Program Files/Microsoft Visual Studio/2026/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" --build-dir out/toolchain
```

如需同时构建并运行模拟器回归测试：

```powershell
python toolchain/build.py --tests --model-runner
```

### 2.3 第二步：编译第一个 Kernel

示例目录已自带 SAXPY 程序（Y = a·X + Y）[examples/toolchain/saxpy.ntcu](examples/toolchain/saxpy.ntcu)，全文仅 7 行：

```ntcu
__global__ void saxpy(float* x, float* y, float a, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程号
    if (i < n) {                                     // 尾部线程屏蔽
        float value = fmaf(a, x[i], y[i]);           // 单条融合乘加指令
        y[i] = value;
    }
}
```

编译生成 NTAS1 模块（`-g` 生成源码调试信息）：

```powershell
python toolchain/ntcc.py examples/toolchain/saxpy.ntcu -g -o out/saxpy
```

输出产物：
- `out/saxpy.ntas`：64 位小端机器码
- `out/saxpy.ntm.json`：模块元数据（ABI、符号、源码映射、SHA-256）
- `out/saxpy.ntasm`：可重新汇编的文本汇编
- `out/saxpy.listing`：PC、机器码、源码行对照表

### 2.4 第三步：准备启动配置并运行

示例已自带配套启动文件 [examples/toolchain/saxpy.launch.json](examples/toolchain/saxpy.launch.json)：

```json
{
  "grid": 1,
  "block": 32,
  "arguments": {
    "x": [1, 2, 3, 4, 5],
    "y": [10, 20, 30, 40, 50],
    "a": 2.5,
    "n": 5
  }
}
```

含义：1 个 CTA、每 CTA 32 个线程（即 1 个 Warp）；参数名必须与 Kernel 形参（x、y、a、n）一一对应。

功能模拟运行：

```powershell
python toolchain/ntrun.py out/saxpy.ntm.json examples/toolchain/saxpy.launch.json --backend func
```

输出 JSON 中 `execution.reason` 为 `completed`，`buffers.y` 即计算结果。按 Y = 2.5·X + Y 手工验算：

```text
[12.5, 25.0, 37.5, 50.0, 62.5]
```

周期模拟运行（额外输出模拟周期数）：

```powershell
python toolchain/ntrun.py out/saxpy.ntm.json examples/toolchain/saxpy.launch.json --backend cycle --budget 10000000 -o out/saxpy.result.json
```

### 2.5 第四步：使用 NTDB 进行源码级调试

启动交互式调试：

```powershell
python toolchain/ntdb.py out/saxpy.ntm.json examples/toolchain/saxpy.launch.json
```

下面这组命令与示例自带的 [saxpy.debug.txt](examples/toolchain/saxpy.debug.txt) 一致，可直接照做：

```text
(ntdb) break 5         # 在第 5 行 y[i] = value 处设断点（停在指令执行前）
(ntdb) continue        # 运行至断点
(ntdb) where           # 查看当前 PC、源码行与源码上下文
(ntdb) print i         # 查看当前 Lane 的线程号
(ntdb) print value     # 查看 fmaf 的计算结果
(ntdb) lane 4          # 切换到第 4 号 Lane（本输入的最后一个有效元素）
(ntdb) print value     # 应显示 62.5
(ntdb) stack           # 查看 SIMT 栈与 active mask（if 造成的 Warp 内发散）
(ntdb) delete 5        # 删除断点
(ntdb) continue        # 运行至结束
(ntdb) memory 65560 20 # 读回 y 缓冲区 20 字节（即 5 个 float，首地址由 bump 分配确定）
(ntdb) quit
```

其他常用命令：`step`（源码级单步）、`stepi`（指令级单步）、`regs`（寄存器与谓词）、`warps`（全部活跃 Warp 状态）、`disasm`（反汇编）。注意：`print` 只能查看当前 Lane **已执行过声明**且仍在作用域内的变量，所以断点要设在变量赋值之后。

批处理模式（自动执行命令文件，输出 JSONL）：

```powershell
python toolchain/ntdb.py out/saxpy.ntm.json examples/toolchain/saxpy.launch.json --commands examples/toolchain/saxpy.debug.txt
```

## 3. 运行示例

### 3.1 教学样例（SAXPY / 矩阵乘 / 原子直方图）

```powershell
python examples/toolchain/teaching/run_tests.py --out out/teaching
```

脚本自动编译 `.ntcu`、在功能与周期双后端运行 28 组测试，并与 NumPy FP64 参考比对（atol=rtol=1e-5）。

### 3.2 LeNet 推理（功能 / 周期 / CUDA 对照）

```powershell
# 功能模拟（默认 3 张图片）
python examples/toolchain/lenet/run.py --backend func --out out/lenet_func

# 周期模拟
python examples/toolchain/lenet/run.py --backend cycle --out out/lenet_cycle

# CUDA 对照（需 NVIDIA GPU 及 NVRTC）
python examples/toolchain/lenet/run.py --backend cuda --out out/lenet_cuda
```

产物包含逐层张量、预测结果、编译统计（寄存器数、指令数、SHA-256）及周期数。

### 3.3 NutCC：用 C++ 语法编写 Kernel（实验性）

`compiler/` 提供另一条独立的编译路线：直接编译**受限 C++17 子集**（`.cpp`），借助 LLVM Clang 的类型分析生成 NTAS1 机器码。它与第 2 节的 Python 工具链（`.ntcu`）互不依赖，可按学习需要二选一。

环境要求：Python 3.9+、Git、LLVM Clang 18.1.x（Linux/macOS）；Windows 请使用 WSL 环境，详细安装与配置见 [compiler/PORTABILITY.md](compiler/PORTABILITY.md)。

```bash
# 在仓库根目录执行；完整测试门槛（任一失败或跳过均返回非零）
python3 compiler/tests/run_tests.py --all --out build/nutcc-test

# 编译 C++ Kernel 并在功能/周期后端运行
python3 compiler/nutcc.py compiler/examples/vector_add.cpp -O1 \
  --kernel vector_add --out build/vector_add
python3 compiler/examples/run_vector_add.py \
  --artifact build/vector_add.json --backend func --n 65
```

成功时输出 `PASS backend=func completed=true`。语言子集范围、O0/O1 优化模式及限制说明见 [compiler/README.md](compiler/README.md)。

### 3.4 TinyLeNet 传统夹具（model_runner）

如需使用 1.1 的模型夹具流程：

```powershell
# 构建 model_runner（需 --model-runner 选项）
python toolchain/build.py --model-runner

# 运行夹具
build/model_runner NutShellGPU_sim/tests/fixtures/residual_small out/residual_small NutShellGPU_sim/configs/timing_v11.ini

# 与独立参考比对
python NutShellGPU_sim/tests/check_model.py NutShellGPU_sim/tests/fixtures/residual_small out/residual_small
```

## 4. 文档导航

| 文档 | 内容 | 适用场景 |
|---|---|---|
| `NutShellGPU_spec/00_总览与配置参数.md` | 系统视图、关键参数、INI 配置 | 建立全局认知 |
| `NutShellGPU_spec/01_编程模型与ABI.md` | Grid/CTA/Warp/Thread、存储空间、启动 ABI | 理解编程模型 |
| `NutShellGPU_spec/02_NTAS1指令集手册.md` | 64 位定长 ISA、R/I/B/M/MI 五种格式 | 查阅指令编码 |
| `NutShellGPU_spec/03_NSM_SIMT核心微结构.md` | 五级流水线、SIMT 栈、记分牌、Operand Collector | 理解核心微结构 |
| `NutShellGPU_spec/04_存储系统微结构.md` | Shared/L1/L2/DRAM、NoC、访存调度 | 理解存储层次 |
| `NutShellGPU_spec/05_模拟器实现契约.md` | 模块划分、接口、确定性要求 | 深入实现细节 |
| `NutShellGPU_spec/06_1.1时序与修复说明.md` | 到期写回、依赖互锁、寄存器 Bank、内存时序 | 理解周期模型 |
| `NutShellGPU_spec/07_程序总时间测量.md` | 计时边界、复现方法 | 性能测量 |
| `NutShellGPU_spec/08_Toolchain使用手册.md` | 工具链体系、构建、编译/汇编/运行、主机 API | 使用工具链 |
| `NutShellGPU_spec/09_NTCC语言与指令集.md` | NTCC 语法、内建函数、内联汇编、操作码覆盖 | 编写 Kernel |
| `NutShellGPU_spec/10_NTDB调试与日志.md` | 调试命令、协议、输出格式、故障定位 | 调试程序 |
| `CHANGELOG.md` | 版本变更记录 | 了解演进历史 |

## 5. 构建与测试（模拟器本体）

如需独立构建 C++ 模拟器（不经过 Python 工具链）：

```powershell
cmake -S NutShellGPU_sim -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

可执行文件位于 `build/Release/`（Windows）或 `build/`（Linux/macOS）。

## 6. 常见问题

**Q: 运行 ntrun/ntdb 时报"找不到 ntsim"？**
A: 请先执行 `python toolchain/build.py`，或设置环境变量 `NTSIM_EXECUTABLE` 指向 `ntsim.exe` 的完整路径。

**Q: 编译报错"register budget exceeds R0..R253"？**
A: 1.2.0 版本无寄存器溢出（spill），请拆分 Kernel 或简化表达式。

**Q: 调试时 print 变量提示"variable unavailable"？**
A: 变量须已初始化、位于当前词法作用域内，且当前 Lane 执行过其声明。循环变量在循环结束后不可查询。

**Q: 周期后端运行报错"cycle execution: "（空消息）？**
A: 此为 1.1.1 已知问题，1.2.0 已修复。请确保使用最新 `ntsim` 可执行文件。

## 7. 许可与致谢

本项目基于教学目的开发，架构代号 Newton，芯片型号 NutShellGPU，计算核心 NSM。参考教材：Aamodt/Fung/Rogers《General-Purpose Graphics Processor Architectures》（Morgan & Claypool, 2018）。
