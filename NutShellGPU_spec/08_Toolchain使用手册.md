# 第 08 册：Toolchain 使用手册


NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.3  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026  


> **注意**：本册描述官方 Python 工具链 NTCC 1.2.0（输入 `.ntcu` 教学语言）。`compiler/` 目录下的 **NutCC** 是独立的社区贡献 C++17 子集编译器，语言子集、构建方式和依赖均不同，详见 `compiler/README.md`。

## 1. 体系与版本

工具链包含 NTCC 编译器、NTASM 汇编器、NTDB 调试器和 Python 主机运行库，版本均为 1.2.0；原生桥接程序 ntsim 连接 NutShellGPU 1.2.0 的功能、周期后端。

```text
.ntcu → NTCC → .ntas / .ntm.json / .ntasm / .listing
                       ↓
              ntrun / Python API → ntsim → FuncSim / CycleSim
                       ↑
                      NTDB
```

NTCC 使用词法分析、递归下降/Pratt 表达式解析、类型与作用域检查、指令降低及单调寄存器分配。宿主负责数据传输和启动，设备计算由 NTAS1 指令执行。语言与汇编格式见[第 09 册](09_NTCC语言与指令集.md)，调试、协议和日志见[第 10 册](10_NTDB调试与日志.md)。

| 模拟器 | 接口与运行能力 |
|---|---|
| 1.1.0 | NTAS1 编码与参数 ABI 基线，可通过原 load_code 接口装载机器码；无工具链桥接与调试服务 |
| 1.2.0 | ntsim 支持 FuncSim/CycleSim；NTDB 在 FuncSim 上提供源码调试 |

模块格式为 `NTMODULE-1`，原生协议版本为 1。元数据 `simulator_min: 1.1.0` 表示 ISA/ABI 基线；`.ntm.json` 由工具链运行库读取，原 model_runner 使用 fixture 格式。

## 2. 环境与构建入口

NTCC/NTDB 使用 Python 3.10+ 标准库，ntsim 使用 C++17；LeNet 参考计算依赖 NumPy。以下命令在项目根目录执行。

```powershell
python toolchain/build.py
```

脚本自动查找 c++/clang++/g++/zig；`--cxx PATH` 选择编译器，`--build-dir DIR` 指定输出目录，`--tests` 运行模拟器回归程序，`--model-runner` 同时生成原 fixture 运行器。默认桥接路径为 `build/toolchain/ntsim.exe`（Unix 为 `ntsim`）；自定义位置通过 `NTSIM_EXECUTABLE` 指定。

CMake 入口为 `cmake -S toolchain -B build/toolchain`，随后 `cmake --build build/toolchain --config Release`。多配置生成器的桥接通常位于 `build/toolchain/Release/ntsim.exe`，该路径同样可设为 `NTSIM_EXECUTABLE`。构建脚本配置严格浮点选项，以保留舍入语义。

## 3. 编译、汇编与运行

NTCC 接受 `.ntcu` 源文件。项目的 CUDA 源文件使用 `.cu`。

```powershell
python toolchain/ntcc.py examples/toolchain/vector_add.ntcu -g -o out/vector
python toolchain/ntasm.py out/vector.ntasm -o out/vector.reassembled.ntas
python toolchain/ntrun.py out/vector.ntm.json examples/toolchain/vector_add.launch.json --backend func
python toolchain/ntrun.py out/vector.ntm.json examples/toolchain/vector_add.launch.json --backend cycle --budget 10000000 -o out/vector.result.json
```

NTCC 的 `-o/--output` 指定产物基础名并自动创建父目录；重复使用该名称会覆盖产物。`-g` 可选，当前总是生成 O0 调试信息。NTCC/NTDB 的 `--version` 输出工具版本。

| 产物 | 内容 |
|---|---|
| .ntas | 64 位小端指令序列，每条 8 字节 |
| .ntm.json | 参数 ABI、资源、代码哈希、源码快照、行表与符号表 |
| .ntasm | 可重新汇编的显式字段文本 |
| .listing | PC、机器码、源码行及编码字段 |

NTASM 生成机器码；ABI 和源码映射由 NTCC 生成。运行库校验机器码 SHA-256，因此 `.ntas` 与 `.ntm.json` 必须来自同一次编译。

启动配置示例：

```json
{"grid":2,"block":4,"arguments":{"a":[1,2,3,4,5],"b":[10,20,30,40,50],"out":[0,0,0,0,0],"n":5}}
```

参数名与 kernel 声明一致；指针参数的列表上传为独立缓冲区，标量按声明类型封送。grid/block 为正整数或 1–3 项整数列表，省略维为 1；每维 grid 上限 65535、总 CTA 上限 65535，每维 block 上限 1024、总线程上限 1024，并受寄存器等资源限制。

ntrun 默认使用 func，预算默认为 10000000、范围 1..100000000；功能预算单位是 warp 指令，周期预算单位是模拟周期。输出为 `execution/buffers` JSON，`-o` 可写入已存在目录。`reason=completed` 表示完成，预算或断点暂停会使 ntrun 报错。

## 4. 主机 API 与模块 ABI

```python
from pathlib import Path
import sys, struct
sys.path.insert(0, str(Path('toolchain').resolve()))
from ntruntime import Session
with Session('out/vector.ntm.json', backend='func') as device:
    a = device.upload(struct.pack('<5f', 1,2,3,4,5))
    b = device.upload(struct.pack('<5f', 10,20,30,40,50))
    out = device.upload(bytes(20))
    device.launch(2, 4, {'a':a, 'b':b, 'out':out, 'n':5})
    event = device.synchronize()
    values = struct.unpack('<5f', device.download(out,20))
```

| API | 行为 |
|---|---|
| Session(module, backend='func', executable=None) | 校验模块并启动会话；executable 优先于环境变量 |
| malloc(size) / upload(bytes) | 分配 1..16 MiB；upload 同时复制数据，返回设备地址 |
| download(address,size) | 读取已分配区间，单次上限 16 MiB |
| launch(grid,block,arguments) | 每会话一次启动 |
| synchronize(budget=10000000) | 执行至完成，其他停止原因抛出异常 |
| command(text) | 原生协议接口 |
| close() / with | 释放会话 |

多层网络为每层建立会话，由宿主传递层间张量。主机传输检查已分配区间；设备访存边界遵循模拟器地址空间规则。

每个模块包含一个 kernel。int/uint/float 参数各占 4 字节、4 字节对齐；指针占 8 字节、8 字节对齐，按声明顺序排列并通过 LDC bank 0 读取。指针使用偶数起始的寄存器对，数组元素均为 4 字节。可分配寄存器为 R0..R253，超限时报错。

元数据包括 `compiler_version/isa/simulator_min`、`kernel/registers/parameter_size/parameters`、`code_file/code_sha256`、`source/source_text/source_sha256`、`line_table/symbols` 和 `instructions/live_mask_register`。调试显示编译时嵌入的源码快照。

## 5. 样例

SAXPY 入口为 `python examples/toolchain/host_saxpy.py`。LeNet 的四个 .ntcu kernel 使用 for 循环，权重与输入位于 `examples/tiny_lenet/assets`。

```powershell
python examples/toolchain/lenet/run.py --backend func --out out/lenet_func
python examples/toolchain/lenet/run.py --backend cycle --out out/lenet_cycle
python examples/toolchain/lenet/run.py --backend func --samples 1 --debug --out out/lenet_debug
```

`--samples` 为 1..3，默认 3。运行器执行六个网络阶段，以独立 NumPy 参考检查中间张量（atol=rtol=1e-4）和分类索引。产物及计时字段见第 10 册。

CUDA 对照入口为 `python examples/toolchain/lenet/run.py --backend cuda --out out/lenet_cuda`，向量对照为 `python examples/toolchain/run_vector_cuda.py`。该路径使用 NVIDIA GPU、驱动及 NVRTC；`NVRTC_LIBRARY` 指定动态库位置。适配器给 .ntcu 文本添加 uint 定义及 extern "C" 包装后交给 NVRTC，与 NTCC 的 NTAS1 生成路径分开。


### 教学计算测试

`examples/toolchain/teaching` 提供三个带中文注释的 .ntcu 样例：

| 样例 | 计算目标与教学要点 |
|---|---|
| [saxpy.ntcu](../examples/toolchain/teaching/saxpy.ntcu) | out=a*x+y；一维线程索引、尾部判断与融合乘加 |
| [matmul.ntcu](../examples/toolchain/teaching/matmul.ntcu) | 行优先矩阵乘法；二维线程布局、for 点积与矩形边界 |
| [histogram.ntcu](../examples/toolchain/teaching/histogram.ntcu) | 16 桶直方图；受约束内联汇编计算桶地址，atomicAdd 处理并发计数 |

```powershell
python examples/toolchain/teaching/run_tests.py --out out/teaching
```

脚本使用已有 ntsim，路径可通过 `--executable` 或 `NTSIM_EXECUTABLE` 指定。固定种子生成 14 组输入，在功能和周期后端共运行 28 次；覆盖空输入、线程块尾部、零长度点积与同桶原子竞争。浮点结果与 NumPy FP64 参考按 atol=rtol=1e-5 比较，整数结果完全匹配，并检查汇编往返、保护区、输入完整性及双后端输出一致性。

直方图将输入低 4 位作为桶编号，bins 初始为零；脚本检查分配区间低于 2^32，以满足该样例的低位地址加法。运行产物位于 `--out`：compiled 为模块，inputs 为启动配置，outputs 为实际/参考数组，results.json 为逐项结果。
