# NutCC 工具链与跨平台使用


NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.4  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026  


NutCC 用 Clang 完成 C++ 语法和类型分析，再由自己的后端生成 NutShellGPU 指令。它不需要 Apple 专用 API，也不使用 NVIDIA 计算库；Clang 是外部编译工具，不是 C++ 标准库。

## 安装与选择

以下命令均在包含 `compiler/` 的仓库根目录执行。完整测试要求分支为 `test/nutcc-v0.1.2`；单独编译和运行不限制分支名。

### Ubuntu 24.04

```sh
sudo apt-get update
sudo apt-get install -y clang-18 g++ python3 git
export NUTCC_CLANG=clang++-18
unset NUTCC_CXX NUTCC_SYSROOT
python3 compiler/tests/run_tests.py --all --out build/nutcc-test
```

`g++` 包提供 Linux 上 Clang 编译运行器时需要的标准 C++ 头文件和库。NutCC 前端仍然使用 Clang 的 JSON AST，不能把 `NUTCC_CLANG` 换成 GCC。其他发行版请安装 LLVM Clang 18.1.x 及其宿主 C++ 开发环境；版本标识被识别不代表该发行版已经完成回归验证。

### macOS：普通 LLVM Clang

保留已安装的 Xcode Command Line Tools / SDK，再安装上游 Clang：

```sh
brew install llvm@18
export NUTCC_CLANG="$(brew --prefix llvm@18)/bin/clang++"
export NUTCC_SYSROOT="$(xcrun --show-sdk-path)"
python3 compiler/tests/run_tests.py --all --out build/nutcc-test
```

这不会替换系统 Clang。`NUTCC_SYSROOT` 显式选择本机 SDK，避免旧 Homebrew LLVM 包把链接器指向已不存在的 SDK 路径（仅设置 `SDKROOT` 在本机 LLVM 18 上不能修复链接器搜索路径）。它只影响宿主运行器、oracle 和回归程序的构建，不参与 NutShellGPU kernel 的无标准库前端分析；该选项写入构建缓存键和测试报告。Linux 使用系统默认开发环境时无需设置。

也可选择现有 Apple Clang 21.0.0：

```sh
export NUTCC_CLANG=/usr/bin/clang++
unset NUTCC_CXX NUTCC_SYSROOT
python3 compiler/tests/run_tests.py --all --out build/nutcc-test-apple
```

先用 `"$NUTCC_CLANG" --version` 检查版本；其他 Apple Clang 版本未适配。

### Windows：先使用 WSL2

Windows 原生 Python、MSVC / clang-cl 不是此版本的支持路线。先在 Windows 安装 WSL2 和 Ubuntu 24.04，再在 Ubuntu 终端内克隆仓库并执行上面的 Ubuntu 步骤。建议将仓库放在 WSL 的 Linux 文件系统，例如 `~/src/nutshellgpu`，不要混用 Windows 的 Python 和编译器。

WSL 的操作系统、架构组合需要单独跑完整门槛；普通 Linux 测试通过不能替代 WSL 实测。此版不宣称已验证 Windows / WSL。参见 [Microsoft WSL 安装说明](https://learn.microsoft.com/windows/wsl/install)。

## 配置优先级

| 用途 | 选择顺序（从高到低） |
| --- | --- |
| kernel 前端 | CLI `--clang` / API `clang=` → `NUTCC_CLANG` → PATH 中的 `clang++` |
| 原生运行器、解码检查器、CPU oracle | API `build_tools(..., cxx=...)` → `NUTCC_CXX` → `NUTCC_CLANG` → PATH 中的 `clang++` |

环境变量的值是**一个可执行文件名或路径**，不要附带命令行参数。含空格的路径用引号包住。带目录的编译器路径及 `NUTCC_SYSROOT` 可以是相对路径，按调用者当前目录转为绝对路径，避免构建切换工作目录后失效。显式 `--clang` 只影响当次 kernel 编译；若还要运行产物或完整测试，推荐导出 `NUTCC_CLANG`，它会传递到子进程。

例如，macOS 上用 LLVM 前端、Apple 工具链构建宿主运行器：

```sh
export NUTCC_CLANG="$(brew --prefix llvm@18)/bin/clang++"
export NUTCC_CXX=/usr/bin/clang++
python3 compiler/tests/run_tests.py --all --out build/nutcc-test-mixed
```

切回同一工具链时执行 `unset NUTCC_CXX`。原生编译器必须支持 C++17 以及 `-frounding-math -ffp-contract=off`，并提供系统头文件和链接库。构建缓存包含编译器命令、版本、选项和源文件哈希。

## 手动验收

选择工具链后，执行：

```sh
python3 compiler/nutcc.py compiler/examples/vector_add.cpp -O1 \
  --kernel vector_add --out build/nutcc-manual/vector_add
python3 compiler/examples/run_vector_add.py \
  --artifact build/nutcc-manual/vector_add.json --backend func --n 65
python3 compiler/examples/run_vector_add.py \
  --artifact build/nutcc-manual/vector_add.json --backend cycle --n 65
```

后两条均应输出 `PASS` 和 `completed=true`。完整门槛还执行 O0/O1、两个模拟器、独立 CPU 参考、非法程序拒绝和原版八项回归。`report.json` 中的 `toolchain`、`versions`、`os`、`arch` 记录实际运行环境；不要只看 `clang --version` 就认定整个流程兼容。

## 实测范围与边界

0.1.2-test.1 的验证组合（2026-09-26）：

| 操作系统 / 架构 | kernel 前端 | 原生运行器 / oracle | 完整门槛 |
| --- | --- | --- | --- |
| macOS 26.6.2 / ARM64 | Apple Clang 21.0.0 | Apple Clang 21.0.0 | PASS |
| macOS 26.6.2 / ARM64 | Homebrew LLVM Clang 18.1.8 | 同版 LLVM，显式本机 SDK | PASS |
| Ubuntu 24.04 / ARM64，Linux 6.8 | Ubuntu LLVM Clang 18.1.3 | 同版 LLVM + libstdc++ | PASS |

每个完整门槛包含 **85 项 unittest、70 个端到端源码用例、O0/O1 合计 140 个编译产物、280 次后端执行、8 项上游回归及 8 项手测检查**。整数使用独立 Python 参考，浮点使用独立 C++ binary32 oracle。版本标识与 manifest 元数据的单元测试只检查接口策略，不充当跨平台执行证据。

Linux 在本机隔离 Ubuntu 虚拟机内实测：2 CPU、4 GiB 内存、16 GiB 磁盘、未共享宿主主目录；整个流程没有 Apple 工具链。macOS 还验证过 LLVM 前端 + Apple 原生工具链的混合配置。上述 ARM64 结果不代表 x86_64、其他发行版或 WSL 已通过。Windows 原生支持仍未实现。所有支持范围仍是 README 中的 C++17 kernel 子集，不因跨平台适配而扩展为完整 C++ / CUDA；此版仍未执行 RTL。

Clang 官方明确不保证不同大版本间 JSON AST 的稳定性，因此只接受已适配的 LLVM 18.1.x 和 Apple 21.0.0；不能简单取消版本检查。参考 [Clang JSON AST 接口说明](https://clang.llvm.org/doxygen/JSONNodeDumper_8h_source.html) 和 [Clang 官方安装指引](https://clang.llvm.org/get_started.html)。
