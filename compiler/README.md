# NutCC 0.1.0-test.1

NutShellGPU 1.10 的实验性 C++ kernel 编译器：复用 Clang 类型分析，生成真实 NTAS1 机器码，在原版 FuncSim / CycleSim 上执行。只新增 `compiler/` 及文档；不使用 NVIDIA 计算库。

这是明确受限的 C++17 子集，**不是完整 C++、CUDA 编译器，也不是 GPU 1.2 或 RTL 验证版本**。分支为 `test/nutcc-v0.1`，本地测试标签为 `nutcc-v0.1.0-test.1`（仅完整门槛通过后创建）。

## 前提与快速测试

当前支持并验证 macOS ARM64、Apple Clang **21.0.0**、Python **3.9+**。JSON AST 随 Clang 版本变化，其他版本明确拒绝。需要 `clang++`、`python3`、`git` 在 PATH。无需 CMake、CUDA Toolkit、NumPy、PyTorch。构建最多使用两个并行 C++ 编译进程。

```sh
cd "/Users/molly/projects/GPU编程/NutShellGPU"
git switch test/nutcc-v0.1

# 完整门槛：任何失败、跳过、缺依赖、测试期间源码变化均返回非零。
python3 compiler/tests/run_tests.py --all --out build/nutcc-test

# 编译源文件，得到 vector_add.ntbin / .ntasm / .json。
python3 compiler/nutcc.py compiler/examples/vector_add.cpp \
  --kernel vector_add --out build/nutcc-manual/vector_add

# 分别消费刚才生成的产物；runner 不会重新编译 kernel。
python3 compiler/examples/run_vector_add.py \
  --artifact build/nutcc-manual/vector_add.json --backend func --n 65
python3 compiler/examples/run_vector_add.py \
  --artifact build/nutcc-manual/vector_add.json --backend cycle --n 65
```

后两条成功时输出 `PASS backend=func|cycle n=65 completed=true`。示例逐元素比较可精确表示的 CPU 加法结果。完整报告为 `build/nutcc-test/report.json`，包含当前提交、源文件/产物 SHA-256、命令、版本、日志位置及真实测试计数。每次执行有独立 `run-*` 目录，不复用旧测试响应；编译、输入、输出及期望文件可供复现。

只允许在最终提交上通过全部门槛、工作区干净且报告提交/源码哈希匹配后创建测试标签。不要将旧报告作为修改后代码的通过证明。单独调试可运行 `python3 -m unittest discover -s compiler/tests -v`，但它不替代完整门槛。

## 编写 kernel

```cpp
#include "nutcc.h"
NT_KERNEL void vector_add(const float* a, const float* b,
                          float* out, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
```

支持内容：

- `void` 入口；参数 `int`、`unsigned int`、`float` 及这些类型和 `unsigned char` 的指针；支持 `const` 输入指针。局部变量也支持 `bool`，必须先初始化。
- 同类型 32 位整数加减乘、位运算、比较；无符号移位有效计数为 0–31。FP32 加减乘、比较、取负；保持顺序，不融合 FMA。
- 数组下标、赋值、复合赋值、作用域；`threadIdx/blockIdx/blockDim/gridDim` 的 x/y/z。
- `if/else`、嵌套条件、带短路语义的 `&&/||`。
- `for (int k=0; k<C; ++k)`，C 为非负编译期常量、每循环最多 256 次；全部展开，展开后机器指令最多 4096 条。循环变量不能在体内更改。编译展开最多访问 65536 次语句节点；超限报 `E_EXPANSION_LIMIT`，防止不生成指令的深层循环拖住编译器。
- `unsigned char` 可以复制，或提升到 `int`；整数常量按类型写 `1` / `1U`，浮点常量写 `1.0f`。允许入口末尾无条件 `return;`。

拒绝而非猜测：隐式/显式整数浮点互转、int/uint 混用、整数窄化、除法、取模、运行时循环、提前 return、break/continue、函数调用、类、模板、STL、异常、动态分配、volatile、shared memory、barrier、原子、warp 原语、64 位整数算术及多文件链接。例如以下都编译失败：

```cpp
NT_KERNEL void bad(float* p) { p[0] = 1; } // int → float 未实现
NT_KERNEL void bad(int* p, int n) { for(int j=0;j<n;++j) p[j]=j; }
NT_KERNEL void bad(double* p) { p[0] = 1.0; }
```

下标越界、有符号溢出、非法移位等仍属于源程序错误。编译器不证明任意动态下标安全，也不保证错误 kernel 不破坏其模拟设备内存。

## 产物与运行 ABI

流水线为：Clang JSON AST → 类型/语法白名单 → 谓词化 IR → 保守活跃区间与寄存器分配 → NTAS1 编码。

`.ntbin` 是小端 64 位指令字流，`.ntasm` 是可读清单，`.json` 是最后发布的完整性标志，包含 SHA-256、参数布局及寄存器数。不能把 `.ntbin` 当成上游 `.ntas` 容器。编译失败返回非零；runner 校验 manifest 与机器码版本、长度及哈希。

Profile：`ntas1-sim-v1.10-low32-predicated-v1`。每个参数从 8 字节边界开始，标量占 4 字节，指针槽占 8 字节。实际缓冲区限低 4 GiB；runner 校验完整分配区间，随后加载低 32 位地址。参数区最多 4096 字节；单次输入数据合计最多 256 MiB；block 最多 1024 线程。使用 R0–R254，资源超限报错，不进行 spill。

通用 runner：

```sh
python3 compiler/run.py --artifact PATH/kernel.json --backend func \
  --launch PATH/launch.json --out build/my-run
```

`launch.json` 的二进制文件路径相对于此 JSON；标量按参数顺序传入：

```json
{
  "format": "nutcc-launch-v1",
  "grid": [3, 1, 1],
  "block": [32, 1, 1],
  "buffers": {"a": "a.bin", "b": "b.bin", "out": "zeros.bin"},
  "args": [
    {"buffer": "a", "offset": 0},
    {"buffer": "b", "offset": 0},
    {"buffer": "out", "offset": 0},
    65
  ]
}
```

FP32 文件为小端 binary32；本例三个文件均含 65 个元素。缓冲区引用重复时保留别名，偏移以字节计，必须符合元素对齐。成功导出每个缓冲区更新后的 `.bin`。默认有 wall timeout 和执行步数预算，并检查 launch、完成状态和前后各 32 字节哨兵；哨兵检测不能替代任意越界检测。

## 测试方法和局限

完整门槛有七类必过项：环境、新构建、编译器测试、原版八个 C++ 回归入口、原文件保护、实际手测、源码稳定性。测试涵盖 1/31/32/33/65/257 元素、二维/三维索引、短路访存 trace、分支快照、循环、别名、编码固定值、产物损坏、超时与资源超限。

FuncSim 和 CycleSim 共享功能语义，二者相等不能证明正确。因此整数使用独立 Python 参考，浮点使用不包含编译器/模拟器代码的 C++ binary32 oracle，严格 RN、禁 FMA，验证次正规数、正负零、无穷和 NaN 分类。测试原有八个入口不表示修复历史 GPU BUG；本阶段未执行可选 CNN 测试。

书中 GPU 架构知识用于三个具体位置：线程层级映射为 S2R；分支转换为保留父掩码的谓词执行；寄存器按活跃区间复用以控制每线程资源。本版不生成一般跳转或汇聚栈指令，只在末尾生成一个无条件 EXIT。CycleSim 输出的周期数是模型值，不是真实显卡运行时间；尚未做性能优化或 RTL 兼容认证。

常见编译诊断包含源文件、行列和错误码：`E_CLANG_VERSION`（前端版本）、`E_CLANG`（C++ 错误）、`E_CONVERSION`（未实现转换）、`E_UNINITIALIZED`、`E_REGISTERS`、`E_CODE_SIZE`、`E_ARTIFACT`。预期编译错误退出 2，内部错误退出 3；runner 和完整门槛失败均返回非零。
