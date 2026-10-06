# 第 09 册：NTCC 语言与指令集


NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.3  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026  


> **注意**：本册描述官方 Python 工具链的 `.ntcu` 教学语言（NTCC 1.2.0）。`compiler/` 目录下的 **NutCC** 使用受限 C++17 子集，是另一条独立编译路线，语言范围不同，详见 `compiler/README.md`。

NTCC/NTASM 1.2.0。NTCC 使用 CUDA 风格的教学 kernel 语言，输入为 .ntcu、目标为 NTAS1；语言、二进制和主机接口与完整 CUDA C++ 不同。编译及运行入口见[第 08 册](08_Toolchain使用手册.md)。

## 1. 源语言

| 类别 | 支持范围 |
|---|---|
| 文件 | 单个 __global__ void kernel；完整消费输入 |
| 类型 | 32 位 int、uint、float；对应 64 位指针 |
| 声明与作用域 | 必须初始化；每条声明一个变量；支持独立语句块；不允许遮蔽 |
| 赋值 | 标量/下标赋值，+= -= *= /= %= &= \|= ^= <<= >>= |
| 算术 | + - *；signed int 的 / %；一元 + - |
| 位运算 | 整数 & \| ^ ~ << >>；int 右移为算术移位，uint 为逻辑移位 |
| 比较 | == != < <= > >= 可组成表达式，结果 int 0/1 |
| 逻辑 | !、&&、\|\|；&&/\|\| 逐 lane 短路，跳过的操作数不会产生访存或修改变量 |
| 自增自减 | 前置和后置 ++/--，限标量变量或下标；下标地址只计算一次 |
| 条件语句 | if/else、else if；可用单语句或花括号体；条件接受数值的零/非零判断 |
| 循环 | while；for(init;condition;update)，三部分均可省略；支持嵌套 |
| 循环控制 | break 退出最近循环；continue 跳过本次循环体，for 仍执行 update；return 退出当前 lane |
| 其他语句 | 空语句、独立表达式语句、块语句 |
| 坐标 | threadIdx/blockIdx/blockDim/gridDim 的 x/y/z |
| 访存 | 一维指针下标，int/uint 索引；指针局部复制 |
| 转换 | (int)/(uint) 显式位模式转换、同类型转换；不开放浮点转换 |
| 设备内联汇编 | asm 模板与 r/f/l/q 操作数约束，见本册第 4 节 |

```ntcu
__global__ void sums(uint* out, uint n) {
    uint lane = threadIdx.x;
    uint sum = 0;
    for (uint i = 0; i < n; ++i) {
        if (i == 2u) continue;
        if (i > lane) break;
        sum += i;
    }
    out[lane] = sum;
}
```

for 初始化变量只在该循环内可见；循环体声明不在 update 中可见。空条件表示真，程序必须自行保证终止。循环的 break/continue/return 均逐 lane 生效，内部使用运行掩码、当前迭代掩码和 VOTE.ANY 保持统一控制推进。

整数/十六进制字面量支持 u/U；浮点支持小数/指数及 f/F；支持 // 与块注释。赋值不是表达式；不支持逗号表达式、多变量声明或有副作用表达式之间的全部 C++ 求值顺序规则。

## 2. 内建函数

| 分类 | 函数与参数 |
|---|---|
| 融合乘加 | fmaf(float,float,float)；__imad(int/uint,int/uint,int/uint)，后者输出低 32 位 |
| 整数 | abs(int)、min/max(int/uint,int/uint) |
| 位操作 | __popc、__brev、__clz、__ctz、__find_msb、__bmsk；参数 int/uint，结果 uint |
| 浮点 | fabsf、rcpf、rsqrtf、sinf、cosf、exp2f、log2f、floorf、ceilf、truncf、nearbyintf；单 float 参数、float 结果 |
| 浮点 min/max | fminf/fmaxf(float,float)，采用 NTAS1 FMNMX，任一 NaN 输入使结果为 NaN |
| 原子 | atomicExch/Add/Min/Max/And/Or/Xor(int*/uint*, 同类型标量)，返回旧值 |
| 比较交换 | atomicCAS(int*/uint*, compare, value)，标量类型须与指针元素完全一致 |
| 线程块屏障 | __syncthreads()，仅顶层且 kernel 中无 return；不接受可能导致部分线程到达的结构化分支/循环用法 |

数学函数由模拟器执行并返回 FP32 结果；rcpf 显式计算倒数。位移量按低 5 位处理；__bmsk 使用低 5 位长度。__clz(0)/__ctz(0)=32，__find_msb(0)=32。

## 3. 类型与执行边界

- 不隐式混合整数与 float；int/uint 混合按 uint 的 32 位位模式处理。源语言 signed int 除余遵循 NTAS1 的除零规则，见本册第 5 节。
- 算术保持 32 位结果，浮点字面量为 FP32。整数溢出和数学函数的边界行为按 NTAS1 及其模拟器实现解释。
- a*b+c 分开执行；只有 fmaf 显式融合。
- 控制流最多嵌套 8 层；寄存器单调分配，R0..R253，无 spill。程序过大时应拆分 kernel。
- 内联汇编在顶层使用，且不能与逻辑 return 混用；它使用物理 SIMT 活跃掩码。需复杂原生控制流时，在模板内部显式编写 BRA/SSY。
- NTAS1 的 .ntas/.ntm.json 不兼容 PTX、SASS 或 cubin；主机 API 也不是 CUDA Runtime。

### 语法范围之外

- do/while、switch、goto、三元表达式、逗号表达式、赋值表达式。
- uint 除余、float 除法、整数/浮点转换及 half/double 类型。
- bool/char/short/long、普通向量类型、结构体、类、模板、lambda。
- const/volatile/restrict、typedef、宏、预处理和头文件。
- 用户设备函数、多 kernel、递归、通用函数调用与链接。
- 局部数组、共享/常量存储源声明、warp shuffle 高层函数。
- <<< >>>、cudaMalloc/cudaMemcpy、stream/event、动态并行及 CUDA 库。

## 4. 内联汇编

```ntcu
__global__ void arithmetic(uint* out) {
    uint a = 3u;
    uint b = 4u;
    uint value = 5u;
    asm("IMAD %0, %1, %2, 0, 0"
        : "+r"(value) : "r"(a), "r"(b));
    out[0] = value; // 17
}
```

asm 的形式为 `asm("template" : outputs : inputs);`。输出或输入为空时保留相应冒号；没有操作数可写 `asm("NOP");`。字符串可相邻拼接，用分号或转义换行分隔指令。模板内 `%0` 开始按先输出、后输入编号，编译器分配独立临时寄存器并在模板之后复制输出。

| 约束 | 类型/寄存器组 |
|---|---|
| r | 一个 int/uint |
| f | 一个 float |
| l | 一个指针，偶数起始的两个寄存器 |
| q | 四个标量变量，偶数起始的四寄存器组，写作 `"=q"(a,b,c,d)` |
| = 前缀 | 只写输出；读取写入前的内容没有定义 |
| + 前缀 | 读写输出，先复制原值，适用于 IMAD、BFI 等读取旧目的值的指令 |

输入只用 r/f/l/q，不加前缀。输出必须是已声明的变量；输入可为类型相符的表达式。宽指令必须使用足够宽的约束，禁止借助窄约束覆盖相邻寄存器，输入寄存器为只读。q 是内联汇编约束，不是源语言 uint4 类型。

模板禁止固定 R0..R255，使用 `%N` 与 RZ；SETP 的谓词目的可写 P0..P6，例如 `SETP P1, %0, %1, 18, 3`。守卫支持 @P1/@!P1 等；标签仅在一个模板内解析，绝对分支由编译器重定位到模块位置。模板不能跳转到其他 asm 或高层语句。

```ntcu
asm("LD128 %0, [%1+0], 80, 0"
    : "=q"(a,b,c,d) : "l"(input));
asm("ST128 %0, [%1+0], 80, 1"
    : : "q"(a,b,c,d), "l"(output));
```

asm 仅允许出现在高层顶层控制流；kernel 中存在 return 时也会拒绝。模板内部原生分支、屏障、访存、EXIT 的正确性由程序保证。不能依赖被跳过路径上的只写输出，也非完整指令和未开放模式会在编译阶段报错。

## 5. 操作码与文本汇编

ISA 共定义 71 个操作码，工具链开放其中 53 个；普通语法或内建函数可生成 41 个，另外 12 个通过内联汇编使用。表中“自动”项同样可用于内联汇编；NTASM 与 NTCC 共用能力和字段检查。支持范围同时受下文模式和操作数约束。

| 编号 | 操作码 | NTCC 入口 | NTASM | 不支持原因 |
|---|---|---|---|---|
| 0x00 | NOP | 内联 | 是 | — |
| 0x01 | MOV | 自动 | 是 | — |
| 0x02 | MOV32I | 自动 | 是 | — |
| 0x03 | S2R | 自动 | 是 | — |
| 0x04 | LDC | 自动 | 是 | — |
| 0x05 | CVTA | 内联 | 是 | — |
| 0x06 | SHFL | 否 | 否 | 方向与越界处理不符合规格完整语义 |
| 0x07 | VOTE | 自动 | 是 | — |
| 0x08 | PRMT | 否 | 否 | 数据源与选择字段实现和规格不一致 |
| 0x09 | SELP | 自动 | 是 | — |
| 0x10 | IADD | 自动 | 是 | — |
| 0x11 | IADD3 | 内联 | 是 | — |
| 0x12 | IMAD | 自动 | 是 | — |
| 0x13 | IMUL | 自动 | 是 | — |
| 0x14 | ISUB | 自动 | 是 | — |
| 0x15 | IMNMX | 自动 | 是 | — |
| 0x16 | LOP | 自动 | 是 | — |
| 0x17 | SHF | 内联 | 是 | — |
| 0x18 | SHL | 自动 | 是 | — |
| 0x19 | SHR | 自动 | 是 | — |
| 0x1A | SAR | 自动 | 是 | — |
| 0x1B | BFE | 内联 | 是 | — |
| 0x1C | BFI | 内联 | 是 | — |
| 0x1D | POPC | 自动 | 是 | — |
| 0x1E | BREV | 自动 | 是 | — |
| 0x1F | IABS | 自动 | 是 | — |
| 0x20 | INEG | 自动 | 是 | — |
| 0x21 | IDIV | 自动 | 是 | — |
| 0x22 | IREM | 自动 | 是 | — |
| 0x23 | LOP3 | 否 | 否 | I 编码没有源寄存器字段，执行却读取 a/c |
| 0x24 | BMSK | 自动 | 是 | — |
| 0x25 | FIND | 自动 | 是 | — |
| 0x30 | FADD | 自动 | 是 | — |
| 0x31 | FSUB | 自动 | 是 | — |
| 0x32 | FMUL | 自动 | 是 | — |
| 0x33 | FFMA | 自动 | 是 | — |
| 0x34 | FMNMX | 自动 | 是 | — |
| 0x35 | FSET | 内联 | 是 | — |
| 0x36 | F2F | 否 | 否 | 半精度转换存在截断和次正规数简化 |
| 0x37 | XCVT | 否 | 否 | 部分类型组合回退为原始位复制 |
| 0x38 | RCP | 自动 | 是 | — |
| 0x39 | RSQ | 自动 | 是 | — |
| 0x3A | MUFU | 自动 | 是 | — |
| 0x3B | FRND | 自动 | 是 | — |
| 0x3C | FABS | 自动 | 是 | — |
| 0x3D | FNEG | 自动 | 是 | — |
| 0x3E | FCMP | 否 | 否 | 功能执行分支缺失 |
| 0x3F | DADD | 否 | 否 | 功能执行分支缺失 |
| 0x40 | SETP | 自动 | 是 | — |
| 0x41 | SETPI | 否 | 否 | I 编码没有源寄存器字段，执行却读取 a |
| 0x42 | PLOP | 否 | 否 | 功能执行分支缺失 |
| 0x43 | PSET2 | 否 | 否 | 功能执行分支缺失 |
| 0x50 | BRA | 自动 | 是 | — |
| 0x51 | BRX | 否 | 否 | 多目标分组排序简化 |
| 0x52 | CALL | 否 | 否 | 配套通用 RET 不完整 |
| 0x53 | RET | 否 | 否 | 发散返回采用 lane 0 的地址 |
| 0x54 | SSY | 内联 | 是 | — |
| 0x55 | BAR | 自动 | 是 | — |
| 0x56 | MEMBAR | 否 | 否 | 没有实现不同 scope 的排序效果 |
| 0x57 | EXIT | 自动 | 是 | — |
| 0x58 | YIELD | 否 | 否 | 没有调度让步实现 |
| 0x59 | TRAP | 否 | 否 | 只推进 PC，没有陷阱效果 |
| 0x5A | BRKPT | 否 | 否 | 只推进 PC，没有指令断点效果 |
| 0x60 | LD | 自动 | 是 | — |
| 0x61 | ST | 自动 | 是 | — |
| 0x62 | LDU | 内联 | 是 | — |
| 0x63 | ATOM | 自动 | 是 | — |
| 0x64 | RED | 内联 | 是 | — |
| 0x65 | PREFETCH | 否 | 否 | 没有预取效果 |
| 0x66 | LD128 | 内联 | 是 | — |
| 0x67 | ST128 | 内联 | 是 | — |

### 模式与操作数范围

- R、I、B、M、MI 五种格式均有编码入口。I 模式当前仅开放 MOV32I；B 模式开放 BRA/SSY；LDC 固定 MI；LD/ST/LDU/LD128/ST128 支持 M/MI；ATOM/RED 使用原生解码器的 R 格式。
- 支持 @P0..@P6、@!P0..@!P6、@PT、@!PT。`.m0`..`.m3` 仅用于 FADD/FSUB/FMUL/FFMA 的舍入方式；拒绝未实现的饱和、FTZ 等修饰。
- 整数默认 32 位；IADD/ISUB 可用 x=6 指定寄存器对。IMUL/IMAD 的 LO/HI/WIDE 模式及有符号位用 x=0/1/8/9/16/17。64 位寄存器组要求偶数起点；宽结果不得越过 R254。
- IDIV/IREM 为 signed int，不接受 uint 除余语法。除零分别返回 0/被除数。INT_MIN / -1、abs(INT_MIN) 等宿主有符号溢出边界不在安全数值输入范围内。
- S2R 不开放简化实现的 SR_WARPID_IN_GRID=40。threadIdx/blockIdx/blockDim/gridDim 的三个坐标均按原生 SR 编号映射。
- SETP 开放 S32/U32/F32 类型，不把更宽类型映射为 U32。FSET 返回全 1/全 0 位模式；高层比较返回 int 的 1/0。
- LDC 按 bank 与立即数偏移读取，地址基寄存器无寻址作用。一般访存按 ISA 宽度和地址空间编码，支持寄存器或有符号立即数偏移。当前 32 位标量访存与原子路径使用低 32 位地址；运行库分配位于该范围。64 位指针 ABI 不代表所有访存路径都支持任意高地址。
- ATOM/RED 仅开放已实现的 U32 宽度、global/shared/local 空间及 0..11 原子操作。CAS 的比较源在 b 字段。源语言 atomic 内建函数使用 global 空间。
- BAR 支持合法 ID 与 SYNC/ARRIVE/WAIT 模式。每个参与线程必须遵循模拟器屏障协议；源语言 __syncthreads() 只在无 return 的顶层控制流开放，固定 BAR.SYNC 0。

汇编阶段检查模式、字段范围和寄存器组对齐。动态地址、数据竞争、屏障到达条件和循环终止仍由程序决定；合法编码不等于任意程序都能完成。

### 文本格式

```text
R:  OP[.mN] Rd, Ra, Rc, x, y[, b]
I:  MOV32I Rd, imm32, x
B:  BRA/SSY label[, x]
MI: LD/ST/LDU/LD128/ST128/LDC Rd, [Ra+offset], x, y
M:  LD/ST/LDU/LD128/ST128 Rd, [Ra+Rc], x, y
```

NOP 和 EXIT 可省略操作数。RZ 是 R255 的别名。x/y/b 为字节值；CAS 的 b 也可写寄存器名。M 格式自动设置 x[7]；MI 不允许该位，访存 y 的保留位不得占用格式判别位。寄存器组是否对齐由有效宽度检查。

B 模式 x=0 为相对当前 PC，x=1 为绝对模块地址，偏移单位均为 8 字节。标签必须唯一，允许末尾标签。立即数可写十进制或 0x 十六进制；MI 偏移为 -32768..32767；MOV32I 接受 signed/unsigned 32 位字面量。

```asm
MOV32I R0, 7, 0
MOV32I R1, 3, 0
IDIV R2, R0, R1, 0, 0
SETP R1, R0, R1, 4, 3
@P1 IADD3 R2, R2, R1, 0, 1
EXIT
```

独立汇编生成 .ntas 机器码，不自动生成参数 ABI、资源声明或源码调试符号；这些信息来自 NTCC 模块。没有通用链接器、宏或第 02 册全部友好汇编别名。

