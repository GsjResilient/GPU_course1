# NutShellGPU 版本变更记录


<!-- =============================================================================
NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
Version 1.3
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
September 2026
============================================================================= -->

## 版本演进概览

- **1.1 CycleSim 周期模拟器**（作者：明同辉，26秋）：新增 T01–T07 时序机制（到期写回、指令差异延迟、依赖互锁、寄存器 Bank 冲突、内存时序等），性能从"跑对"变为"可量化"
- **1.2 调试器 + 官方工具链**（作者：王一锴，26秋）：模拟器增加调试钩子，NTDB 支持源码级断点/单步/Lane 切换/SIMT 栈观察；补齐 NTCC、NTASM、ntrun、ntruntime 完整文档
- **1.3 NutCC**（作者：崔文君，26秋）：a .cpp compatible compiler on LLVM Clang 18.1.x 和 Apple Clang 21.0.0

主线：**1.1 证明性能可测 → 1.2 让学生能自己写 kernel 并源码级调试 → 1.3 引入真实 C++ 语法编译路线**。

---

## 1.2.0

### 新增内容

#### 1. 调试器集成（FuncSim）
- 于功能模拟器核心中引入执行前回调机制（`debug_before`），支持以 SM、Warp、PC 三元组为粒度设置断点与单步执行。
- 实现可恢复的轮询游标（`debug_resume_sm` / `debug_resume_warp`），确保调试暂停后恢复执行时，当前调度轮内已推进的 Warp 不会被重复执行。
- 暂停期间 `sync()` 返回 `ERR_OK`，区分"暂停"与"完成"两种终态，便于上层调试器正确解析。

#### 2. 工具链（Toolchain 1.2.0）
- **NTCC 编译器**：支持 `.ntcu` 教学语言（类 CUDA C 语法），提供词法分析、递归下降/Pratt 表达式解析、类型与作用域检查、指令降低及单调寄存器分配；可生成 NTAS1 机器码、模块 ABI、源码映射、文本汇编及 listing。
- **NTASM 汇编器**：基于显式字段的 NTAS1 文本汇编，支持机器码往返验证。
- **ntsim 原生桥接**：UTF-8 行协议连接 FuncSim / CycleSim，提供内存分配、数据传输、Kernel 启动、断点管理、单步/连续执行及状态查询等原语。
- **ntrun 运行器**：命令行封装，支持功能/周期双后端一键运行。
- **ntruntime 主机库**：Python `Session` API，封装 `upload` / `download` / `launch` / `synchronize` 等操作。
- **NTDB 调试器**：源码级交互式调试，支持行断点、源码步进、指令步进、Warp/Lane 焦点切换、作用域变量查询、寄存器/谓词/SIMT 栈/内存查看及反汇编输出；同时支持批处理模式（命令文件 JSONL 输出）。

#### 3. 教学示例扩展
- `examples/toolchain/` 下新增 SAXPY、矩阵乘法、原子直方图三个教学样例（带中文注释），配套双后端回归验证脚本。
- `examples/toolchain/lenet/` 提供 LeNet 四个 Kernel 的 `.ntcu` 实现，可复用 TinyLeNet 固定权重与输入，支持功能模拟、周期模拟及 CUDA 对照运行。
- `examples/toolchain/teaching/` 提供 14 组固定种子的回归测试，覆盖空输入、线程块尾部、零长度点积与同桶原子竞争等边界场景。

#### 4. 文档补充
- 新增第 08 册《Toolchain 使用手册》：体系架构、环境构建、编译/汇编/运行流程、主机 API 及模块 ABI 说明。
- 新增第 09 册《NTCC 语言与指令集》：语法、内建函数、内联汇编约束及 53 个操作码覆盖范围。
- 新增第 10 册《NTDB 调试与日志》：调试命令、暂停/步进语义、原生协议、输出格式及故障定位指南。

#### 5. NutCC 实验性 C++ Kernel 编译器（社区贡献同步）
- 同步 Gitee 仓库 `compiler/` 目录（NutCC 0.1.2-test.1，贡献者 WenjunCui）：基于受限 C++17 子集，复用 LLVM Clang 18 / Apple Clang 21 的类型分析生成真实 NTAS1 机器码，可直接在本版本 FuncSim / CycleSim 上执行，提供 O0/O1 两级模式与完整测试门槛。
- 该编译器与本版本自带的 Python 工具链（NTCC `.ntcu`）相互独立、互不依赖；运行时桥接仅使用模拟器既有接口（`load_code`、`launch`、`run`、`step_one_round`），与 1.2.0 调试钩子兼容。
- 同步配套设计文档 `docs/superpowers/`。跨平台要求（Clang 版本、Windows WSL 路线）见 `compiler/PORTABILITY.md`。

### 缺陷修复
- **FIND 指令 MSVC 分支计算错误**：`isa/ntisa.hpp` 中 `find_msb()` 的 MSVC 分支误用 `31 - _BitScanReverse` 结果（BSR 返回值本身即最高置位位号，等价于 `31 - clz`），导致 MSVC 编译时 `__find_msb(16)` 错误返回 27（clz 值）而非 4；已改为直接返回 BSR 索引。GCC/Clang 分支公式正确，故该问题仅在 Visual Studio 环境暴露。修复后工具链 40 项回归与模拟器 7 套 C++ 测试全部通过。
- **ntsim 周期后端预算处理**：修复 `CycleSim::run` 因预算耗尽返回 `ERR_DEADLOCK` 时，ntsim 错误地使用 `s.error.msg`（此时消息为空）导致输出无意义空字符串的问题；现区分 `budget` 与 `completed` 两种停止原因，行为与功能后端一致。
- **版本号统一**：将分散于各组件中的版本标识（模拟器、工具链、ntsim、ntcc、ntdb、ntrun、ntasm、ntruntime、规格书等）统一修订为 1.2.0，消除 1.1.1 / 1.0.0 混用导致的认知偏差。
- **冗余文档清理**：删除项目根目录下与 `NutShellGPU_spec/` 内容重复的第 08–10 册旧副本（仍标注 1.0.0 / 1.1.1），统一以 `NutShellGPU_spec/` 为唯一权威文档来源。
- **入门教程勘误**：修正 README 快速上手中 SAXPY 启动配置示例参数名与 Kernel 形参不一致（误写为 vector_add 的 a/b/out）、调试示例断点早于变量声明导致 `print` 不可用的问题，并补充预期输出与逐命令注释。

---

## 1.1.0

### 新增内容

#### 1. 周期级微结构模型（CycleSim）
基于 1.0 功能语义，引入以下七类时序机制：
- **T01 到期写回**：发射与操作数收集阶段不提前修改架构状态；指令仅在 `ready_cycle` 到期且获得提交端口后，方执行功能语义并更新寄存器/内存。
- **T02 指令差异延迟**：搬运、整数、乘法、除法、浮点、SFU、DP、Shuffle、控制流等类别分别配置执行延迟。
- **T03 依赖互锁**：每 Warp 维护未提交指令互锁，覆盖 RAW、WAW、WAR、谓词、宽寄存器对及控制流依赖。
- **T04 寄存器组冲突**：GPR 四 Bank 端口竞争、重复源广播及跨收集器轮询仲裁。
- **T05 执行资源**：延迟与启动间隔分离；有限执行槽、Lane 数吞吐限制及统一提交端口竞争。
- **T06 内存时序**：扇区合并、Shared Bank 冲突、L1/L2 时序标签、有限 MSHR、NoC 带宽预约、DRAM Bank/行缓冲与总线服务。
- **T07 配置与统计**：纳秒到周期换算、配置合法性检查、1.1 统计字段扩充及带宽单位修正；新增 `simulated_seconds`（按 `cycles / (core_clock_ghz × 10^9)` 换算）。

#### 2. 程序运行时间测量
- 新增 `benchmark_process.py`，使用 Python 标准库在进程外测量完整运行时间（含装载、初始化、模拟、逐层导出与退出），记录重复测量、退出码、模拟周期、样本数、平台及文件哈希。

#### 3. 测试扩展
- 新增 `t_runtime_v2`、`t_timing_v11`、`t_repairs`、`t_stats_time` 等回归测试。
- 新增 `model_runner` 及残差/深度可分离网络夹具，支持逐层与独立 NumPy/PyTorch 参考比对。

### 缺陷修复（F01–F14）
- **F01**：多模块启动按加载的模块身份执行，避免同名或相同入口混用。
- **F02**：支持非零 `code_offset`，拒绝非法启动尺寸、入口及超额资源。
- **F03**：初始驻留之外的 CTA 继续补发；未发完不能误判完成。
- **F04**：静态与动态共享内存、寄存器、Warp 及 CTA 使用统一分配与回收账本。
- **F05**：非法/未对齐 PC 及代码越界返回明确错误。
- **F06**：可重用屏障按代登记到达状态，同代每 Warp 仅登记一次。
- **F07**：FP32 算术不再错误按整数舍入；编译设置保留舍入模式（MSVC `/fp:strict`，GCC/Clang `-frounding-math -ffp-contract=off`）。
- **F08**：RZ 读取恒零；SHFL 先快照所有源 Lane，再提交各 Lane 结果。
- **F09**：稀疏页号保留 64 位；跨页访问、复制、clear 的页指针生命周期正确。
- **F10**：禁止重复取同一条未提交指令；在途状态排空后方复用 CTA/Warp 槽。
- **F11**：LDC 宽度按 `ldc_width` 解码，不误用通用 `mem_width` 的对齐检查。
- **F12**：每次 `run` 使用增量预算；超预算失败；占用率按时间累计。
- **F13**：Flat 解码偏移与 Lane 基址互逆，Local 地址包含 NSM 身份；宽访存高字与 Local/Const 路径统一，访问执行边界检查。
- **F14**：原子读写失败向调用方传播；未实现的原子宽度明确拒绝；CVTA 地址转换不产生访存请求。

---

## 1.0.0

### 基础能力
- **架构规格**：6 册 Markdown 文档（总览与配置、编程模型与 ABI、NTAS1 指令集手册、NSM SIMT 核心微结构、存储系统微结构、模拟器实现契约）。
- **功能模拟器（FuncSim）**：NTAS1 指令精确执行、SIMT 栈状态机、五级流水线概念模型、CTA 调度与资源管理、分层内存（Global/Local/Shared/Const/Param）及稀疏页存储。
- **周期模拟器（CycleSim）**：基于 FuncSim 的简化名义延迟模型（1.0 时期），输出 `stats.json`。
- **RTL**：SystemVerilog 实现（NSM 顶层、SIMT 栈、记分牌、Operand Collector、L1D、NoC、L2、DRAM 控制器及测试平台）。
- **测试套件**：`t_decode`（解码器单元测试）、`t_saxpy`（端到端正确性）、`t_diverge`（SIMT 栈快照）。
- **示例**：`examples/tiny_lenet` 提供三层 TinyLeNet 固定权重及逐层测试。
