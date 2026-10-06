# 第 10 册：NTDB 调试与日志


NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education  
Version 1.3  
Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN  
Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)  
References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018  
September 2026  


NTDB 1.2.0 基于 NutShellGPU 1.2.0 的 FuncSim 执行前钩子提供源码调试；CycleSim 提供运行结果和周期统计。

## 1. 使用流程

编译后的模块及启动配置与普通运行相同：

```powershell
python toolchain/ntdb.py out/vector.ntm.json examples/toolchain/vector_add.launch.json
python toolchain/ntdb.py out/vector.ntm.json examples/toolchain/vector_add.launch.json --commands examples/toolchain/vector_add.debug.txt
```

交互启动后参数已上传、kernel 尚未运行，默认观察 SM=0、warp=0、lane=0。向量示例可依次使用 `break 5`、`continue`、`print value`、`delete 5`、`step`、`continue`。批处理逐行执行命令文件，输出 `{command,result}` JSONL，错误时中止并返回非零。

| 命令 | 行为 |
|---|---|
| `break LINE` / `b LINE` | 在该行每个连续机器码范围的入口设置断点 |
| `delete LINE` | 删除该行断点 |
| `continue` / `c` | 执行至断点、结束、故障或指令预算 |
| `stepi` / `si` | 调度顺序中的一条 warp 指令 |
| `step` / `s` | 源码行级推进焦点 warp |
| `where` | 当前 PC、源码行和嵌入源码文本 |
| `warps` | 活跃 warp、CTA、PC 与退出状态 |
| `warp SM SLOT` | 切换观察 warp |
| `lane N` | 切换观察 lane，必须为有效 lane |
| `print NAME` / `p NAME` | 当前 lane 的源码变量值与寄存器位置 |
| `regs` | 通用寄存器和谓词 |
| `stack` | SIMT 栈与硬件 active mask |
| `memory ADDRESS BYTES` | 读取已分配全局内存，返回十六进制字节 |
| `disasm` | PC、源码行与编码字段 |
| `quit` | 退出并释放会话 |

## 2. 暂停、步进与变量

断点停在指令执行前，暂停整个功能执行器。调度轮的 NSM/warp 游标保存当前位置，恢复时继续该轮。当前停止位置仅跳过一次断点检查；其他 warp 或下一次循环到达同一 PC 时仍触发断点。断点集合独立于机器码。

stepi 执行调度顺序中的一条 warp 指令；step 保持其他 warp 推进，直到焦点 warp 换行、遇到断点、退出或耗尽预算。warp/lane 命令只改变观察焦点。continue 预算为 10000000 条 warp 指令，step 最多推进 100000 次调度单步。

一行源码可对应多个 PC 区间，行断点覆盖每个连续区间的入口。print 接受变量名；变量须已初始化、位于词法作用域内，且当前 lane 执行过其声明。指针合并两个寄存器，float 按 IEEE 位模式解释。SIMT 栈反映线程执行分歧，区别于函数调用栈。

for 的初始化、条件和 update 按各自源码行映射。continue 后执行 update；break 跳过该 lane 的 update。循环初始化变量在循环结束后不可查询，循环体变量在 update 处也已离开作用域。

内联汇编映射到 asm 开始行；模板内部用 stepi/disasm/regs 查看。输出变量在模板结束后复制回高层变量，模板执行中的最新值位于临时寄存器。

当前调试接口提供无条件源码/PC 断点及只读状态查询；条件断点、数据观察点、反向执行、变量写入和 IDE/DAP 接口不在支持范围。

## 3. 原生接口与执行事件

启动方式为 `ntsim code.ntas registers parameter_bytes func|cycle`。协议版本为 1：UTF-8 行命令，每条返回一行 JSON；握手为首行响应。

```text
alloc BYTES
write ADDRESS HEX
read ADDRESS BYTES
launch GRID_X BLOCK_X PARAM_HEX
launch3 GRID_X GRID_Y GRID_Z BLOCK_X BLOCK_Y BLOCK_Z PARAM_HEX
break PC
delete PC
run BUDGET
step 1
state SM WARP LANE
warps
quit
```

无参数字节用 `-`，地址和 PC 用十进制数字。launch3 的三维限制见第 08 册。每条成功响应包含 `ok:true`，错误为 `ok:false,error`；命令处理成功和 kernel 完成分别判断。

| 字段 | 含义 |
|---|---|
| reason | completed 完成；breakpoint/step/budget 暂停；源码前端另有 source_step/focus_exited |
| instructions | 会话累计 warp 指令数 |
| cycles | CycleSim 模拟周期数 |
| sm/warp/pc、line | 停止位置及 1 基源码行；PC 为 8 字节对齐的字节地址 |
| registers/predicates | 焦点 lane 的原始寄存器与谓词 |
| stack/active_mask | 真实 SIMT 栈和硬件活跃掩码 |
| cta | state 中为 [x,y,z]；warps 概览中为 x 坐标 |

print 的 active 同时考虑编译器 live mask 与硬件 mask。功能预算暂停可继续；周期后端错误通过 error 字段报告。

模拟器接入使用 `debug_before(sm,warp,pc)`：返回 true 时暂停；`debug_paused` 标记暂停，`debug_resume_sm/debug_resume_warp` 保存轮内游标。init 清空状态，完整轮结束后复位游标并处理 CTA 退休/补发。sync 在暂停时返回 ERR_OK，是否完成由 is_done() 判断。实现文件和改动记录见[CHANGELOG](../CHANGELOG.md)。

## 4. 输出与日志

| 工具 | 正常输出 | 错误/失败 |
|---|---|---|
| build.py | 编译器输出、Built 路径；--tests 附带模拟器测试结果 | 编译/测试子进程非零导致异常和非零退出 |
| ntcc.py | 模块 JSON 路径 | stderr `ntcc: ...`；源码问题带行号；退出 1 |
| ntasm.py | 指令数、字节数、目标文件 | 非法字段/标签报错，非零退出 |
| ntrun.py | execution/buffers JSON，或写 -o | 模块/启动/执行错误非零退出 |
| ntdb.py 交互 | 命令结果及提示符 | 显示 error，可继续输入 |
| ntdb.py 批处理 | 每行 `{command,result}` JSON | 错误中止并非零退出 |
| ntsim | 首行握手 JSON，每条命令一个响应 | 命令级 `{ok:false,error:...}`，进程可继续；启动错误 stderr/非零 |

stdout/stderr 可重定向保存；交互 NTDB 含提示符，批处理为 JSONL。Python traceback 是宿主错误信息，设备执行错误通过原生错误码和 PC 定位。

LeNet 的 `--out` 目录包含：

| 产物 | 内容 |
|---|---|
| compiled/ | 四个 kernel 的机器码、模块、文本汇编和 listing |
| 样本_阶段.bin | NCHW FP32 张量；argmax 为小端 int32 |
| results.json | backend/debug/samples、预测与标签、accuracy_on_fixture、compilation、layers、total_cycles |
| debug.json | sample/layer 下的 command/result 列表；普通运行为空列表 |

compilation 记录寄存器数、指令数、代码 SHA-256 和 assembly_roundtrip；layers 记录 sample/layer/shape/max_abs_error/host_layer_seconds/execution。stdout 依次输出 COMPILED、阶段 JSON 和预测总结。

host_layer_seconds 为包含会话、传输和执行的墙钟秒；cycles 为虚拟周期；accuracy_on_fixture 只对应所选的 1–3 张固定输入。完整进程计时口径见[第 07 册](07_程序总时间测量.md)。

## 5. 故障定位

| 信息/现象 | 处理 |
|---|---|
| requires .ntcu input | 将 NTCC 设备源保存为 .ntcu；真正 CUDA 源继续由 CUDA 工具处理 |
| type mismatch | 本语言无 int/float 隐式转换；采用正确类型的参数和字面量 |
| register budget exceeds | 拆分 kernel 或简化表达式；本版本无 spill |
| 找不到 ntsim | 先构建；检查默认目录或 NTSIM_EXECUTABLE |
| code hash does not match | 保证同一次编译的 .ntas 和 .ntm.json 配套，重新编译 |
| invalid launch/resources | 检查三维乘积、参数类型、寄存器与 CTA 资源 |
| breakpoint outside code | PC 必须 8 字节对齐且落在代码内 |
| no executable instruction | 该源码行仅为空白/注释/括号或无映射，选择可执行语句行 |
| variable unavailable | 尚未初始化、超出词法范围或本 lane 声明被关闭 |
| budget | 功能调试可 continue；循环应自行终止；ntrun 要求完成所以会报错 |
| cycle execution error | 检查周期预算和模拟器 error；该状态表示运行失败 |
| NVRTC_LIBRARY | 指定 NVRTC 动态库及其依赖，或只运行 func/cycle |
| 逐层误差检查失败 | 依据日志和张量核对权重、布局、输入与浮点设置 |

| 编译信息 | 含义 |
|---|---|
| unsupported simulator capability / modifier / mode | 指令或模式未开放，范围见第 09 册 |
| IDIV/IREM support signed int only | uint 除余不受支持 |
| break/continue outside a loop | 语句位于 for/while 之外 |
| asm requires top-level control without return | asm 仅用于顶层且 kernel 中无逻辑 return |
| asm register group exceeds its constraint or writes an input | 约束宽度或输出方向与指令不符 |
| floating casts unavailable | 浮点转换未开放 |
