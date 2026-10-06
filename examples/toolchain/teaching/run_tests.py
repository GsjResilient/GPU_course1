"""教学测试目标：验证 SAXPY、矩阵乘法和 16 桶原子直方图。

编译三个 .ntcu，在功能与周期后端各运行 14 组输入，与独立 NumPy
参考比较，同时检查保护区、输入完整性和两种后端的输出一致性。
"""

# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

# 从样例位置定位工具链，支持从任意工作目录启动。
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / 'toolchain'))
import numpy as np
from ntcc import compile_file
from ntasm import assemble
from ntruntime import Session, prepare, collect


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def cases():
    # 固定随机种子，生成边界输入与普通输入，并独立计算参考结果。
    rng = np.random.default_rng(20260923)
    # SAXPY：覆盖空向量、单元素及不足整块的尾部；参考使用 FP64。
    for n in (0, 1, 33, 257, 4099):
        x = rng.uniform(-8, 8, max(n, 1)).astype(np.float32)
        y = rng.uniform(-8, 8, max(n, 1)).astype(np.float32)
        ref = 1.75*x[:n].astype(np.float64)+y[:n].astype(np.float64)
        yield dict(kernel='saxpy', name=f'n{n}', size=f'n={n}',
                   grid=max(1, (n+63)//64), block=64,
                   arguments={'x': x.tolist(), 'y': y.tolist(),
                              'out': [-123456.75]*(n+8), 'a': 1.75, 'n': n},
                   output='out', reference=ref, count=n, exact=False,
                   primary=n == 4099, reference_kind='NumPy FP64 a*x+y')
    # 矩阵乘法：覆盖矩形矩阵、二维边缘线程及 k=0 的空点积。
    for m, k, n in ((1, 1, 1), (3, 5, 7), (17, 19, 13), (4, 0, 5)):
        a = rng.uniform(-2, 2, (m, k)).astype(np.float32)
        b = rng.uniform(-2, 2, (k, n)).astype(np.float32)
        ref = (a.astype(np.float64) @ b.astype(np.float64)).ravel()
        yield dict(kernel='matmul', name=f'm{m}_k{k}_n{n}',
                   size=f'{m}x{k} @ {k}x{n}', grid=[(n+7)//8, (m+3)//4], block=[8, 4],
                   arguments={'a': a.ravel().tolist() or [0.0],
                              'b': b.ravel().tolist() or [0.0],
                              'c': [-123456.75]*(m*n+8), 'm': m, 'n': n, 'k': k},
                   output='c', reference=ref, count=m*n, exact=False,
                   primary=m == 17, reference_kind='NumPy FP64 matrix multiplication')
    # 直方图：随机分桶和全部命中同一桶，验证原子更新的完整性。
    for name, n in (('empty', 0), ('single', 1), ('tail', 257),
                    ('all_one_bin', 4099), ('random', 4099)):
        values = (np.full(max(n, 1), 7, dtype=np.uint32) if name == 'all_one_bin'
                  else rng.integers(0, 65536, max(n, 1), dtype=np.uint32))
        ref = np.bincount((values[:n] & 15).astype(np.int64), minlength=16)
        yield dict(kernel='histogram', name=name, size=f'n={n}, bins=16',
                   grid=max(1, (n+63)//64), block=64,
                   arguments={'input': values.tolist(), 'bins': [0]*16+[0xdecafbad]*8, 'n': n},
                   output='bins', reference=ref, count=16, exact=True,
                   primary=name == 'random', reference_kind='NumPy bincount(input & 15)')


def run_case(case, module, executable, backend, output_dir):
    # 保存可复现的启动参数，再上传数据、启动 kernel 并取回全部缓冲区。
    config = {key: case[key] for key in ('grid', 'block', 'arguments')}
    prefix = f"{case['kernel']}_{case['name']}"
    save_json(output_dir/'inputs'/(prefix+'.launch.json'), config)
    start = time.perf_counter()
    with Session(module, backend, executable) as device:
        buffers = prepare(device, config)
        # 直方图使用低 32 位地址加法，检查地址与分配区间末端均在该范围。
        assert all(address + count*4 < 2**32 for address, count, _ in buffers.values())
        event = device.synchronize(100000000)
        returned = collect(device, buffers)
    elapsed = time.perf_counter()-start
    name, count = case['output'], case['count']
    actual = np.array(returned[name][:count], dtype=np.uint32 if case['exact'] else np.float32)
    reference = case['reference']
    error = np.abs(actual.astype(np.float64)-reference.astype(np.float64))
    # 输出尾部的 8 个保护值用于发现越界写；输入缓冲区应保持不变。
    guards_ok = returned[name][count:] == case['arguments'][name][count:]
    inputs_ok = all(returned[key] == case['arguments'][key]
                    for key in buffers if key != name)
    # 整数结果要求完全一致；浮点结果按绝对与相对容差比较。
    if case['exact']:
        correct = np.array_equal(actual, reference)
        assert int(actual.sum()) == case['arguments']['n']
    else:
        correct = np.allclose(actual, reference, atol=1e-5, rtol=1e-5)
        # SAXPY 还应与高精度参考一次舍入到 FP32 的结果逐字节一致。
        if case['kernel'] == 'saxpy':
            assert actual.tobytes() == reference.astype(np.float32).tobytes()
    assert event['reason'] == 'completed' and correct and guards_ok and inputs_ok, (prefix, backend)
    np.savez(output_dir/'outputs'/f'{prefix}.{backend}.npz',
             actual=actual, reference=reference, full_output=np.array(returned[name]))
    record = dict(kernel=case['kernel'], case=case['name'], size=case['size'], backend=backend,
                  grid=case['grid'], block=case['block'], primary=case['primary'],
                  reason=event['reason'], passed=True, max_abs_error=float(error.max()) if count else 0.0,
                  reference=case['reference_kind'], guards_intact=guards_ok, inputs_unchanged=inputs_ok,
                  instructions=event.get('instructions'), cycles=event.get('cycles'),
                  host_seconds=elapsed, output_sha256=sha(actual.tobytes()),
                  preview=actual[:16].tolist())
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=PROJECT/'out/teaching',
                        help='生成的模块、输入、结果与数组目录')
    parser.add_argument('--executable', type=Path,
                        help='ntsim 路径；省略时使用运行库默认路径或 NTSIM_EXECUTABLE')
    args = parser.parse_args()
    output_dir = args.out.resolve()
    executable = args.executable.resolve() if args.executable else None
    for folder in ('compiled', 'inputs', 'outputs'):
        (output_dir/folder).mkdir(parents=True, exist_ok=True)

    # 编译并重汇编，确认文本汇编与原始机器码完全一致。
    compilation, modules = {}, {}
    for kernel in ('saxpy', 'matmul', 'histogram'):
        source = HERE/(kernel+'.ntcu')
        modules[kernel] = compile_file(source, output_dir/'compiled'/kernel)
        meta = json.loads(modules[kernel].read_text(encoding='utf-8'))
        code = (output_dir/'compiled'/(kernel+'.ntas')).read_bytes()
        assert assemble((output_dir/'compiled'/(kernel+'.ntasm')).read_text(encoding='utf-8')) == code
        compilation[kernel] = dict(registers=meta['registers'], static_instructions=len(code)//8,
                                   source_sha256=sha(source.read_bytes()), code_sha256=sha(code),
                                   assembly_roundtrip=True)
    results = dict(seed=20260923, compilation=compilation, runs=[], passed=False,
                   backend_outputs_identical=False)
    try:
        # 同一组输入分别运行两个后端，比较输出字节哈希。
        for case in cases():
            records = [run_case(case, modules[case['kernel']], executable, backend, output_dir)
                       for backend in ('func', 'cycle')]
            results['runs'].extend(records)
            assert records[0]['output_sha256'] == records[1]['output_sha256']
        results['backend_outputs_identical'] = True
        results['passed'] = True
    finally:
        # 出错时也保存已完成用例的记录；异常向上传播并返回非零退出码。
        save_json(output_dir/'results.json', results)
    print('ALL_PASS', len(results['runs']), flush=True)


if __name__ == '__main__':
    main()
