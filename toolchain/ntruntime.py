"""Host runtime 1.2.0: allocate, copy, launch and synchronize NTCC kernels."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parent


class Session:
    def __init__(self, module, backend='func', executable=None):
        module = Path(module).resolve()
        self.meta = json.loads(module.read_text(encoding='utf-8'))
        if self.meta['format'] != 'NTMODULE-1' or self.meta['isa'] != 'NTAS1':
            raise ValueError('unsupported module format/ISA')
        code = module.parent / self.meta['code_file']
        if hashlib.sha256(code.read_bytes()).hexdigest() != self.meta['code_sha256']:
            raise ValueError('code hash does not match debug/module metadata')
        override = executable or os.environ.get('NTSIM_EXECUTABLE')
        exe = Path(override) if override else ROOT.parent/'build/toolchain'/('ntsim.exe' if os.name == 'nt' else 'ntsim')
        self.process = subprocess.Popen([str(exe), str(code), str(self.meta['registers']), str(self.meta['parameter_size']), backend],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        try:
            self._receive()
        except Exception:
            self.close(); raise

    def _receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('native transport exited: '+self.process.stderr.read())
        result = json.loads(line)
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'transport error'))
        return result

    def command(self, text):
        self.process.stdin.write(text+'\n'); self.process.stdin.flush()
        return self._receive()

    def malloc(self, size):
        return self.command(f'alloc {size}')['address']

    def upload(self, data):
        address = self.malloc(len(data))
        self.command(f'write {address} {data.hex()}')
        return address

    def download(self, address, size):
        return bytes.fromhex(self.command(f'read {address} {size}')['hex'])

    def launch(self, grid, block, arguments):
        expected = {p['name'] for p in self.meta['parameters']}
        if set(arguments) != expected:
            raise ValueError(f'kernel arguments must be {sorted(expected)}')
        params = bytearray(self.meta['parameter_size'])
        for p in self.meta['parameters']:
            t = p['type']; fmt = 'Q' if t.endswith('*') else {'int':'i', 'uint':'I', 'float':'f'}[t]
            struct.pack_into('<'+fmt, params, p['offset'], arguments[p['name']])
        def dimensions(value):
            dims = [value] if isinstance(value, int) else list(value)
            if not 1 <= len(dims) <= 3 or any(type(v) is not int or v < 1 for v in dims):
                raise ValueError('dimensions must contain one to three positive integers')
            return dims + [1] * (3-len(dims))
        dims = dimensions(grid) + dimensions(block)
        return self.command('launch3 '+' '.join(map(str,dims))+' '+(params.hex() or '-'))

    def synchronize(self, budget=10000000):
        event = self.command(f'run {budget}')
        if event['reason'] != 'completed':
            raise RuntimeError('execution stopped: '+event['reason'])
        return event

    def close(self):
        if not hasattr(self, 'process'):
            return
        if self.process.poll() is None:
            try:
                self.process.stdin.write('quit\n'); self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill(); self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def prepare(session, config):
    """Prepare named arrays/scalars from a JSON launch description."""

# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Author: Di Zhao, Trae CN
# Contact: zhaodi@ucas.ac.cn
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers,
#             General-Purpose Graphics Processor Architecture,
#             Morgan & Claypool, 2018
# September 2026
# Improved by: 明同辉（26秋），王一锴（26秋），崔文君（26秋）
# =============================================================================
    arguments, buffers = {}, {}
    supplied = config['arguments']
    if set(supplied) != {p['name'] for p in session.meta['parameters']}:
        raise ValueError('launch configuration arguments do not match kernel')
    for p in session.meta['parameters']:
        value = supplied[p['name']]
        if p['type'].endswith('*'):
            fmt = {'int*':'i', 'uint*':'I', 'float*':'f'}[p['type']]
            data = struct.pack('<'+fmt*len(value), *value)
            address = session.upload(data)
            arguments[p['name']] = address
            buffers[p['name']] = (address, len(value), fmt)
        else:
            arguments[p['name']] = value
    session.launch(config['grid'], config['block'], arguments)
    return buffers


def collect(session, buffers):
    return {name: list(struct.unpack('<'+fmt*n, session.download(address, n*4))) for name, (address,n,fmt) in buffers.items()}
