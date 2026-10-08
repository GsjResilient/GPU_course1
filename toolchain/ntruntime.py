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
        self.modules = {0: self.meta}
        self.module_paths = {str(module): 0}
        self.selected_module = 0
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
        self.write(address, data)
        return address

    def write(self, address, data):
        """Update an existing allocation without changing its address."""
        return self.command(f'write {address} {bytes(data).hex() or "-"}')

    def load_module(self, module):
        """Load and verify another NTMODULE-1; retain all device allocations."""
        module = Path(module).resolve()
        if str(module) in self.module_paths:
            return self.module_paths[str(module)]
        meta = json.loads(module.read_text(encoding='utf-8'))
        if meta['format'] != 'NTMODULE-1' or meta['isa'] != 'NTAS1':
            raise ValueError('unsupported module format/ISA')
        code = module.parent / meta['code_file']
        if hashlib.sha256(code.read_bytes()).hexdigest() != meta['code_sha256']:
            raise ValueError('code hash does not match debug/module metadata')
        # Hex-encode paths so whitespace and Unicode survive the text protocol.
        identifier = self.command(f'modulehex {str(code).encode("utf-8").hex()} {meta["registers"]} {meta["parameter_size"]}')['module']
        self.modules[identifier] = meta
        self.module_paths[str(module)] = identifier
        return identifier

    def select_module(self, identifier):
        if identifier not in self.modules:
            raise ValueError('unknown module')
        self.command(f'select {identifier}')
        self.selected_module = identifier
        self.meta = self.modules[identifier]

    def download(self, address, size):
        return bytes.fromhex(self.command(f'read {address} {size}')['hex'])

    def launch(self, grid, block, arguments, dynamic_smem=0, module=None):
        if module is not None:
            self.select_module(module)
        if type(dynamic_smem) is not int or dynamic_smem < 0:
            raise ValueError('dynamic_smem must be a nonnegative integer')
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
        if dynamic_smem:
            return self.command('launch4 '+' '.join(map(str,dims))+' '+str(dynamic_smem)+' '+(params.hex() or '-'))
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
# Version 1.3
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
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
