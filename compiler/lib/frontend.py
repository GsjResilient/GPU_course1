# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
"""Versioned Clang JSON AST adapter; never interprets source text as C++."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import shlex
import subprocess
from .diagnostics import CompileError, SourceLoc
from .toolchain import frontend_compiler, supported_frontend, SUPPORTED_FRONTENDS

INCLUDE = Path(__file__).resolve().parents[1] / 'include'
SDK = INCLUDE / 'nutcc.h'


@dataclass
class ParsedUnit:
    source: Path
    source_bytes: bytes
    ast: dict
    clang_version: str
    dependencies: dict

    def location(self, node):
        p = node.get('loc', node.get('range', {}).get('begin', {}))
        p = p.get('expansionLoc', p.get('spellingLoc', p))
        filename = p.get('file', node.get('_file', str(self.source)))
        try:
            data = self.source_bytes if filename == str(self.source) else Path(filename).read_bytes()
        except OSError:
            data = b''
        offset = p.get('offset', 0)
        line = p.get('line', data[:offset].count(b'\n') + 1)
        column = p.get('col', offset - data.rfind(b'\n', 0, offset))
        return SourceLoc(filename, line, column)

    def error(self, node, code, message):
        raise CompileError(code, message, self.location(node))


def annotate_files(ast, default):
    # Clang elides repeated file names in source locations throughout its dump.
    current = default
    def visit(n, inherited):
        nonlocal current
        loc = n.get('loc', n.get('range', {}).get('begin', {}))
        loc = loc.get('expansionLoc', loc.get('spellingLoc', loc))
        if 'file' in loc:
            current = loc['file']
        n['_file'] = current if loc else inherited
        for child in n.get('inner', []):
            visit(child, n['_file'])
    visit(ast, default)


def parse_source(source, clang=None):
    clang = frontend_compiler(clang)
    source = Path(source).resolve()
    loc = SourceLoc(str(source), 1, 1)
    try:
        data = source.read_bytes()
        version = subprocess.run([clang, '--version'], capture_output=True, text=True,
                                 timeout=10, check=True).stdout.splitlines()[0]
        if not supported_frontend(version):
            raise CompileError('E_CLANG_VERSION', 'supported frontend: ' + SUPPORTED_FRONTENDS + '; got ' + version, loc)
        with TemporaryDirectory(prefix='nutcc-clang-') as td:
            depfile = Path(td) / 'source.d'
            cmd = [clang, '-std=c++17', '-fsyntax-only', '-Xclang', '-ast-dump=json',
                   '-nostdinc', '-nostdinc++', '-I', str(INCLUDE), '-include', str(SDK),
                   '-MD', '-MF', str(depfile), str(source)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise CompileError('E_CLANG', result.stderr.strip(), loc)
            ast = json.loads(result.stdout)
            if ast.get('kind') != 'TranslationUnitDecl':
                raise CompileError('E_AST', 'missing translation unit', loc)
            deps = shlex.split(depfile.read_text().replace('\\\n', ' ').split(':', 1)[1])
            hashes = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in deps}
        if source.read_bytes() != data:
            raise CompileError('E_SOURCE_CHANGED', 'source changed during compilation', loc)
        annotate_files(ast, str(source))
        return ParsedUnit(source, data, ast, version, hashes)
    except CompileError:
        raise
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        raise CompileError('E_FRONTEND', str(e), loc) from e
