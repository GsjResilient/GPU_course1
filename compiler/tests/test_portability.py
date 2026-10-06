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
"""Compatibility policy and user-selected toolchains must agree end to end."""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from compiler.lib.artifact import load_artifact
from compiler.lib.build import build_tools, REPO
from compiler.lib.diagnostics import CompileError
from compiler.lib.driver import compile_file
from compiler.tests.support import source_file

try:
    from compiler.lib import toolchain
except ImportError:
    toolchain = None


class PortabilityTests(unittest.TestCase):
    def test_frontend_policy_accepts_testable_clang_families(self):
        self.assertIsNotNone(toolchain, 'shared toolchain policy is missing')
        for version in (
            'Apple clang version 21.0.0 (clang-2100.1.1.101)',
            'clang version 18.1.8',
            'Homebrew clang version 18.1.8',
            'Ubuntu clang version 18.1.3 (1ubuntu1)',
            'Debian clang version 18.1.8 (3+b1)',
        ):
            with self.subTest(version=version):
                self.assertTrue(toolchain.supported_frontend(version))
        for version in ('', None, 18, 'g++ 18.1.8', 'clang version 19.1.0',
                        'clang version 18.10.0', 'Apple clang version 18.1.8',
                        'Apple clang version 21.0.01', 'clang version 18.1.8evil'):
            with self.subTest(version=version):
                self.assertFalse(toolchain.supported_frontend(version))

    def test_nonapple_artifact_metadata_is_consumable(self):
        with source_file('NT_KERNEL void k(int* out){out[0]=42;}') as source:
            manifest = compile_file(source, 'k', source.parent / 'kernel')
            meta = json.loads(manifest.read_text())
            meta['clang_version'] = 'Ubuntu clang version 18.1.3 (1ubuntu1)'
            manifest.write_text(json.dumps(meta))
            try:
                loaded, code = load_artifact(manifest)
            except CompileError as error:
                self.fail('LLVM Clang metadata was rejected: ' + str(error))
            self.assertEqual(loaded['kernel'], 'k')
            self.assertGreater(len(code), 0)
            meta['clang_version'] = 'clang version 99.0.0'
            manifest.write_text(json.dumps(meta))
            with self.assertRaisesRegex(CompileError, 'E_ARTIFACT'):
                load_artifact(manifest)

    def test_environment_frontend_is_used_by_api_and_cli(self):
        with source_file('NT_KERNEL void k(int* out){out[0]=42;}') as source:
            missing = str(source.parent / 'missing-clang')
            with patch.dict(os.environ, {'NUTCC_CLANG': missing}):
                with self.assertRaisesRegex(CompileError, 'E_FRONTEND'):
                    compile_file(source, 'k', source.parent / 'api')
                result = subprocess.run(
                    [sys.executable, str(REPO / 'compiler/nutcc.py'), str(source),
                     '--kernel', 'k', '--out', str(source.parent / 'cli')],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('E_FRONTEND', result.stderr)
                self.assertFalse((source.parent / 'cli.json').exists())

    def test_explicit_frontend_overrides_environment(self):
        selected = os.environ.get('NUTCC_CLANG', 'clang++')
        with source_file('NT_KERNEL void k(int* out){out[0]=42;}') as source:
            with patch.dict(os.environ, {'NUTCC_CLANG': str(source.parent / 'missing-clang')}):
                manifest = compile_file(source, 'k', source.parent / 'kernel', clang=selected)
                self.assertEqual(load_artifact(manifest)[0]['kernel'], 'k')

    def test_native_builder_honors_environment_compiler(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {'NUTCC_CXX': str(root / 'missing-cxx')}):
                with self.assertRaises(OSError):
                    build_tools(root / 'tools')

    def test_native_builder_does_not_ignore_selected_sysroot(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {'NUTCC_SYSROOT': str(root / 'missing-sdk')}):
                with self.assertRaises(RuntimeError):
                    build_tools(root / 'tools')

    def test_relative_compiler_path_survives_native_build_directory(self):
        selected = shutil.which(toolchain.frontend_compiler())
        selected_native = shutil.which(toolchain.native_compiler())
        self.assertIsNotNone(selected)
        self.assertIsNotNone(selected_native)
        with TemporaryDirectory(prefix='nutcc tool path ') as directory:
            root = Path(directory)
            executable = root / 'clang++'
            executable.symlink_to(Path(selected).resolve())
            relative = os.path.relpath(executable)
            native = root / 'native' / 'clang++'
            native.parent.mkdir()
            native.symlink_to(Path(selected_native).resolve())
            with patch.dict(os.environ, {'NUTCC_CLANG': relative, 'NUTCC_CXX': os.path.relpath(native)}):
                source = root / 'kernel.cpp'
                source.write_text('NT_KERNEL void k(int* out){out[0]=42;}')
                manifest = compile_file(source, 'k', root / 'kernel')
                self.assertEqual(load_artifact(manifest)[0]['kernel'], 'k')
                tools = build_tools(root / 'tools')
                self.assertTrue(all(p.is_file() for p in tools.values()))

    def test_relative_sysroot_is_resolved_before_changing_build_directory(self):
        with TemporaryDirectory(prefix='nutcc sdk path ') as directory:
            root = Path(directory)
            relative = os.path.relpath(root)
            with patch.dict(os.environ, {'NUTCC_SYSROOT': relative}):
                self.assertIn('--sysroot=' + str(root.resolve()), toolchain.native_command())

    def test_frontend_selection_also_selects_native_compiler(self):
        self.assertIsNotNone(toolchain, 'shared toolchain policy is missing')
        with patch.dict(os.environ, {'NUTCC_CLANG': '/chosen/clang++-18'}, clear=True):
            self.assertEqual(toolchain.native_compiler(), '/chosen/clang++-18')
            self.assertEqual(toolchain.frontend_compiler(), '/chosen/clang++-18')
            with patch.dict(os.environ, {'NUTCC_CXX': '/other/clang++'}):
                self.assertEqual(toolchain.native_compiler(), '/other/clang++')
                self.assertEqual(toolchain.native_compiler('/explicit/cxx'), '/explicit/cxx')
