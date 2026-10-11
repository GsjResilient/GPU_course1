"""Shared frontend compatibility policy and explicit tool selection.

Clang's JSON AST is versioned by its producer, not a stable public ABI.
Only the adapted release families below are accepted; platform validation
and exact tested versions are recorded separately in the release report.
"""

# =============================================================================
# NutShellGPU: GPU Hardware Emulator for Education & GPU Simulator for Education
# Version 1.4
# Author: Di Zhao (zhaodi@ucas.ac.cn), Trae CN
# Improved by: Tonghui Ming (26F)，Yikai Wang (26F)，Wenjun Cui (26F)，Guiyi He (26F)
# References: Tor M. Aamodt, Wilson W. Lun Fung, Timothy G. Rogers, General-Purpose Graphics Processor Architecture, Morgan & Claypool, 2018
# September 2026
# =============================================================================
import os
import re
from pathlib import Path


SUPPORTED_FRONTENDS = 'LLVM Clang 18.1.x or Apple Clang 21.0.0'


def supported_frontend(version):
    if not isinstance(version, str):
        return False
    return bool(re.match(
        r'^(?:Apple clang version 21\.0\.0|'
        r'(?:(?:Ubuntu|Debian|Homebrew) )?clang version 18\.1\.[0-9]+)(?:\s|$)',
        version))


def _executable(value):
    value = os.fspath(value)
    # Preserve the executable basename: resolving clang++'s symlink to clang
    # can change the driver's C++ link behavior. Only anchor relative paths.
    return os.path.abspath(value) if os.path.dirname(value) else value


def frontend_compiler(explicit=None):
    return _executable(explicit or os.environ.get('NUTCC_CLANG') or 'clang++')


def native_compiler(explicit=None):
    return _executable(explicit or os.environ.get('NUTCC_CXX') or frontend_compiler())


def native_command(explicit=None):
    command = [native_compiler(explicit)]
    # Pass a real driver option: older Clang installations may honor SDKROOT
    # for headers but retain a baked-in default sysroot for the linker.
    sysroot = os.environ.get('NUTCC_SYSROOT')
    if sysroot:
        command.append('--sysroot=' + str(Path(sysroot).resolve()))
    return command
