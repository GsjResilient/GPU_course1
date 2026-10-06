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
"""Python host program: compile, upload, launch, synchronize, download."""
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'toolchain'))
from ntcc import compile_file
from ntruntime import Session

module=compile_file(ROOT/'examples/toolchain/saxpy.ntcu',ROOT/'out/saxpy')
with Session(module) as device:
    x=device.upload(struct.pack('<5f',1,2,3,4,5))
    y=device.upload(struct.pack('<5f',10,20,30,40,50))
    device.launch(grid=1,block=32,arguments={'x':x,'y':y,'a':2.5,'n':5})
    device.synchronize()
    result=struct.unpack('<5f',device.download(y,20))
    assert result==(12.5,25.0,37.5,50.0,62.5)
    print(result)
