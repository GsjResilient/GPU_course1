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
"""Run the same basic .ntcu kernel via an explicit CUDA adapter on real CUDA through NVRTC."""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'lenet'))
from run import NativeCUDA
c=NativeCUDA()
try:
    values={'a':np.arange(1,6,dtype='f4'),'b':np.arange(10,60,10,dtype='f4'),'out':np.zeros(5,'f4'),'n':5}
    result,event=c.execute('vector_add',values,(5,),(2,),(4,))
    np.testing.assert_array_equal(result,[11,22,33,44,55])
    print(json.dumps({'output':result.tolist(),**event}))
finally:c.close()
