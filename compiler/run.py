#!/usr/bin/env python3
"""Run a compiled NutCC artifact using a data-only launch JSON file."""
import argparse,json,re,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compiler.lib.runtime import run_artifact,BufferRef
from compiler.lib.artifact import unique_object


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',required=True,type=Path)
    parser.add_argument('--backend',required=True,choices=('func','cycle'))
    parser.add_argument('--launch',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args(argv)
    try:
        doc=json.loads(args.launch.read_text(),object_pairs_hook=unique_object)
        if set(doc)!={'format','grid','block','buffers','args'} or doc['format']!='nutcc-launch-v1':raise ValueError('invalid launch schema')
        buffers={}
        for name,file in doc['buffers'].items():
            if not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*',name):raise ValueError('invalid buffer name')
            buffers[name]=(args.launch.parent/file).read_bytes()
        values=[]
        for v in doc['args']:
            if isinstance(v,dict):
                if set(v)!={'buffer','offset'}:raise ValueError('invalid buffer argument')
                v=BufferRef(v['buffer'],v['offset'])
            values.append(v)
        result=run_artifact(args.artifact,args.backend,buffers,values,doc['grid'],doc['block'],args.out)
        for name,data in result.buffers.items():(args.out/(name+'.bin')).write_bytes(data)
        print(f'PASS backend={args.backend} completed=true instructions={result.instructions} cycles={result.cycles}')
        return 0
    except Exception as e:
        print(str(e),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
