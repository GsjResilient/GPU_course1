from dataclasses import dataclass
from typing import Optional, Tuple, List
from .diagnostics import SourceLoc


@dataclass(frozen=True)
class VReg:
    id: int
    kind: str


@dataclass
class IRInst:
    op: str
    dst: Optional[VReg]
    args: Tuple[VReg, ...]
    imm: Optional[int]
    guard: Optional[VReg]
    loc: SourceLoc


@dataclass
class Program:
    name: str
    params: list
    param_size: int
    instructions: List[IRInst]
