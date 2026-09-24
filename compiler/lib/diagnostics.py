from dataclasses import dataclass


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int
    column: int


class CompileError(Exception):
    def __init__(self, code, message, loc):
        self.code, self.message, self.loc = code, message, loc
        super().__init__(f'{loc.file}:{loc.line}:{loc.column}: {code}: {message}')
