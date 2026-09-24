from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def source_file(text):
    with TemporaryDirectory(prefix='nutcc 测试 ') as directory:
        path = Path(directory) / 'kernel.cpp'
        path.write_text(text, encoding='utf-8')
        yield path
