import contextlib
import sys

import tqdm


class DummyTqdmFile:
    def __init__(self, _file):
        self._file = _file

    def write(self, c):
        if len(c.rstrip()) > 0:
            tqdm.tqdm.write(c, file=self._file)


@contextlib.contextmanager
def stdout_redirect_to_tqdm():
    stdout_bak = sys.stdout
    try:
        sys.stdout = DummyTqdmFile(sys.stdout)
        yield stdout_bak
    except Exception as e:
        raise e
    finally:
        sys.stdout = stdout_bak
