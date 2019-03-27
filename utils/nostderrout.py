import contextlib
import sys
import os


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostderrout():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = DummyFile()
    sys.stderr = DummyFile()
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr
