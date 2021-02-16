
# Imports

import sys
import unittest
import tempfile
import pathlib
import contextlib

# Constants

THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR / 'data'


# Helper classes


class FileSystemTestCase(unittest.TestCase):

    def setUp(self):
        self._tempdir_obj = tempfile.TemporaryDirectory()
        self.tempdir = pathlib.Path(self._tempdir_obj.__enter__()).resolve()

    def tearDown(self):
        self.tempdir = None
        self._tempdir_obj.__exit__(None, None, None)
        self._tempdir_obj = None


class StreamRecord(object):

    def __init__(self):
        self.lines = []

    def write(self, line):
        self.lines.append(line)


# Helper functions


@contextlib.contextmanager
def record_stdout():
    """ Record the standard output so we can poke it later """

    old_stdout = sys.stdout

    try:
        sys.stdout = StreamRecord()
        yield sys.stdout
    finally:
        sys.stdout = old_stdout
