""" Tests for the subprocess utilities """

# Imports
import unittest
import pathlib

# Our own imports
from ..helpers import FileSystemTestCase, record_stdout

from deep_hipsc_tracking.utils import proc_utils

# Tests


class TestCall(FileSystemTestCase):

    def test_calls_a_simple_function(self):

        cmd = ['touch', self.tempdir / 'hi.txt']

        proc_utils.call(cmd)

        self.assertTrue((self.tempdir / 'hi.txt').is_file())

    def test_runs_commands_silently_if_asked(self):

        cmd = ['echo', 12]
        with record_stdout() as rec:
            proc_utils.call(cmd, quiet=True)

        self.assertEqual(''.join(rec.lines), '')

    def test_calls_a_simple_function_dry_run(self):

        cmd = ['touch', self.tempdir / 'hi.txt']

        proc_utils.call(cmd, dry_run=True)

        self.assertFalse((self.tempdir / 'hi.txt').is_file())


class TestPrintCmd(unittest.TestCase):

    def test_prints_simple_command(self):

        cmd = ['foo']
        with record_stdout() as rec:
            proc_utils.print_cmd(cmd)

        exp = 'Final command: \n\nfoo \\\n\n'

        self.assertEqual(''.join(rec.lines), exp)

    def test_prints_simple_command_with_non_str(self):

        cmd = [pathlib.Path('foo'), '--bees', 1]
        with record_stdout() as rec:
            proc_utils.print_cmd(cmd)

        exp = 'Final command: \n\nfoo \\\n\t--bees 1\n\n'

        self.assertEqual(''.join(rec.lines), exp)

    def test_prints_complex_command(self):

        cmd = ['foo', 'bar', 'baz']
        with record_stdout() as rec:
            proc_utils.print_cmd(cmd)

        exp = ['Final command: \n',
               '\n',
               'foo \\\n',
               '\tbar \\\n',
               '\tbaz\n\n']

        self.assertEqual(''.join(rec.lines), ''.join(exp))

    def test_prints_complex_command_with_options(self):

        cmd = ['foo', '--bar', 'baz', '-b', 'boff']
        with record_stdout() as rec:
            proc_utils.print_cmd(cmd)

        exp = ['Final command: \n',
               '\n',
               'foo \\\n',
               '\t--bar baz \\\n',
               '\t-b boff\n\n']

        self.assertEqual(''.join(rec.lines), ''.join(exp))

    def test_prints_python_command_with_options(self):

        cmd = ['python2', 'foo', '--bar', 'baz', '-b', 'boff']
        with record_stdout() as rec:
            proc_utils.print_cmd(cmd)

        exp = ['Final command: \n',
               '\n',
               'python2 foo \\\n',
               '\t--bar baz \\\n',
               '\t-b boff\n\n']

        self.assertEqual(''.join(rec.lines), ''.join(exp))

    def test_prints_bash_command_with_options(self):

        cmd = ['bash', 'foo', '--bar', 'baz', '-b', 'boff']
        with record_stdout() as rec:
            proc_utils.print_cmd(cmd)

        exp = ['Final command: \n',
               '\n',
               'bash foo \\\n',
               '\t--bar baz \\\n',
               '\t-b boff\n\n']

        self.assertEqual(''.join(rec.lines), ''.join(exp))
