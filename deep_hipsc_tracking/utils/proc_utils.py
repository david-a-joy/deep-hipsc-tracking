""" Utilities for managing processes and printing statuses

Functions:

* :py:func:`print_qa_summary` - Print out a stats summary with colors for statuses
* :py:func:`print_cmd` - Pretty print a shell command
* :py:func:`call` - Call a shell command with nice debug output
* :py:func:`which` - Work out the full path to a shell command
* :py:func:`pair_inputs` - Pair off sets of FASTQ files

API Documentation
-----------------
"""

# Imports
import os
import time
import pathlib
import subprocess
from typing import List

# 3rd party
from termcolor import colored

# Functions


def print_qa_summary(stats: List):
    """ Pretty print the QA summary

    Example:

        >>> stats = [('good', 'happy message'), ('bad', 'sad message')]
        >>> print_qa_summary(stats)
        [GOOD] happy message
        [BAD ] sad message

    :param list[str, str] stats:
        The list of status: message pairs for each run
    """

    # Summarize the final runs
    print('\nFinal QA:')
    overall = colored('[GOOD]', 'green')
    for stat, message in stats:
        stat = stat.lower()
        if stat == 'skip':
            stat = colored('[SKIP]', 'yellow')
        elif stat == 'good':
            stat = colored('[GOOD]', 'green')
        elif stat == 'bad':
            stat = colored('[BAD ]', 'red')
            overall = colored('[BAD ]', 'red')
        else:
            stat = '[{}]'.format(stat.upper())
        print('{} - {}'.format(stat, message))
    print('Overall run seems {}'.format(overall))


def startswith_shell(cmd: List[str]) -> bool:
    """ Detect if a command starts with a shell name

    :param list[str] cmd:
        The command to check
    :returns:
        True if the first element seems like a shell, False otherwise
    """
    shells = ['python', 'bash']
    if len(cmd) > 1:
        if any([pathlib.Path(cmd[0]).name.startswith(s) for s in shells]):
            return True
    return False


def print_cmd(cmd: List):
    """ Print the output command

    :param list[object] cmd:
        The command to run
    """
    cmd = [str(c) for c in cmd]

    # Collect all the switches in a sensible way
    was_switch = False
    had_interp = False

    lines = []
    cur_line = []

    for i, c in enumerate(cmd):
        if i == 0 and startswith_shell(cmd):
            had_interp = True
            cur_line.append(c)
            continue
        if had_interp:
            had_interp = False
            cur_line.append(c)
            lines.append(cur_line)
            cur_line = []
            continue

        if c.startswith(('-', '--')):
            if cur_line != []:
                lines.append(cur_line)
            cur_line = [c]
            was_switch = True
        else:
            if was_switch:
                cur_line.append(c)
            else:
                if cur_line != []:
                    lines.append(cur_line)
                cur_line = [c]
            was_switch = False

    if cur_line != []:
        lines.append(cur_line)

    # Now pretty print the command with tabs
    print('Final command: \n')
    for i, line in enumerate(lines):
        line = ' '.join(line)
        if i == 0:
            print(line + ' \\')
        elif i < len(lines) - 1:
            print('\t' + line + ' \\')
        else:
            print('\t' + line)
    print('')


def which(cmd: str, extra_paths=None) -> pathlib.Path:
    """ Locate a command in the $PATH

    :param str cmd:
        The name of the command to locate
    :param list/str/None extra_paths:
        The path or paths to search in addition to os.environ['PATH']
    :returns:
        The pathlib.Path to the command
    """
    if extra_paths is None:
        extra_paths = []
    elif isinstance(extra_paths, str):
        extra_paths = [extra_paths]

    if cmd.endswith('.exe'):
        cmd = cmd.rsplit('.', 1)[0]
    envpath = os.environ.get('PATH', '').split(os.pathsep)
    for p in extra_paths + envpath:
        pcmd = pathlib.Path(p) / cmd
        if pcmd.is_file():
            return pcmd
        pcmd = pathlib.Path(p) / (cmd + '.exe')
        if pcmd.is_file():
            return pcmd
    raise OSError(f'{cmd} not found in PATH')


def call(*args, **kwargs):
    """ Call a function

    :param bool dry_run:
        If True, don't actually call the function, just appear to call it
    :param \\*args:
        The command line arguments
    :param \\*\\*kwargs:
        Options to subprocess.check_call
    """

    # Remove some special keyword only arguments...
    dry_run = kwargs.pop('dry_run', False)
    quiet = kwargs.pop('quiet', False)

    # Handle being passed a single argument
    if len(args) == 1:
        if isinstance(args[0], str):
            args = [args[0]]
        else:
            args = list(args[0])

    cmd = [str(a) for a in args]
    if startswith_shell(cmd):
        cmd_name = cmd[1]
    else:
        cmd_name = cmd[0]
    if not quiet:
        print('#' * 5 + cmd_name + '#' * 5)
        print('')

        print_cmd(cmd)

        if kwargs != {}:
            print('Options:')
            for key, val in sorted(kwargs.items()):
                print(f'* {key}: {val}')
            print('')

    if 'cwd' in kwargs:
        kwargs['cwd'] = str(kwargs['cwd'])

    t0 = time.time()

    if not dry_run:
        subprocess.check_call(cmd, **kwargs)

    dt = time.time() - t0

    if not quiet:
        print('')
        print(f'Finished in {dt:0.4f} secs')
        print('')
        print('#' * 5 + cmd_name + '#' * 5)
