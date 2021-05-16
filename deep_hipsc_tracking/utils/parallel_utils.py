""" Multiprocessing utilities

Main class:

* :py:class:`Hypermap`: Parallel processing aware map class

Helper classes:

* :py:class:`MaybeProcessDict`: Wrapper for functions where args are passed as a dict

"""

# Standard lib
import multiprocessing
import traceback
from typing import Optional, Callable, Tuple, Iterable, Dict, Any

# Classes


class Hypermap(object):
    """ Hyper version of map

    Example::

        def add_one(x):
            return x + 1

        with Hypermap(processes=2, lazy=False) as pool:
            res = pool.map(add_one, [1, 2, 3, 4, 5])
        assert res == [2, 3, 4, 5, 6]

    When processes <= 1, normal python map() is used.
    When processes > 1, a multiprocessing.Pool is used.

    :param int processes:
        Number of processes to use (default: multiprocessing.cpu_count)
    :param bool lazy:
        If True, do lazy evaluation and return an iterator, otherwise
        return a list
    :param int maxtasksperchild:
        If not None, end single child processes after processing this many items
    :param func initializer:
        If not None, call this function once at the start of each child task
    :param tuple initargs:
        If not None, call the initializer with these arguments
    :param str wrapper:
        If not None, wrap functions in one of the maybe process functions:

        * "dict" - Each item is a dictionary, passed to the map argument as \\*\\*kwargs
    """

    def __init__(self,
                 processes: Optional[int] = None,
                 lazy: bool = False,
                 maxtasksperchild: Optional[int] = None,
                 initializer: Optional[Callable] = None,
                 initargs: Optional[Tuple] = None,
                 wrapper: Optional[str] = None):
        self.lazy = lazy
        self.maxtasksperchild = maxtasksperchild

        self.initializer = initializer
        self.initargs = initargs if initargs else tuple()

        self.wrapper = wrapper

        self._processes = processes
        self._pool = None

    @classmethod
    def cpu_count(self) -> int:
        """ Count the number of available CPU cores

        :returns:
            Expected number of CPU cores on this machine
        """
        return multiprocessing.cpu_count()

    @property
    def processes(self) -> int:
        """ How many processes to use for this multiprocessing pool """
        # Cache the process number we want to use
        if self._processes is None:
            self._processes = self.cpu_count()
        return self._processes

    def open(self):
        """ Start the multiprocessing task pool """
        processes = self.processes
        if processes > 1 and self._pool is None:
            self._pool = multiprocessing.Pool(
                processes,
                initializer=self.initializer,
                initargs=self.initargs,
                maxtasksperchild=self.maxtasksperchild)

    def close(self):
        """ End the multiprocessing task pool """
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        self._pool = None

    def __enter__(self) -> 'Hypermap':
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def map(self, function: Callable, iterable: Iterable) -> Iterable:
        """ Map the function over the iterable

        :param func function:
            Function to call on every item in the iterable
        :param iter iterable:
            The iterable to call map over
        :returns:
            The mapped values
        """
        if self.wrapper == 'dict':
            function = MaybeProcessDict(function)
        elif self.wrapper is not None:
            raise ValueError(f'Unknown wrapper function "{self.wrapper}"')

        if self.processes > 1:
            if self.lazy:
                return self._pool.imap_unordered(function, iterable)
            else:
                # Have to do async + get to prevent a timeout bug
                return self._pool.map_async(function, iterable).get()
        else:
            # Simulate the sequence of initialize once, then map
            if self.initializer is not None:
                self.initializer(*self.initargs)
            if self.lazy:
                return map(function, iterable)
            else:
                return list(map(function, iterable))


class MaybeProcessDict(object):
    """ Wrap the function in an exception handler

    :param callable func:
        The function to wrap. Item is assumed to be a dict that is translated to \\*\\*kwargs
    :returns:
        A function that converts exceptions to a True return code for success,
        False for failure
    """

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, item: Dict[str, Any]) -> bool:
        try:
            self.func(**item)
        except Exception:
            traceback.print_exc()
            return False
        return True
