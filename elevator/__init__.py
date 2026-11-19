"""Latest elevator perception and motion workflow. Hardware opens only on explicit execution."""
from contextlib import contextmanager


_verbose = True


@contextmanager
def terminal_output(verbose):
    global _verbose
    previous, _verbose = _verbose, verbose
    try:
        yield
    finally:
        _verbose = previous


def terminal_verbose():
    return _verbose


def detail(*args, **kwargs):
    if _verbose:
        print(*args, **kwargs)
