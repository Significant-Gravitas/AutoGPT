import os
from pathlib import Path


def is_path(f):
    return isinstance(f, (bytes, str, Path))


def is_directory(f):
    """Checks if an object is a string, and that it points to a directory."""
    return is_path(f) and os.path.isdir(f)


class DeferredError:
    def __init__(self, ex):
        self.ex = ex

    def __getattr__(self, elt):
        raise self.ex
