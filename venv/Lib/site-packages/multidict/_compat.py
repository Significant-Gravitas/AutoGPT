import os
import platform

NO_EXTENSIONS = bool(os.environ.get("MULTIDICT_NO_EXTENSIONS"))

PYPY = platform.python_implementation() == "PyPy"

USE_EXTENSIONS = not NO_EXTENSIONS and not PYPY

if USE_EXTENSIONS:
    try:
        from . import _multidict  # noqa
    except ImportError:
        USE_EXTENSIONS = False
