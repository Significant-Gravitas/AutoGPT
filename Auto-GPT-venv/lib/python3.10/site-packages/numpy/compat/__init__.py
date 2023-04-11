"""
Compatibility module.

This module contains duplicated code from Python itself or 3rd party
extensions, which may be included for the following reasons:

  * compatibility
  * we may only need a small subset of the copied library/module

"""
from . import _inspect
from . import py3k
from ._inspect import getargspec, formatargspec
from .py3k import *

__all__ = []
__all__.extend(_inspect.__all__)
__all__.extend(py3k.__all__)
