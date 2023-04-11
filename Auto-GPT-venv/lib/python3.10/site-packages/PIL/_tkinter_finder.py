""" Find compiled module linking to Tcl / Tk libraries
"""
import sys
import tkinter
from tkinter import _tkinter as tk

from ._deprecate import deprecate

try:
    if hasattr(sys, "pypy_find_executable"):
        TKINTER_LIB = tk.tklib_cffi.__file__
    else:
        TKINTER_LIB = tk.__file__
except AttributeError:
    # _tkinter may be compiled directly into Python, in which case __file__ is
    # not available. load_tkinter_funcs will check the binary first in any case.
    TKINTER_LIB = None

tk_version = str(tkinter.TkVersion)
if tk_version == "8.4":
    deprecate(
        "Support for Tk/Tcl 8.4", 10, action="Please upgrade to Tk/Tcl 8.5 or newer"
    )
