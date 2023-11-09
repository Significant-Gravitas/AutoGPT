# A sample distutils script to show to build your own
# extension module which extends pywintypes or pythoncom.
#
# Use 'python setup.py build' to build this extension.
import os
from distutils.core import Extension, setup
from sysconfig import get_paths

sources = ["win32_extension.cpp"]
lib_dir = get_paths()["platlib"]

# Specify the directory where the PyWin32 .h and .lib files are installed.
# If you are doing a win32com extension, you will also need to add
# win32com\Include and win32com\Libs.
ext = Extension(
    "win32_extension",
    sources,
    include_dirs=[os.path.join(lib_dir, "win32", "include")],
    library_dirs=[os.path.join(lib_dir, "win32", "libs")],
)

setup(
    name="win32 extension sample",
    version="0.1",
    ext_modules=[ext],
)
