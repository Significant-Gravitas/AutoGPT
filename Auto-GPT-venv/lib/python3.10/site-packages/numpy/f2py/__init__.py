#!/usr/bin/env python3
"""Fortran to Python Interface Generator.

"""
__all__ = ['run_main', 'compile', 'get_include']

import sys
import subprocess
import os

from . import f2py2e
from . import diagnose

run_main = f2py2e.run_main
main = f2py2e.main


def compile(source,
            modulename='untitled',
            extra_args='',
            verbose=True,
            source_fn=None,
            extension='.f',
            full_output=False
           ):
    """
    Build extension module from a Fortran 77 source string with f2py.

    Parameters
    ----------
    source : str or bytes
        Fortran source of module / subroutine to compile

        .. versionchanged:: 1.16.0
           Accept str as well as bytes

    modulename : str, optional
        The name of the compiled python module
    extra_args : str or list, optional
        Additional parameters passed to f2py

        .. versionchanged:: 1.16.0
            A list of args may also be provided.

    verbose : bool, optional
        Print f2py output to screen
    source_fn : str, optional
        Name of the file where the fortran source is written.
        The default is to use a temporary file with the extension
        provided by the ``extension`` parameter
    extension : ``{'.f', '.f90'}``, optional
        Filename extension if `source_fn` is not provided.
        The extension tells which fortran standard is used.
        The default is ``.f``, which implies F77 standard.

        .. versionadded:: 1.11.0

    full_output : bool, optional
        If True, return a `subprocess.CompletedProcess` containing
        the stdout and stderr of the compile process, instead of just
        the status code.

        .. versionadded:: 1.20.0


    Returns
    -------
    result : int or `subprocess.CompletedProcess`
        0 on success, or a `subprocess.CompletedProcess` if
        ``full_output=True``

    Examples
    --------
    .. literalinclude:: ../../source/f2py/code/results/compile_session.dat
        :language: python

    """
    import tempfile
    import shlex

    if source_fn is None:
        f, fname = tempfile.mkstemp(suffix=extension)
        # f is a file descriptor so need to close it
        # carefully -- not with .close() directly
        os.close(f)
    else:
        fname = source_fn

    if not isinstance(source, str):
        source = str(source, 'utf-8')
    try:
        with open(fname, 'w') as f:
            f.write(source)

        args = ['-c', '-m', modulename, f.name]

        if isinstance(extra_args, str):
            is_posix = (os.name == 'posix')
            extra_args = shlex.split(extra_args, posix=is_posix)

        args.extend(extra_args)

        c = [sys.executable,
             '-c',
             'import numpy.f2py as f2py2e;f2py2e.main()'] + args
        try:
            cp = subprocess.run(c, capture_output=True)
        except OSError:
            # preserve historic status code used by exec_command()
            cp = subprocess.CompletedProcess(c, 127, stdout=b'', stderr=b'')
        else:
            if verbose:
                print(cp.stdout.decode())
    finally:
        if source_fn is None:
            os.remove(fname)

    if full_output:
        return cp
    else:
        return cp.returncode


def get_include():
    """
    Return the directory that contains the ``fortranobject.c`` and ``.h`` files.

    .. note::

        This function is not needed when building an extension with
        `numpy.distutils` directly from ``.f`` and/or ``.pyf`` files
        in one go.

    Python extension modules built with f2py-generated code need to use
    ``fortranobject.c`` as a source file, and include the ``fortranobject.h``
    header. This function can be used to obtain the directory containing
    both of these files.

    Returns
    -------
    include_path : str
        Absolute path to the directory containing ``fortranobject.c`` and
        ``fortranobject.h``.

    Notes
    -----
    .. versionadded:: 1.21.1

    Unless the build system you are using has specific support for f2py,
    building a Python extension using a ``.pyf`` signature file is a two-step
    process. For a module ``mymod``:

    * Step 1: run ``python -m numpy.f2py mymod.pyf --quiet``. This
      generates ``_mymodmodule.c`` and (if needed)
      ``_fblas-f2pywrappers.f`` files next to ``mymod.pyf``.
    * Step 2: build your Python extension module. This requires the
      following source files:

      * ``_mymodmodule.c``
      * ``_mymod-f2pywrappers.f`` (if it was generated in Step 1)
      * ``fortranobject.c``

    See Also
    --------
    numpy.get_include : function that returns the numpy include directory

    """
    return os.path.join(os.path.dirname(__file__), 'src')


def __getattr__(attr):

    # Avoid importing things that aren't needed for building
    # which might import the main numpy module
    if attr == "test":
        from numpy._pytesttester import PytestTester
        test = PytestTester(__name__)
        return test

    else:
        raise AttributeError("module {!r} has no attribute "
                              "{!r}".format(__name__, attr))


def __dir__():
    return list(globals().keys() | {"test"})
