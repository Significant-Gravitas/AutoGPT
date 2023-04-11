"""
.. deprecated:: 1.20

*This module is deprecated.  Instead of importing functions from*
``numpy.dual``, *the functions should be imported directly from NumPy
or SciPy*.

Aliases for functions which may be accelerated by SciPy.

SciPy_ can be built to use accelerated or otherwise improved libraries
for FFTs, linear algebra, and special functions. This module allows
developers to transparently support these accelerated functions when
SciPy is available but still support users who have only installed
NumPy.

.. _SciPy : https://www.scipy.org

"""
import warnings


warnings.warn('The module numpy.dual is deprecated.  Instead of using dual, '
              'use the functions directly from numpy or scipy.',
              category=DeprecationWarning,
              stacklevel=2)

# This module should be used for functions both in numpy and scipy if
#  you want to use the numpy version if available but the scipy version
#  otherwise.
#  Usage  --- from numpy.dual import fft, inv

__all__ = ['fft', 'ifft', 'fftn', 'ifftn', 'fft2', 'ifft2',
           'norm', 'inv', 'svd', 'solve', 'det', 'eig', 'eigvals',
           'eigh', 'eigvalsh', 'lstsq', 'pinv', 'cholesky', 'i0']

import numpy.linalg as linpkg
import numpy.fft as fftpkg
from numpy.lib import i0
import sys


fft = fftpkg.fft
ifft = fftpkg.ifft
fftn = fftpkg.fftn
ifftn = fftpkg.ifftn
fft2 = fftpkg.fft2
ifft2 = fftpkg.ifft2

norm = linpkg.norm
inv = linpkg.inv
svd = linpkg.svd
solve = linpkg.solve
det = linpkg.det
eig = linpkg.eig
eigvals = linpkg.eigvals
eigh = linpkg.eigh
eigvalsh = linpkg.eigvalsh
lstsq = linpkg.lstsq
pinv = linpkg.pinv
cholesky = linpkg.cholesky

_restore_dict = {}

def register_func(name, func):
    if name not in __all__:
        raise ValueError("{} not a dual function.".format(name))
    f = sys._getframe(0).f_globals
    _restore_dict[name] = f[name]
    f[name] = func

def restore_func(name):
    if name not in __all__:
        raise ValueError("{} not a dual function.".format(name))
    try:
        val = _restore_dict[name]
    except KeyError:
        return
    else:
        sys._getframe(0).f_globals[name] = val

def restore_all():
    for name in _restore_dict.keys():
        restore_func(name)
