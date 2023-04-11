"""
**Note:** almost all functions in the ``numpy.lib`` namespace
are also present in the main ``numpy`` namespace.  Please use the
functions as ``np.<funcname>`` where possible.

``numpy.lib`` is mostly a space for implementing functions that don't
belong in core or in another NumPy submodule with a clear purpose
(e.g. ``random``, ``fft``, ``linalg``, ``ma``).

Most contains basic functions that are used by several submodules and are
useful to have in the main name-space.

"""
import math

from numpy.version import version as __version__

# Public submodules
# Note: recfunctions and (maybe) format are public too, but not imported
from . import mixins
from . import scimath as emath

# Private submodules
# load module names. See https://github.com/networkx/networkx/issues/5838
from . import type_check
from . import index_tricks
from . import function_base
from . import nanfunctions
from . import shape_base
from . import stride_tricks
from . import twodim_base
from . import ufunclike
from . import histograms
from . import polynomial
from . import utils
from . import arraysetops
from . import npyio
from . import arrayterator
from . import arraypad
from . import _version

from .type_check import *
from .index_tricks import *
from .function_base import *
from .nanfunctions import *
from .shape_base import *
from .stride_tricks import *
from .twodim_base import *
from .ufunclike import *
from .histograms import *

from .polynomial import *
from .utils import *
from .arraysetops import *
from .npyio import *
from .arrayterator import Arrayterator
from .arraypad import *
from ._version import *
from numpy.core._multiarray_umath import tracemalloc_domain

__all__ = ['emath', 'math', 'tracemalloc_domain', 'Arrayterator']
__all__ += type_check.__all__
__all__ += index_tricks.__all__
__all__ += function_base.__all__
__all__ += shape_base.__all__
__all__ += stride_tricks.__all__
__all__ += twodim_base.__all__
__all__ += ufunclike.__all__
__all__ += arraypad.__all__
__all__ += polynomial.__all__
__all__ += utils.__all__
__all__ += arraysetops.__all__
__all__ += npyio.__all__
__all__ += nanfunctions.__all__
__all__ += histograms.__all__

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
