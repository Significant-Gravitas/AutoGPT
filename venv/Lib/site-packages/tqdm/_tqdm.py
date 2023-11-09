from warnings import warn

from .std import *  # NOQA
from .std import __all__  # NOQA
from .std import TqdmDeprecationWarning

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.std.*` instead of `tqdm._tqdm.*`",
     TqdmDeprecationWarning, stacklevel=2)
