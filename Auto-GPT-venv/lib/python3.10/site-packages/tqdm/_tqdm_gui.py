from warnings import warn

from .gui import *  # NOQA
from .gui import __all__  # NOQA
from .std import TqdmDeprecationWarning

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.gui.*` instead of `tqdm._tqdm_gui.*`",
     TqdmDeprecationWarning, stacklevel=2)
