from warnings import warn

from .std import TqdmDeprecationWarning
from .utils import (  # NOQA, pylint: disable=unused-import
    CUR_OS, IS_NIX, IS_WIN, RE_ANSI, Comparable, FormatReplace, SimpleTextIOWrapper,
    _environ_cols_wrapper, _is_ascii, _is_utf, _screen_shape_linux, _screen_shape_tput,
    _screen_shape_windows, _screen_shape_wrapper, _supports_unicode, _term_move_up, colorama)

warn("This function will be removed in tqdm==5.0.0\n"
     "Please use `tqdm.utils.*` instead of `tqdm._utils.*`",
     TqdmDeprecationWarning, stacklevel=2)
